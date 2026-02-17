from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from .config import GameConfig
from .core import (
    CoreState,
    MarketArrays,
    initial_core_state,
    step_core,
    step_core_jit,
    step_core_numba,
)
from .data import MarketData
from .execution import summarize_execution
from .orders import summarize_orders
from .portal import MarketPortal
from .portfolio import snapshot_from_state
from .recorder import NullRecorder, Recorder
from .telemetry import get_telemetry
from .types import Action, EnvState, Observation, OrderType, Side, StepResult


class MarketEnv:
    """Stateful stock simulation environment.

    Notes
    -----
    Public engine API is intentionally limited to ``reset``, ``step``,
    ``observe``, and optional ``step_many``.
    """

    def __init__(
        self,
        data: MarketData,
        cfg: GameConfig,
        recorder: Recorder | None = None,
    ):
        self._market_arrays = MarketArrays(
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            n=data.n,
        )
        self._cfg = cfg
        self._rng = default_rng(cfg.seed)
        self._portal = MarketPortal(
            market_arrays=self._market_arrays,
            observation_window=self._cfg.observation_window,
        )
        self._core_state = initial_core_state(
            initial_cash=self._cfg.initial_cash,
            max_orders=self._cfg.max_open_orders,
        )
        self._telemetry = get_telemetry()
        self._config_hash = self._cfg.stable_hash()
        self._telemetry.set_config_hash(self._config_hash)
        self._recorder = recorder if recorder is not None else NullRecorder()

    def reset(self, seed: int | None = None) -> EnvState:
        """Reset the environment and return the initial state.

        Parameters
        ----------
        seed
            Optional seed for deterministic replay.

        Returns
        -------
        EnvState
            Initial state snapshot.
        """
        actual_seed: int
        if seed is not None:
            self._rng = default_rng(seed)
            actual_seed = seed
        else:
            self._rng = default_rng(self._cfg.seed)
            actual_seed = self._cfg.seed
        self._core_state = initial_core_state(
            initial_cash=self._cfg.initial_cash,
            max_orders=self._cfg.max_open_orders,
        )
        self._recorder.start_episode(actual_seed)
        return self._to_env_state(self._core_state)

    def step(self, action: Action) -> StepResult:
        """Advance the simulator by one action.

        Parameters
        ----------
        action
            Validated environment action.

        Returns
        -------
        StepResult
            Next state, observation, and done flag.
        """
        if self._core_state.done:
            observation = self.observe()
            return StepResult(
                state=self._to_env_state(self._core_state),
                observation=observation,
                done=True,
            )

        with self._telemetry.step_span(
            use_numba=self._cfg.use_numba,
            action_side=action.side,
            action_type=action.order_type,
        ):
            if action.side != "hold":
                self._telemetry.on_order(
                    side=action.side,
                    order_type=action.order_type,
                    units=action.units,
                )
            with self._telemetry.child_span(
                "orders.match",
                {
                    "sim.use_numba": self._cfg.use_numba,
                    "sim.action_side": action.side,
                    "sim.action_type": action.order_type,
                },
            ):
                random_draws = self._next_random_draws()
                if self._cfg.use_numba:
                    output = step_core_numba(
                        state=self._core_state,
                        action=action,
                        market_arrays=self._market_arrays,
                        config=self._cfg,
                        random_draws=random_draws,
                    )
                else:
                    output = step_core(
                        state=self._core_state,
                        action=action,
                        market_arrays=self._market_arrays,
                        config=self._cfg,
                        random_draws=random_draws,
                    )
            execution = summarize_execution(output)
            self._core_state = output.state
            with self._telemetry.child_span(
                "orders.fill",
                {
                    "sim.has_fill": execution.fills > 0,
                    "sim.fill_bucket": (
                        "none"
                        if execution.fills == 0
                        else "single"
                        if execution.fills == 1
                        else "multi"
                    ),
                },
            ):
                if execution.fills > 0:
                    self._telemetry.on_fill(count=execution.fills)

            observation = self.observe()
            state = self._to_env_state(self._core_state)
            self._recorder.on_step(
                step=state.t,
                action=action,
                fills=execution.fills,
                equity=observation.equity,
                leverage=observation.leverage,
                price=observation.price,
            )
            self._telemetry.on_step(
                equity=observation.equity,
                equity_delta=execution.equity_delta,
                use_numba=self._cfg.use_numba,
                config_hash=self._config_hash,
            )
            if execution.done:
                self._recorder.end_episode()
                self._telemetry.on_episode_end(
                    steps=output.state.t + 1,
                    final_equity=observation.equity,
                )
            return StepResult(state=state, observation=observation, done=self._core_state.done)

    def step_many(
        self, actions: NDArray[np.float64]
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.bool_]]:
        """Execute multiple encoded actions in a tight loop.

        Parameters
        ----------
        actions
            Matrix with shape ``[num_steps, 4]`` encoded as
            ``[side_code, units, order_type_code, limit_price_or_nan]``.

        Returns
        -------
        tuple[dict[str, numpy.ndarray], numpy.ndarray, numpy.ndarray]
            Stacked observations, rewards, and done flags.
        """
        if actions.ndim != 2 or actions.shape[1] != 4:
            raise ValueError("actions must have shape [num_steps, 4]")

        num_steps = int(actions.shape[0])
        market_handles = np.zeros((num_steps, 4), dtype=np.float64)
        portfolio_vectors = np.zeros((num_steps, 4), dtype=np.float64)
        order_summary_vectors = np.zeros((num_steps, 3), dtype=np.float64)
        rewards = np.zeros(num_steps, dtype=np.float64)
        dones = np.zeros(num_steps, dtype=np.bool_)
        random_draws_batch = self._rng.uniform(
            self._cfg.partial_fill_min,
            self._cfg.partial_fill_max,
            size=(num_steps, self._cfg.max_open_orders),
        ).astype(np.float64, copy=False)

        # Intentionally bypass telemetry/recorder callbacks for headless bulk execution.
        for i in range(num_steps):
            if self._core_state.done:
                observation = self.observe()
                tensors = observation.to_numpy_tensors()
                market_handles[i] = tensors["market_window_handle"]
                portfolio_vectors[i] = tensors["portfolio_vector"]
                order_summary_vectors[i] = tensors["order_summary_vector"]
                rewards[i] = 0.0
                dones[i] = True
                continue

            action_side_code, action_units, action_order_code, action_limit = actions[i]
            if self._cfg.use_numba:
                (
                    next_t,
                    next_done,
                    next_portfolio,
                    next_order_active,
                    next_order_side,
                    next_order_type,
                    next_order_units,
                    next_order_limit_price,
                    next_order_eligible_t,
                    next_order_ttl,
                    _fills,
                    equity_delta,
                ) = step_core_jit(
                    t=self._core_state.t,
                    done=self._core_state.done,
                    portfolio=self._core_state.portfolio,
                    order_active=self._core_state.order_active,
                    order_side=self._core_state.order_side,
                    order_type=self._core_state.order_type,
                    order_units=self._core_state.order_units,
                    order_limit_price=self._core_state.order_limit_price,
                    order_eligible_t=self._core_state.order_eligible_t,
                    order_ttl=self._core_state.order_ttl,
                    action_side=np.int8(int(action_side_code)),
                    action_units=float(action_units),
                    action_order_type=np.int8(int(action_order_code)),
                    action_limit_price=(
                        0.0 if np.isnan(action_limit) else float(action_limit)
                    ),
                    market_open=self._market_arrays.open,
                    market_high=self._market_arrays.high,
                    market_low=self._market_arrays.low,
                    market_close=self._market_arrays.close,
                    market_n=self._market_arrays.n,
                    market_latency_bars=self._cfg.market_latency_bars,
                    limit_ttl_bars=self._cfg.limit_ttl_bars,
                    random_draws=random_draws_batch[i],
                    fee_bps=self._cfg.fee_bps,
                    slippage_bps=self._cfg.slippage_bps,
                )
                self._core_state = CoreState(
                    t=int(next_t),
                    done=bool(next_done),
                    portfolio=next_portfolio,
                    order_active=next_order_active,
                    order_side=next_order_side,
                    order_type=next_order_type,
                    order_units=next_order_units,
                    order_limit_price=next_order_limit_price,
                    order_eligible_t=next_order_eligible_t,
                    order_ttl=next_order_ttl,
                )
                rewards[i] = float(equity_delta)
            else:
                action = self._action_from_encoded_row(actions[i])
                output = step_core(
                    state=self._core_state,
                    action=action,
                    market_arrays=self._market_arrays,
                    config=self._cfg,
                    random_draws=random_draws_batch[i],
                )
                self._core_state = output.state
                rewards[i] = float(output.equity_delta)

            observation = self.observe()
            tensors = observation.to_numpy_tensors()
            market_handles[i] = tensors["market_window_handle"]
            portfolio_vectors[i] = tensors["portfolio_vector"]
            order_summary_vectors[i] = tensors["order_summary_vector"]
            dones[i] = self._core_state.done

        return (
            {
                "market_window_handle": market_handles,
                "portfolio_vector": portfolio_vectors,
                "order_summary_vector": order_summary_vectors,
            },
            rewards,
            dones,
        )

    def observe(self) -> Observation:
        """Build current observation from internal engine state.

        Returns
        -------
        Observation
            Current observation object.
        """
        market_handle = self._portal.view_handle(self._core_state.t)
        portfolio = snapshot_from_state(self._core_state, market_handle.current_price)
        orders = summarize_orders(self._core_state)
        return Observation(
            market=market_handle,
            portfolio_vector=portfolio.to_vector(),
            order_summary_vector=orders.to_vector(),
            done=self._core_state.done,
        )

    def _to_env_state(self, state: CoreState) -> EnvState:
        portfolio = snapshot_from_state(state, self._portal.current_price(state.t))
        orders = summarize_orders(state)
        return EnvState(
            t=state.t,
            cash=portfolio.cash,
            units=portfolio.units,
            equity=portfolio.equity,
            leverage=portfolio.leverage,
            open_orders=orders.open_orders,
            done=state.done,
        )

    def _next_random_draws(self) -> NDArray[np.float64]:
        return self._rng.uniform(
            self._cfg.partial_fill_min,
            self._cfg.partial_fill_max,
            size=self._cfg.max_open_orders,
        ).astype(np.float64, copy=False)

    def _action_from_encoded_row(self, row: NDArray[np.float64]) -> Action:
        side_code = int(row[0])
        units = float(row[1])
        order_code = int(row[2])
        limit_price = None if np.isnan(row[3]) else float(row[3])

        if side_code == 0:
            return Action(side="hold")
        side: Side
        if side_code == 1:
            side = "buy"
        elif side_code == -1:
            side = "sell"
        else:
            raise ValueError("invalid side code; use -1 (sell), 0 (hold), 1 (buy)")

        order_type: OrderType
        if order_code == 0:
            order_type = "market"
        elif order_code == 1:
            order_type = "limit"
        else:
            raise ValueError("invalid order_type code; use 0 (market), 1 (limit)")

        return Action(
            side=side,
            units=units,
            order_type=order_type,
            limit_price=limit_price,
        )
