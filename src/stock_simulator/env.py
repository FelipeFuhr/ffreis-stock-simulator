from __future__ import annotations

from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from numpy import int8 as np_int8
from numpy import isnan as np_isnan
from numpy import zeros as np_zeros
from numpy.random import default_rng
from numpy.typing import NDArray

from .config import GameConfig
from .core import (
    CoreState,
    MarketArrays,
    _exec_price_from_jit,
    _fill_slot_from_jit,
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
from .trace_writer import TraceJsonlWriter
from .types import (
    Action,
    EnvState,
    MarketWindowContent,
    Observation,
    OrderType,
    Side,
    StepResult,
    StepTraceRow,
)


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
        trace_sink: TraceJsonlWriter | None = None,
    ):
        self._market_arrays = MarketArrays(
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            n=data.n,
        )
        # Volume is deliberately kept out of MarketArrays (the numba step core) and
        # carried alongside it so the observation surface can expose it unchanged.
        self._volume = data.volume
        self._cfg = cfg
        self._rng = default_rng(cfg.seed)
        self._portal = MarketPortal(
            market_arrays=self._market_arrays,
            observation_window=self._cfg.observation_window,
            volume=self._volume,
        )
        self._core_state = initial_core_state(
            initial_cash=self._cfg.initial_cash,
            max_orders=self._cfg.max_open_orders,
        )
        self._telemetry = get_telemetry()
        self._config_hash = self._cfg.stable_hash()
        self._telemetry.set_config_hash(self._config_hash)
        self._recorder = recorder if recorder is not None else NullRecorder()
        # N10 debugging/verification aid: when set, every step_many trace row is
        # written to this sink regardless of the per-request include_trace flag
        # (see STOCK_SIM_TRACE_JSONL in server.py). Distinct from Recorder above,
        # which is the structured episode-replay port and is never wired into
        # step_many's headless bulk-execution path by design.
        self._trace_sink = trace_sink

    def close(self) -> None:
        """Release resources this environment owns (e.g. a configured trace sink)."""
        if self._trace_sink is not None:
            self._trace_sink.close()

    def reset(self, seed: int | None = None, start_t: int = 0) -> EnvState:
        """Reset the environment and return the initial state.

        Parameters
        ----------
        seed
            Optional seed for deterministic replay.
        start_t
            Bar index to start the episode at. Defaults to ``0``, matching the
            prior always-zero behavior. Lets a caller begin an episode after
            enough history has accumulated for indicator warm-up (see
            :meth:`market_window`) or run walk-forward training over different
            slices of the same loaded market data. ``reset(seed=S,
            start_t=T)`` is deterministic: calling it twice with the same
            ``(seed, start_t)`` pair produces byte-identical episodes.

        Returns
        -------
        EnvState
            Initial state snapshot.

        Raises
        ------
        ValueError
            If ``start_t`` is outside ``[0, n)``, where ``n`` is the number of
            loaded market bars.
        """
        n = self._market_arrays.n
        if not 0 <= start_t < n:
            raise ValueError(f"start_t must satisfy 0 <= start_t < {n} (market has {n} bars); got {start_t}")
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
            start_t=start_t,
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
            if execution.fills == 0:
                fill_bucket = "none"
            elif execution.fills == 1:
                fill_bucket = "single"
            else:
                fill_bucket = "multi"
            with self._telemetry.child_span(
                "orders.fill",
                {
                    "sim.has_fill": execution.fills > 0,
                    "sim.fill_bucket": fill_bucket,
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
        self,
        actions: NDArray[np_float64],
        *,
        include_trace: bool = False,
    ) -> tuple[dict[str, NDArray[np_float64]], NDArray[np_float64], NDArray[np_bool_], tuple[StepTraceRow, ...]]:
        """Execute multiple encoded actions in a tight loop.

        Parameters
        ----------
        actions
            Matrix with shape ``[num_steps, 4]`` encoded as
            ``[side_code, units, order_type_code, limit_price_or_nan]``.

        Returns
        -------
        tuple[dict[str, numpy.ndarray], numpy.ndarray, numpy.ndarray, tuple[StepTraceRow, ...]]
            Stacked observations, rewards, done flags, and optional trace rows.
        """
        if actions.ndim != 2 or actions.shape[1] != 4:
            raise ValueError("actions must have shape [num_steps, 4]")

        num_steps = int(actions.shape[0])
        market_handles = np_zeros((num_steps, 4), dtype=np_float64)
        portfolio_vectors = np_zeros((num_steps, 4), dtype=np_float64)
        order_summary_vectors = np_zeros((num_steps, 3), dtype=np_float64)
        rewards = np_zeros(num_steps, dtype=np_float64)
        dones = np_zeros(num_steps, dtype=np_bool_)
        trace_rows: list[StepTraceRow] = []
        random_draws_batch = self._rng.uniform(
            self._cfg.partial_fill_min,
            self._cfg.partial_fill_max,
            size=(num_steps, self._cfg.max_open_orders),
        ).astype(np_float64, copy=False)

        # Intentionally bypass telemetry/recorder callbacks for headless bulk execution.
        for i in range(num_steps):
            action_row = actions[i]
            action_side_code, action_units, action_order_code, action_limit = action_row
            step_fills, rewards[i], filled_order_slot, exec_price = self._advance_one_step(
                action_row, random_draws_batch[i]
            )
            observation = self.observe()
            tensors = observation.to_numpy_tensors()
            market_handles[i] = tensors["market_window_handle"]
            portfolio_vectors[i] = tensors["portfolio_vector"]
            order_summary_vectors[i] = tensors["order_summary_vector"]
            dones[i] = self._core_state.done
            # Trace rows are built whenever a client asked for them (include_trace)
            # or a server-side STOCK_SIM_TRACE_JSONL sink is configured (N10) — the
            # sink captures every step's trace regardless of what any one caller
            # requested; only the include_trace branch ever reaches the response.
            if include_trace or self._trace_sink is not None:
                trace_row = self._build_trace_row(
                    index=i,
                    action_side_code=action_side_code,
                    action_units=action_units,
                    action_order_code=action_order_code,
                    action_limit=action_limit,
                    step_fills=step_fills,
                    reward=float(rewards[i]),
                    done=bool(dones[i]),
                    observation=observation,
                    filled_order_slot=filled_order_slot,
                    exec_price=exec_price,
                )
                if include_trace:
                    trace_rows.append(trace_row)
                if self._trace_sink is not None:
                    self._trace_sink.write(trace_row)

        return (
            {
                "market_window_handle": market_handles,
                "portfolio_vector": portfolio_vectors,
                "order_summary_vector": order_summary_vectors,
            },
            rewards,
            dones,
            tuple(trace_rows),
        )

    def _advance_one_step(
        self,
        action_row: NDArray[np_float64],
        random_draws: NDArray[np_float64],
    ) -> tuple[int, float, int | None, float | None]:
        """Advance the core state by one encoded action row.

        Returns
        -------
        tuple[int, float, int | None, float | None]
            ``(fills, equity_delta, filled_order_slot, exec_price)`` for the step;
            all zero/``None`` when already done. The last two are the N9
            observability fields — see :class:`stock_simulator.core.CoreStepOutput`.
        """
        if self._core_state.done:
            return 0, 0.0, None, None
        action_side_code, action_units, action_order_code, action_limit = action_row
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
                fills,
                equity_delta,
                fill_slot,
                fill_exec_price,
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
                action_side=np_int8(int(action_side_code)),
                action_units=float(action_units),
                action_order_type=np_int8(int(action_order_code)),
                action_limit_price=(0.0 if np_isnan(action_limit) else float(action_limit)),
                market_open=self._market_arrays.open,
                market_high=self._market_arrays.high,
                market_low=self._market_arrays.low,
                market_close=self._market_arrays.close,
                market_n=self._market_arrays.n,
                market_latency_bars=self._cfg.market_latency_bars,
                limit_ttl_bars=self._cfg.limit_ttl_bars,
                random_draws=random_draws,
                fee_bps=self._cfg.fee_bps,
                slippage_bps=self._cfg.slippage_bps,
                max_leverage=self._cfg.max_leverage,
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
            return (
                int(fills),
                float(equity_delta),
                _fill_slot_from_jit(fill_slot),
                _exec_price_from_jit(fill_exec_price),
            )
        action = self._action_from_encoded_row(action_row)
        output = step_core(
            state=self._core_state,
            action=action,
            market_arrays=self._market_arrays,
            config=self._cfg,
            random_draws=random_draws,
        )
        self._core_state = output.state
        return output.fills, float(output.equity_delta), output.filled_order_slot, output.exec_price

    def _build_trace_row(
        self,
        *,
        index: int,
        action_side_code: np_float64,
        action_units: np_float64,
        action_order_code: np_float64,
        action_limit: np_float64,
        step_fills: int,
        reward: float,
        done: bool,
        observation: Observation,
        filled_order_slot: int | None = None,
        exec_price: float | None = None,
    ) -> StepTraceRow:
        """Build a :class:`StepTraceRow` from per-step values."""
        has_limit_price = not np_isnan(action_limit)
        return StepTraceRow(
            index=index,
            side_code=int(action_side_code),
            requested_units=float(action_units),
            order_type_code=int(action_order_code),
            has_limit_price=bool(has_limit_price),
            limit_price=None if not has_limit_price else float(action_limit),
            fills=step_fills,
            reward=reward,
            done=done,
            t=observation.t,
            cash=observation.cash,
            position_units=observation.units,
            equity=observation.equity,
            leverage=observation.leverage,
            open_orders=observation.open_orders,
            market_price=observation.price,
            has_filled_order=filled_order_slot is not None,
            filled_order_slot=filled_order_slot,
            exec_price=exec_price,
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

    def market_window(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> MarketWindowContent:
        """Return raw OHLCV rows for the current market window.

        Reads through to the portal against the engine's current bar index, so
        rows with index greater than ``t`` are never returned. With explicit
        ``start``/``end`` bounds an agent can fetch earlier history (clamped to
        ``[0, t + 1)``) to warm up indicators at episode start.

        Parameters
        ----------
        start, end
            Optional window bounds. When both are omitted the current
            observation window is returned.

        Returns
        -------
        MarketWindowContent
            Window metadata plus per-bar open/high/low/close/volume values.
        """
        return self._portal.window_content(self._core_state.t, start=start, end=end)

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

    def _next_random_draws(self) -> NDArray[np_float64]:
        return self._rng.uniform(
            self._cfg.partial_fill_min,
            self._cfg.partial_fill_max,
            size=self._cfg.max_open_orders,
        ).astype(np_float64, copy=False)

    def _action_from_encoded_row(self, row: NDArray[np_float64]) -> Action:
        side_code = int(row[0])
        units = float(row[1])
        order_code = int(row[2])
        limit_price = None if np_isnan(row[3]) else float(row[3])

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
