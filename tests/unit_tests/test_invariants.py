from __future__ import annotations

from collections.abc import Callable
from math import isnan as math_isnan

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest import approx as pytest_approx

from stock_simulator.config import GameConfig
from stock_simulator.core import is_below_maintenance_margin
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.recorder import InMemoryRecorder
from stock_simulator.types import Action, OrderType, Side


@st.composite
def _action_strategy(draw: st.DrawFn) -> Action:
    side: Side = draw(st.sampled_from(["hold", "buy", "sell"]))
    if side == "hold":
        return Action(side="hold")

    order_type: OrderType = draw(st.sampled_from(["market", "limit"]))
    units = draw(
        st.floats(
            min_value=0.01,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    limit_price: float | None = None
    if order_type == "limit":
        limit_price = draw(
            st.floats(
                min_value=80.0,
                max_value=160.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
    return Action(
        side=side,
        units=float(units),
        order_type=order_type,
        limit_price=limit_price,
    )


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(min_value=1, max_value=2_000_000),
    actions=st.lists(_action_strategy(), min_size=10, max_size=180),
)
def test_engine_invariants(
    seed: int,
    actions: list[Action],
    market_data_factory: Callable[..., MarketData],
) -> None:
    recorder = InMemoryRecorder()
    cfg = GameConfig(seed=seed, use_numba=False, max_leverage=3.0)
    env = MarketEnv(
        data=market_data_factory(n=1024, slope=0.05, spread=0.3, volume=25_000.0),
        cfg=cfg,
        recorder=recorder,
    )
    state = env.reset(seed=seed)
    previous_cash = state.cash

    for action in actions:
        result = env.step(action)
        state = result.state
        observation = result.observation
        row = recorder.replay()[-1]

        expected_equity = state.cash + state.units * observation.price
        assert state.equity == pytest_approx(expected_equity)
        assert observation.equity == pytest_approx(expected_equity)

        # `max_leverage` is the INITIAL-margin cap: it bounds what an order may open,
        # not what an open position may drift to as the mark moves. The invariant that
        # actually holds on every step is the MAINTENANCE one — a live book is never
        # carried below its maintenance requirement.
        notional = abs(state.units) * observation.price
        assert not is_below_maintenance_margin(
            state.equity,
            notional,
            cfg.maintenance_margin_rate,
            cfg.maintenance_amount,
        )

        values = (
            state.cash,
            state.units,
            state.equity,
            state.leverage,
            observation.price,
            observation.equity,
            observation.leverage,
        )
        assert all(not math_isnan(v) for v in values)

        if row.fills == 0:
            assert state.cash == pytest_approx(previous_cash)
        previous_cash = state.cash

        if result.done:
            break
