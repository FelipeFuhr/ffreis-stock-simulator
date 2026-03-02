from __future__ import annotations

from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import float64 as np_float64
from numpy import int8 as np_int8
from numpy import int32 as np_int32
from numpy import testing as np_testing

from stock_simulator.config import GameConfig
from stock_simulator.core import CoreState, CoreStepOutput, MarketArrays, initial_core_state, step_core, step_core_numba
from stock_simulator.types import Action


def _market_arrays() -> MarketArrays:
    open_ = np_asarray([100.0, 101.0, 102.0, 103.0], dtype=np_float32)
    high = np_asarray([101.0, 102.0, 103.0, 104.0], dtype=np_float32)
    low = np_asarray([99.0, 100.0, 101.0, 102.0], dtype=np_float32)
    close = np_asarray([100.0, 101.0, 102.0, 103.0], dtype=np_float32)
    return MarketArrays(open=open_, high=high, low=low, close=close, n=4)


def test_step_core_done_short_circuit() -> None:
    base = initial_core_state(initial_cash=1000.0, max_orders=2)
    done_state = CoreState(
        t=base.t,
        done=True,
        portfolio=base.portfolio,
        order_active=base.order_active,
        order_side=base.order_side,
        order_type=base.order_type,
        order_units=base.order_units,
        order_limit_price=base.order_limit_price,
        order_eligible_t=base.order_eligible_t,
        order_ttl=base.order_ttl,
    )
    result = step_core(
        done_state,
        Action(side="hold"),
        _market_arrays(),
        GameConfig(use_numba=False),
        np_asarray([1.0, 1.0], dtype=np_float64),
    )
    assert isinstance(result, CoreStepOutput)
    assert result.state.done is True
    assert result.fills == 0
    assert result.equity_delta == 0.0


def test_step_core_rejects_invalid_units() -> None:
    state = initial_core_state(initial_cash=1000.0, max_orders=2)
    try:
        step_core(
            state,
            Action(side="buy", units=0.0, order_type="market"),
            _market_arrays(),
            GameConfig(use_numba=False),
            np_asarray([1.0, 1.0], dtype=np_float64),
        )
        raise AssertionError("expected ValueError for non-positive units")
    except ValueError as exc:
        assert "units must be > 0" in str(exc)


def test_step_core_rejects_order_capacity_exceeded() -> None:
    state = initial_core_state(initial_cash=1000.0, max_orders=1)
    state.order_active[0] = np_int8(1)
    state.order_side[0] = np_int8(-1)
    try:
        step_core(
            state,
            Action(side="buy", units=1.0, order_type="market"),
            _market_arrays(),
            GameConfig(use_numba=False),
            np_asarray([1.0], dtype=np_float64),
        )
        raise AssertionError("expected ValueError for capacity exceeded")
    except ValueError as exc:
        assert "order capacity exceeded" in str(exc)


def test_step_core_expires_unfilled_limit_order() -> None:
    state = initial_core_state(initial_cash=1000.0, max_orders=1)
    state.order_active[0] = np_int8(1)
    state.order_side[0] = np_int8(1)
    state.order_type[0] = np_int8(1)  # limit
    state.order_units[0] = 1.0
    state.order_limit_price[0] = 50.0  # far below low -> not filled
    state.order_eligible_t[0] = np_int32(0)
    state.order_ttl[0] = np_int32(1)

    result = step_core(
        state,
        Action(side="hold"),
        _market_arrays(),
        GameConfig(use_numba=False, limit_ttl_bars=1),
        np_asarray([1.0], dtype=np_float64),
    )
    assert result.fills == 0
    assert result.state.order_active[0] == 0
    assert result.state.order_ttl[0] == 0


def test_step_core_numba_matches_python_path_for_simple_market_order() -> None:
    state_py = initial_core_state(initial_cash=1000.0, max_orders=2)
    state_nb = initial_core_state(initial_cash=1000.0, max_orders=2)
    config = GameConfig(
        use_numba=True,
        market_latency_bars=0,
        limit_ttl_bars=2,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    random_draws = np_asarray([1.0, 1.0], dtype=np_float64)
    action = Action(side="buy", units=1.0, order_type="market")

    py_result = step_core(state_py, action, _market_arrays(), config, random_draws)
    nb_result = step_core_numba(state_nb, action, _market_arrays(), config, random_draws)

    assert py_result.state.t == nb_result.state.t
    assert py_result.state.done == nb_result.state.done
    assert py_result.fills == nb_result.fills
    np_testing.assert_allclose(py_result.state.portfolio, nb_result.state.portfolio)
