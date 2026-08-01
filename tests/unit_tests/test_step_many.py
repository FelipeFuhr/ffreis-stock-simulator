from __future__ import annotations

from collections.abc import Callable

import pytest
from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from numpy import ndarray as np_ndarray
from numpy import testing as np_testing
from numpy.typing import NDArray

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.types import Action


def test_step_many_shapes_and_step_parity(
    market_data_factory: Callable[..., MarketData],
    encode_actions: Callable[[list[Action]], NDArray[np_float64]],
) -> None:
    actions = [
        Action(side="hold"),
        Action(side="buy", units=3.0, order_type="market"),
        Action(side="sell", units=2.0, order_type="limit", limit_price=105.0),
        Action(side="buy", units=1.0, order_type="limit", limit_price=101.0),
        Action(side="hold"),
        Action(side="sell", units=1.0, order_type="market"),
    ]
    encoded = encode_actions(actions)

    cfg = GameConfig(seed=999, use_numba=False)
    env_many = MarketEnv(
        data=market_data_factory(n=256, slope=0.2, spread=0.3, volume=20_000.0),
        cfg=cfg,
    )
    env_seq = MarketEnv(
        data=market_data_factory(n=256, slope=0.2, spread=0.3, volume=20_000.0),
        cfg=cfg,
    )

    env_many.reset(seed=999)
    seq_state = env_seq.reset(seed=999)
    prev_equity = seq_state.equity

    obs_stack, rewards_many, dones_many, trace_rows = env_many.step_many(encoded)

    rewards_seq: list[float] = []
    dones_seq: list[bool] = []
    portfolio_seq: list[np_ndarray] = []
    market_seq: list[np_ndarray] = []
    orders_seq: list[np_ndarray] = []
    for action in actions:
        result = env_seq.step(action)
        rewards_seq.append(result.observation.equity - prev_equity)
        prev_equity = result.observation.equity
        dones_seq.append(result.done)
        tensors = result.observation.to_numpy_tensors()
        market_seq.append(tensors["market_window_handle"])
        portfolio_seq.append(tensors["portfolio_vector"])
        orders_seq.append(tensors["order_summary_vector"])

    assert obs_stack["market_window_handle"].shape == (len(actions), 4)
    assert obs_stack["portfolio_vector"].shape == (len(actions), 4)
    assert obs_stack["order_summary_vector"].shape == (len(actions), 3)
    assert rewards_many.shape == (len(actions),)
    assert dones_many.shape == (len(actions),)
    assert len(trace_rows) == 0

    np_testing.assert_allclose(
        obs_stack["market_window_handle"],
        np_asarray(market_seq, dtype=np_float64),
    )
    np_testing.assert_allclose(
        obs_stack["portfolio_vector"],
        np_asarray(portfolio_seq, dtype=np_float64),
    )
    np_testing.assert_allclose(
        obs_stack["order_summary_vector"],
        np_asarray(orders_seq, dtype=np_float64),
    )
    np_testing.assert_array_equal(dones_many, np_asarray(dones_seq, dtype=np_bool_))
    np_testing.assert_allclose(rewards_many, np_asarray(rewards_seq, dtype=np_float64))


def test_step_many_include_trace_returns_one_row_per_action(
    market_data_factory: Callable[..., MarketData],
    encode_actions: Callable[[list[Action]], NDArray[np_float64]],
) -> None:
    actions = [
        Action(side="hold"),
        Action(side="buy", units=2.0, order_type="market"),
        Action(side="sell", units=1.0, order_type="limit", limit_price=105.0),
    ]
    encoded = encode_actions(actions)
    env = MarketEnv(
        data=market_data_factory(n=128, slope=0.15, spread=0.3, volume=20_000.0),
        cfg=GameConfig(seed=2024, use_numba=False),
    )
    _ = env.reset(seed=2024)

    _, rewards, dones, trace_rows = env.step_many(encoded, include_trace=True)

    assert len(trace_rows) == len(actions)
    assert trace_rows[0].index == 0
    assert trace_rows[1].side_code == 1
    assert trace_rows[2].order_type_code == 1
    assert trace_rows[0].reward == float(rewards[0])
    assert trace_rows[-1].done == bool(dones[-1])


@pytest.mark.parametrize("use_numba", [False, True], ids=["python", "numba"])
def test_delayed_limit_fill_exec_price_matches_equity_delta_effect(
    market_data_factory: Callable[..., MarketData],
    encode_actions: Callable[[list[Action]], NDArray[np_float64]],
    use_numba: bool,
) -> None:
    """N9: recover a delayed-fill limit order's real execution price from the
    trace, and cross-check it against the real economic effect (not just presence
    of a number). A buy limit is submitted at bar 0 but the price doesn't cross
    it until bar 4 — several "hold" steps later — so the submitted action on the
    filling step is itself a hold (its own ``limit_price`` is ``None``); before
    N9 there was no way to recover what price the order actually filled at.
    """
    data = market_data_factory(close=[100.0, 100.0, 100.0, 100.0, 80.0, 80.0], spread=0.5)
    cfg = GameConfig(
        seed=1,
        use_numba=use_numba,
        initial_cash=1000.0,
        max_leverage=10.0,
        market_latency_bars=0,
        fee_bps=0.0,
        slippage_bps=0.0,
        partial_fill_min=1.0,
        partial_fill_max=1.0,
    )
    env = MarketEnv(data=data, cfg=cfg)
    env.reset(seed=cfg.seed)

    actions = [
        Action(side="buy", units=10.0, order_type="limit", limit_price=85.0),
        Action(side="hold"),
        Action(side="hold"),
        Action(side="hold"),
        Action(side="hold"),
    ]
    encoded = encode_actions(actions)
    _, rewards, _, trace_rows = env.step_many(encoded, include_trace=True)

    assert len(trace_rows) == 5
    for row in trace_rows[:4]:
        assert row.fills == 0
        assert row.filled_order_slot is None
        assert row.exec_price is None

    fill_row = trace_rows[4]
    assert fill_row.fills == 1
    assert fill_row.filled_order_slot == 0
    # The submitted action *this* step was "hold" — limit_price alone cannot
    # recover the price the queued order actually filled at.
    assert fill_row.limit_price is None
    assert fill_row.exec_price is not None
    exec_price = fill_row.exec_price
    assert exec_price == pytest.approx(85.0)

    # Cross-check exec_price against the real economic effect: with zero
    # fee/slippage, this step's reward (equity_delta) is exactly the newly
    # opened position's unrealized mark, computed from exec_price — not the
    # submitted limit_price (which is unavailable) and not an arbitrary number.
    expected_reward = 10.0 * (80.0 - exec_price)
    assert float(rewards[4]) == pytest.approx(expected_reward)
    assert fill_row.reward == pytest.approx(expected_reward)
