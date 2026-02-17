from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.types import Action


def test_step_many_shapes_and_step_parity(
    market_data_factory: Callable[..., MarketData],
    encode_actions: Callable[[list[Action]], NDArray[np.float64]],
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

    obs_stack, rewards_many, dones_many = env_many.step_many(encoded)

    rewards_seq: list[float] = []
    dones_seq: list[bool] = []
    portfolio_seq: list[np.ndarray] = []
    market_seq: list[np.ndarray] = []
    orders_seq: list[np.ndarray] = []
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

    np.testing.assert_allclose(
        obs_stack["market_window_handle"],
        np.asarray(market_seq, dtype=np.float64),
    )
    np.testing.assert_allclose(
        obs_stack["portfolio_vector"],
        np.asarray(portfolio_seq, dtype=np.float64),
    )
    np.testing.assert_allclose(
        obs_stack["order_summary_vector"],
        np.asarray(orders_seq, dtype=np.float64),
    )
    np.testing.assert_array_equal(dones_many, np.asarray(dones_seq, dtype=np.bool_))
    np.testing.assert_allclose(rewards_many, np.asarray(rewards_seq, dtype=np.float64))
