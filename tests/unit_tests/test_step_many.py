from __future__ import annotations

from collections.abc import Callable

from numpy import asarray as np_asarray, bool_ as np_bool_, float64 as np_float64, ndarray as np_ndarray, testing as np_testing
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

    obs_stack, rewards_many, dones_many = env_many.step_many(encoded)

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
