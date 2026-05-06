from __future__ import annotations

from collections.abc import Callable

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.types import Action


def _run_equity_path(seed: int, market_data_factory: Callable[..., MarketData]) -> list[float]:
    env = MarketEnv(
        data=market_data_factory(n=256, slope=0.1, spread=0.5, volume=10_000.0),
        cfg=GameConfig(seed=seed, use_numba=False),
    )
    env.reset(seed=seed)

    actions = (
        Action(side="hold"),
        Action(side="buy", units=5.0, order_type="market"),
        Action(side="sell", units=3.0, order_type="limit", limit_price=103.0),
        Action(side="buy", units=2.0, order_type="limit", limit_price=101.0),
    )

    equities: list[float] = []
    for i in range(120):
        result = env.step(actions[i % len(actions)])
        equities.append(result.observation.equity)
        if result.done:
            break
    return equities


def test_same_seed_produces_identical_equity_path(
    market_data_factory: Callable[..., MarketData],
) -> None:
    path_a = _run_equity_path(seed=2026, market_data_factory=market_data_factory)
    path_b = _run_equity_path(seed=2026, market_data_factory=market_data_factory)
    assert path_a == path_b
