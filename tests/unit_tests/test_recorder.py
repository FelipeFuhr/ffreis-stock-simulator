from __future__ import annotations

from collections.abc import Callable

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.recorder import InMemoryRecorder
from stock_simulator.types import Action


def _run_episode(seed: int, market_data_factory: Callable[..., MarketData]) -> tuple[float, tuple]:
    recorder = InMemoryRecorder()
    env = MarketEnv(
        data=market_data_factory(n=300, slope=0.2, spread=0.4, volume=20_000.0),
        cfg=GameConfig(seed=seed, use_numba=False),
        recorder=recorder,
    )
    env.reset(seed=seed)

    actions = (
        Action(side="hold"),
        Action(side="buy", units=4.0, order_type="market"),
        Action(side="sell", units=2.0, order_type="limit", limit_price=103.0),
        Action(side="buy", units=1.0, order_type="limit", limit_price=101.0),
    )
    result = None
    for i in range(100):
        result = env.step(actions[i % len(actions)])
        if result.done:
            break
    assert result is not None
    return result.observation.equity, recorder.replay()


def test_recorder_captures_required_fields(
    market_data_factory: Callable[..., MarketData],
) -> None:
    final_equity, rows = _run_episode(seed=111, market_data_factory=market_data_factory)
    assert rows
    row = rows[0]
    assert row.side in {"buy", "sell", "hold"}
    assert row.order_type in {"market", "limit"}
    assert isinstance(row.fills, int)
    assert isinstance(row.equity, float)
    assert isinstance(row.leverage, float)
    assert isinstance(row.price, float)
    assert isinstance(final_equity, float)


def test_replay_is_deterministic_for_same_seed(
    market_data_factory: Callable[..., MarketData],
) -> None:
    equity_a, replay_a = _run_episode(seed=2026, market_data_factory=market_data_factory)
    equity_b, replay_b = _run_episode(seed=2026, market_data_factory=market_data_factory)
    assert equity_a == equity_b
    assert replay_a == replay_b
