from __future__ import annotations

from collections.abc import Callable

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.types import Action


def _run(env: MarketEnv, actions: list[Action]) -> tuple[float, list[float]]:
    state = env.reset(seed=123)
    equities = [state.equity]
    for action in actions:
        result = env.step(action)
        equities.append(result.observation.equity)
        if result.done:
            break
    return equities[-1], equities


def test_flat_market_no_trades(
    market_data_factory: Callable[..., MarketData],
    cfg_factory: Callable[..., GameConfig],
) -> None:
    data = market_data_factory(close=[100.0] * 48, spread=0.1)
    env = MarketEnv(data=data, cfg=cfg_factory())
    actions = [Action(side="hold") for _ in range(30)]
    final_equity, equities = _run(env, actions)

    assert all(eq == equities[0] for eq in equities)
    assert final_equity == equities[0]


def test_monotonic_uptrend_long_is_profitable(
    market_data_factory: Callable[..., MarketData],
    cfg_factory: Callable[..., GameConfig],
) -> None:
    close = [100.0 + i * 0.5 for i in range(80)]
    data = market_data_factory(close=close, spread=0.1)
    env = MarketEnv(data=data, cfg=cfg_factory())
    actions = [Action(side="buy", units=10.0, order_type="market")] + [Action(side="hold") for _ in range(40)]
    final_equity, equities = _run(env, actions)

    assert final_equity > equities[0]
    assert max(equities) == final_equity


def test_monotonic_downtrend_long_loses_money(
    market_data_factory: Callable[..., MarketData],
    cfg_factory: Callable[..., GameConfig],
) -> None:
    close = [140.0 - i * 0.5 for i in range(80)]
    data = market_data_factory(close=close, spread=0.1)
    env = MarketEnv(data=data, cfg=cfg_factory())
    actions = [Action(side="buy", units=10.0, order_type="market")] + [Action(side="hold") for _ in range(40)]
    final_equity, equities = _run(env, actions)

    assert final_equity < equities[0]
    assert min(equities) == final_equity


def test_shock_bar_creates_large_drawdown(
    market_data_factory: Callable[..., MarketData],
    cfg_factory: Callable[..., GameConfig],
) -> None:
    close = [100.0] * 20 + [60.0] + [61.0, 62.0, 63.0, 64.0] + [64.0] * 20
    data = market_data_factory(close=close, spread=0.1)
    env = MarketEnv(data=data, cfg=cfg_factory())
    actions = [Action(side="buy", units=20.0, order_type="market")] + [Action(side="hold") for _ in range(35)]
    _, equities = _run(env, actions)

    pre_shock_equity = equities[10]
    trough = min(equities)
    assert trough < pre_shock_equity * 0.995


def test_margin_call_condition_is_detectable(
    market_data_factory: Callable[..., MarketData],
    cfg_factory: Callable[..., GameConfig],
) -> None:
    close = [100.0, 100.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    data = market_data_factory(close=close, spread=0.1)
    cfg = cfg_factory(max_leverage=3.0)
    env = MarketEnv(data=data, cfg=cfg)
    actions = [Action(side="buy", units=2000.0, order_type="market")] + [Action(side="hold") for _ in range(5)]
    _, equities = _run(env, actions)

    assert min(equities) <= 0.0
