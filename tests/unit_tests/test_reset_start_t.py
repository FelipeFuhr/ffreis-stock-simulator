"""MarketEnv.reset(start_t=...) — configurable episode start bar.

Fixes inert EMA warm-up in the downstream RL agent: episodes previously always
began at bar 0 (see :func:`stock_simulator.core.initial_core_state`), so a
warm-up fetch of history via ``/v1/market_window`` right after reset only ever
returned 1 bar, no matter how much history was loaded. ``start_t`` lets an
episode begin at an arbitrary bar index so warm-up windows (and walk-forward
training splits over different slices of the same market data) actually have
real history behind them.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.types import Action


def test_reset_default_start_t_is_zero_unchanged(
    market_data_factory: Callable[..., MarketData],
) -> None:
    env = MarketEnv(data=market_data_factory(n=64), cfg=GameConfig(seed=1, use_numba=False))
    state = env.reset(seed=1)
    assert state.t == 0
    assert state.done is False


def test_reset_nonzero_start_t_begins_mid_series(
    market_data_factory: Callable[..., MarketData],
) -> None:
    env = MarketEnv(data=market_data_factory(n=2500), cfg=GameConfig(seed=1, use_numba=False))
    state = env.reset(seed=1, start_t=2000)
    assert state.t == 2000
    assert state.done is False

    observation = env.observe()
    assert observation.t == 2000


def test_reset_start_t_at_last_valid_bar_is_allowed(
    market_data_factory: Callable[..., MarketData],
) -> None:
    # n - 1 is still a valid bar index (no future leakage yet occurred); the
    # episode simply has zero further steps before `done`.
    env = MarketEnv(data=market_data_factory(n=10), cfg=GameConfig(seed=1, use_numba=False))
    state = env.reset(start_t=9)
    assert state.t == 9
    assert state.done is False


@pytest.mark.parametrize("bad_start_t", [64, 65, 1_000, -1, -100])
def test_reset_out_of_range_start_t_raises_value_error(
    bad_start_t: int,
    market_data_factory: Callable[..., MarketData],
) -> None:
    env = MarketEnv(data=market_data_factory(n=64), cfg=GameConfig(seed=1, use_numba=False))
    with pytest.raises(ValueError, match="start_t"):
        env.reset(start_t=bad_start_t)


def test_reset_same_seed_and_start_t_is_deterministic(
    market_data_factory: Callable[..., MarketData],
) -> None:
    data = market_data_factory(n=512, slope=0.1, spread=0.5, volume=10_000.0)
    actions = (
        Action(side="hold"),
        Action(side="buy", units=5.0, order_type="market"),
        Action(side="sell", units=3.0, order_type="limit", limit_price=203.0),
        Action(side="buy", units=2.0, order_type="limit", limit_price=195.0),
    )

    def _run() -> list[float]:
        env = MarketEnv(data=data, cfg=GameConfig(seed=99, use_numba=False))
        state = env.reset(seed=99, start_t=300)
        equities = [state.equity]
        for i in range(40):
            result = env.step(actions[i % len(actions)])
            equities.append(result.observation.equity)
            if result.done:
                break
        return equities

    assert _run() == _run()


def test_reset_same_seed_and_start_t_is_deterministic_under_numba(
    market_data_factory: Callable[..., MarketData],
) -> None:
    data = market_data_factory(n=512, slope=0.1, spread=0.5, volume=10_000.0)
    actions = (
        Action(side="hold"),
        Action(side="buy", units=5.0, order_type="market"),
        Action(side="sell", units=3.0, order_type="limit", limit_price=203.0),
    )

    def _run() -> list[float]:
        env = MarketEnv(data=data, cfg=GameConfig(seed=7, use_numba=True))
        state = env.reset(seed=7, start_t=200)
        equities = [state.equity]
        for i in range(30):
            result = env.step(actions[i % len(actions)])
            equities.append(result.observation.equity)
            if result.done:
                break
        return equities

    assert _run() == _run()


def test_reset_different_start_t_same_seed_diverges(
    market_data_factory: Callable[..., MarketData],
) -> None:
    # Sanity check that start_t actually changes the episode (not silently
    # ignored): different starting bars on a sloped series produce different
    # initial observed prices.
    data = market_data_factory(n=512, slope=0.1, spread=0.5, volume=10_000.0)
    env_a = MarketEnv(data=data, cfg=GameConfig(seed=5, use_numba=False))
    env_b = MarketEnv(data=data, cfg=GameConfig(seed=5, use_numba=False))

    state_a = env_a.reset(seed=5, start_t=0)
    state_b = env_b.reset(seed=5, start_t=300)

    assert state_a.t != state_b.t
    assert env_a.observe().price != env_b.observe().price
