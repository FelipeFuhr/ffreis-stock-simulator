"""Parity tests between the numba-jit path and pure-Python core step.

The numba path is a hot path that mirrors step_core in Python; if it diverges,
real production runs will silently use a different engine. These tests pin
parity between the two and exercise the previously-untested numba branch.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from numpy import float64 as np_float64
from numpy.typing import NDArray

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.types import Action


@pytest.fixture(params=[False, True], ids=["python", "numba"])
def use_numba(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


def _actions() -> list[Action]:
    return [
        Action(side="hold"),
        Action(side="buy", units=2.0, order_type="market"),
        Action(side="sell", units=1.0, order_type="market"),
        Action(side="buy", units=1.0, order_type="limit", limit_price=101.0),
        Action(side="hold"),
        Action(side="sell", units=2.0, order_type="limit", limit_price=105.0),
    ]


class TestUseNumbaPath:
    """Run step_many under both python and numba backends."""

    def test_step_many_runs_under_numba(
        self,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
        use_numba: bool,
    ) -> None:
        cfg = GameConfig(seed=42, use_numba=use_numba)
        env = MarketEnv(data=market_data_factory(n=128), cfg=cfg)
        env.reset(seed=42)
        encoded = encode_actions(_actions())
        obs, rewards, dones, _ = env.step_many(encoded)
        # obs is a dict of stacked arrays.
        assert rewards.shape == (len(encoded),)
        assert dones.shape == (len(encoded),)
        for arr in obs.values():
            assert arr.shape[0] == len(encoded)

    def test_python_and_numba_match_on_rewards(
        self,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
    ) -> None:
        encoded = encode_actions(_actions())

        # Python path.
        env_py = MarketEnv(data=market_data_factory(n=128), cfg=GameConfig(seed=7, use_numba=False))
        env_py.reset(seed=7)
        _, rewards_py, dones_py, _ = env_py.step_many(encoded)

        # Numba path with identical inputs.
        env_nb = MarketEnv(data=market_data_factory(n=128), cfg=GameConfig(seed=7, use_numba=True))
        env_nb.reset(seed=7)
        _, rewards_nb, dones_nb, _ = env_nb.step_many(encoded)

        # Both paths must agree.
        for i in range(len(encoded)):
            assert rewards_py[i] == pytest.approx(float(rewards_nb[i]), rel=1e-5, abs=1e-5)
            assert bool(dones_py[i]) == bool(dones_nb[i])

    @pytest.mark.parametrize("fee_bps", [0.0, 2.0, 10.0])
    def test_fees_apply_consistently(
        self,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
        fee_bps: float,
    ) -> None:
        cfg = GameConfig(seed=1, use_numba=False, fee_bps=fee_bps)
        env = MarketEnv(data=market_data_factory(n=32), cfg=cfg)
        env.reset(seed=1)
        encoded = encode_actions([Action(side="buy", units=1.0, order_type="market") for _ in range(4)])
        _, rewards, _, _ = env.step_many(encoded)
        for r in rewards:
            assert not math.isnan(float(r))
            assert math.isfinite(float(r))

    @pytest.mark.parametrize("slippage_bps", [0.0, 5.0, 20.0])
    def test_slippage_applies_consistently(
        self,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
        slippage_bps: float,
    ) -> None:
        cfg = GameConfig(seed=2, use_numba=False, slippage_bps=slippage_bps)
        env = MarketEnv(data=market_data_factory(n=32), cfg=cfg)
        env.reset(seed=2)
        encoded = encode_actions([Action(side="sell", units=1.0, order_type="market") for _ in range(4)])
        _, rewards, _, _ = env.step_many(encoded)
        for r in rewards:
            assert not math.isnan(float(r))
