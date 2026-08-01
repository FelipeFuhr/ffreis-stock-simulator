"""Parity tests between the numba-jit path and pure-Python core step.

The numba path is a hot path that mirrors step_core in Python; if it diverges,
real production runs will silently use a different engine. These tests pin
parity between the two and exercise the previously-untested numba branch.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from numpy import bool_ as np_bool_
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

    def test_python_and_numba_match_through_margin_clip_and_liquidation(
        self,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
    ) -> None:
        # Same trajectory as TestNumbaParity in test_margin_enforcement.py, but driven
        # through step_many so the second step_core_jit call site is covered too: a
        # buy clipped to the max_leverage cap, then a price crash that liquidates.
        closes = [100.0, 100.0, 40.0, 40.0, 40.0, 40.0]
        encoded = encode_actions(
            [Action(side="buy", units=500.0, order_type="market")] + [Action(side="hold") for _ in range(4)]
        )

        def _run(use_numba: bool) -> tuple[NDArray[np_float64], NDArray[np_float64], NDArray[np_bool_]]:
            cfg = GameConfig(
                seed=9,
                use_numba=use_numba,
                initial_cash=1_000.0,
                max_leverage=3.0,
                market_latency_bars=0,
                partial_fill_min=1.0,
                partial_fill_max=1.0,
            )
            env = MarketEnv(data=market_data_factory(close=closes, spread=0.1), cfg=cfg)
            env.reset(seed=9)
            obs, rewards, dones, _ = env.step_many(encoded)
            return obs["portfolio_vector"], rewards, dones

        portfolio_py, rewards_py, dones_py = _run(use_numba=False)
        portfolio_nb, rewards_nb, dones_nb = _run(use_numba=True)

        assert portfolio_py.tolist() == portfolio_nb.tolist()
        assert rewards_py.tolist() == rewards_nb.tolist()
        assert dones_py.tolist() == dones_nb.tolist()
        # The clip capped the opening buy at the default fee/slippage:
        # 3.0 * 1000 / (100.01 * (1 + 3.0 * 0.0004)) = 29.96104704351778 units.
        assert portfolio_py[0][1] == pytest.approx(29.96104704351778, rel=1e-12)
        # ...and the crash then liquidated the book and terminated the episode.
        assert bool(dones_py[1]) is True
        assert portfolio_py[1][1] == 0.0

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
