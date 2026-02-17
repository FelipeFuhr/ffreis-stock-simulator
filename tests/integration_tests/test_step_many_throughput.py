from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable

import numpy as np
import pytest

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv


def _build_actions(num_steps: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions = np.zeros((num_steps, 4), dtype=np.float64)
    for i in range(num_steps):
        roll = rng.integers(0, 5)
        if roll == 0:
            actions[i] = np.array([0.0, 0.0, 0.0, np.nan], dtype=np.float64)
        elif roll in {1, 2}:
            side_code = 1.0 if roll == 1 else -1.0
            units = float(rng.uniform(0.1, 3.0))
            actions[i] = np.array([side_code, units, 0.0, np.nan], dtype=np.float64)
        else:
            side_code = 1.0 if roll == 3 else -1.0
            units = float(rng.uniform(0.1, 3.0))
            limit = float(rng.uniform(90.0, 120.0))
            actions[i] = np.array([side_code, units, 1.0, limit], dtype=np.float64)
    return actions


def test_step_many_throughput_smoke(
    market_data_factory: Callable[..., MarketData],
    cfg_factory: Callable[..., GameConfig],
) -> None:
    if sys.gettrace() is not None:
        pytest.skip("throughput smoke is not stable under active tracing/coverage")

    num_steps = 20_000
    min_steps_per_sec = float(os.getenv("SIM_MIN_STEP_MANY_SPS", "2000"))

    data = market_data_factory(n=2048, slope=0.02, spread=0.2, volume=20_000.0)
    cfg = cfg_factory(seed=123, use_numba=False)
    env = MarketEnv(data=data, cfg=cfg)
    _ = env.reset(seed=123)
    actions = _build_actions(num_steps=num_steps, seed=123)

    start = time.perf_counter()
    _, _, _ = env.step_many(actions)
    elapsed = time.perf_counter() - start
    steps_per_sec = num_steps / elapsed if elapsed > 0 else float("inf")

    assert steps_per_sec >= min_steps_per_sec, (
        f"step_many throughput regression: {steps_per_sec:.0f} < {min_steps_per_sec:.0f} steps/sec"
    )
