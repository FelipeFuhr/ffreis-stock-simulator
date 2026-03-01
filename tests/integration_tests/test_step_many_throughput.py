from __future__ import annotations

from os import getenv as os_getenv
from sys import gettrace as sys_gettrace
from time import perf_counter as time_perf_counter
from collections.abc import Callable

from numpy import array as np_array, float64 as np_float64, nan as np_nan, ndarray as np_ndarray, random as np_random, zeros as np_zeros
from pytest import skip as pytest_skip

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv


def _build_actions(num_steps: int, seed: int = 123) -> np_ndarray:
    rng = np_random.default_rng(seed)
    actions = np_zeros((num_steps, 4), dtype=np_float64)
    for i in range(num_steps):
        roll = rng.integers(0, 5)
        if roll == 0:
            actions[i] = np_array([0.0, 0.0, 0.0, np_nan], dtype=np_float64)
        elif roll in {1, 2}:
            side_code = 1.0 if roll == 1 else -1.0
            units = float(rng.uniform(0.1, 3.0))
            actions[i] = np_array([side_code, units, 0.0, np_nan], dtype=np_float64)
        else:
            side_code = 1.0 if roll == 3 else -1.0
            units = float(rng.uniform(0.1, 3.0))
            limit = float(rng.uniform(90.0, 120.0))
            actions[i] = np_array([side_code, units, 1.0, limit], dtype=np_float64)
    return actions


def test_step_many_throughput_smoke(
    market_data_factory: Callable[..., MarketData],
    cfg_factory: Callable[..., GameConfig],
) -> None:
    if sys_gettrace() is not None:
        pytest_skip("throughput smoke is not stable under active tracing/coverage")

    num_steps = 20_000
    min_steps_per_sec = float(os_getenv("SIM_MIN_STEP_MANY_SPS", "2000"))

    data = market_data_factory(n=2048, slope=0.02, spread=0.2, volume=20_000.0)
    cfg = cfg_factory(seed=123, use_numba=False)
    env = MarketEnv(data=data, cfg=cfg)
    _ = env.reset(seed=123)
    actions = _build_actions(num_steps=num_steps, seed=123)

    start = time_perf_counter()
    _, _, _ = env.step_many(actions)
    elapsed = time_perf_counter() - start
    steps_per_sec = num_steps / elapsed if elapsed > 0 else float("inf")

    assert steps_per_sec >= min_steps_per_sec, (
        f"step_many throughput regression: {steps_per_sec:.0f} < {min_steps_per_sec:.0f} steps/sec"
    )
