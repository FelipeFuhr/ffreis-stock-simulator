from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.random import default_rng

if TYPE_CHECKING:
    from stock_simulator.config import GameConfig
    from stock_simulator.core import MarketArrays
    from stock_simulator.types import Action

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_TOTAL_STEPS = 1_000_000
DEFAULT_SERIES_LENGTH = 4096


def _build_market_arrays(n: int, seed: int) -> MarketArrays:
    from stock_simulator.core import MarketArrays

    rng = default_rng(seed)
    returns = rng.normal(loc=0.0001, scale=0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    spread = rng.uniform(0.0003, 0.01, size=n)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.uniform(10_000.0, 100_000.0, size=n)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    _ = ts, volume  # keep generation closer to real app path
    return MarketArrays(
        open=np.asarray(open_, dtype=np.float32),
        high=np.asarray(high, dtype=np.float32),
        low=np.asarray(low, dtype=np.float32),
        close=np.asarray(close, dtype=np.float32),
        n=n,
    )


def _action_stream() -> tuple[Action, ...]:
    from stock_simulator.types import Action

    return (
        Action(side="hold", units=0.0),
        Action(side="buy", units=8.0, order_type="market"),
        Action(side="sell", units=8.0, order_type="market"),
        Action(side="buy", units=5.0, order_type="limit", limit_price=99.0),
        Action(side="sell", units=5.0, order_type="limit", limit_price=101.0),
    )


def _run_pure_python(
    total_steps: int,
    market: MarketArrays,
    cfg: GameConfig,
    actions: tuple[Action, ...],
) -> float:
    from stock_simulator.core import initial_core_state, step_core

    state = initial_core_state(cfg.initial_cash, max_orders=cfg.max_open_orders)
    rng = default_rng(cfg.seed)
    start = time.perf_counter()
    for i in range(total_steps):
        action = actions[i % len(actions)]
        random_draws = rng.uniform(
            cfg.partial_fill_min,
            cfg.partial_fill_max,
            size=cfg.max_open_orders,
        ).astype(np.float64, copy=False)
        output = step_core(
            state=state,
            action=action,
            market_arrays=market,
            config=cfg,
            random_draws=random_draws,
        )
        state = output.state
        if state.done:
            state = initial_core_state(cfg.initial_cash, max_orders=cfg.max_open_orders)
    elapsed = time.perf_counter() - start
    return total_steps / elapsed


def _run_numba(
    total_steps: int,
    market: MarketArrays,
    cfg: GameConfig,
    actions: tuple[Action, ...],
) -> float:
    from stock_simulator.core import initial_core_state, step_core_numba

    state = initial_core_state(cfg.initial_cash, max_orders=cfg.max_open_orders)
    rng = default_rng(cfg.seed)

    # Warm-up compile so measured time reflects steady-state throughput.
    warmup_draws = rng.uniform(
        cfg.partial_fill_min,
        cfg.partial_fill_max,
        size=cfg.max_open_orders,
    ).astype(np.float64, copy=False)
    _ = step_core_numba(
        state=state,
        action=actions[0],
        market_arrays=market,
        config=cfg,
        random_draws=warmup_draws,
    )

    state = initial_core_state(cfg.initial_cash, max_orders=cfg.max_open_orders)
    rng = default_rng(cfg.seed)
    start = time.perf_counter()
    for i in range(total_steps):
        action = actions[i % len(actions)]
        random_draws = rng.uniform(
            cfg.partial_fill_min,
            cfg.partial_fill_max,
            size=cfg.max_open_orders,
        ).astype(np.float64, copy=False)
        output = step_core_numba(
            state=state,
            action=action,
            market_arrays=market,
            config=cfg,
            random_draws=random_draws,
        )
        state = output.state
        if state.done:
            state = initial_core_state(cfg.initial_cash, max_orders=cfg.max_open_orders)
    elapsed = time.perf_counter() - start
    return total_steps / elapsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-core throughput benchmark.")
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.getenv("BENCH_STEPS", str(DEFAULT_TOTAL_STEPS))),
    )
    parser.add_argument(
        "--series-length",
        type=int,
        default=int(os.getenv("BENCH_SERIES_LENGTH", str(DEFAULT_SERIES_LENGTH))),
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    from stock_simulator.config import GameConfig

    args = _parse_args()
    total_steps = int(args.steps)
    series_length = int(args.series_length)
    cfg = GameConfig(seed=int(args.seed))
    market = _build_market_arrays(n=series_length, seed=cfg.seed)
    actions = _action_stream()

    py_sps = _run_pure_python(total_steps, market, cfg, actions)
    nb_sps = _run_numba(total_steps, market, cfg, actions)

    print(f"steps: {total_steps:,}")
    print(f"pure Python: {py_sps:,.0f} steps/sec")
    print(f"Numba JIT:   {nb_sps:,.0f} steps/sec")
    if py_sps > 0:
        print(f"speedup:     {nb_sps / py_sps:.2f}x")


if __name__ == "__main__":
    main()
