"""Narrated walkthrough of margin enforcement: fill clipping + insolvency liquidation.

Run it with:

    uv run python examples/margin_scenario.py

No extras are required — this script only touches the core simulation engine
(`stock_simulator.env.MarketEnv`), not the HTTP/gRPC transports, so the base
project dependencies (already installed by a plain `uv sync`) are enough.

What this demonstrates, in order, over a synthetic 55-bar market (50 flat bars at
100.0, then a single 40% crash bar down to 60.0, then 4 more flat bars so the
crash is never coincidentally the last bar of data):

1. **Fill clipping** — requesting a buy far beyond `GameConfig.max_leverage`'s
   default 3.0x cap gets silently sized down to the largest fill that lands
   exactly at the cap, not rejected outright (a real broker declines the extra
   margin, it does not error).
2. **Insolvency liquidation** — once that capped, leveraged position is open, the
   40% crash bar wipes out equity below zero. The engine force-closes the whole
   position at that instant (not one bar earlier, not one bar late) and ends the
   episode.
3. **Server-side trace recording** — the same `STOCK_SIM_TRACE_JSONL` sink a
   production server wires up at startup (`server.py::_load_engine`) is wired
   here directly against `MarketEnv`, and its JSONL output is printed at the end
   so you can see exactly what a debugging session watching that file would see
   for both the clipped fill and the liquidating step. This sink is only ever
   written to from `MarketEnv.step_many` (the RL agent's bulk HTTP path) — the
   singular `MarketEnv.step` already has the separate `Recorder` port for that
   job (see `trace_writer.py`'s module docstring) — so this script drives every
   step through `step_many` with single-action batches, not `step`, specifically
   so the trace file actually has content to show at the end.

For the complementary case — a buy sized *just under* the cap that does NOT get
clipped, then liquidation verified through the real HTTP `/v1/step_many` surface
with exact assertions — see
`tests/integration_tests/test_margin_lifecycle.py`. That test and this script
intentionally use the same bar shape (50 flat + 1 crash + 4 tail, 100.0 -> 60.0)
so the two are easy to compare side by side; this script's buy is deliberately
*over* the cap instead of just under it, to show the clip mechanism actually
triggering.
"""

from __future__ import annotations

from json import dumps as json_dumps
from json import loads as json_loads
from math import isclose as math_isclose
from pathlib import Path
from tempfile import TemporaryDirectory

from numpy import asarray as np_asarray
from numpy import float64 as np_float64
from numpy import nan as np_nan
from numpy.typing import NDArray
from pandas import DataFrame as pd_DataFrame
from pandas import date_range as pd_date_range

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.trace_writer import TraceJsonlWriter
from stock_simulator.types import Action, StepTraceRow

# Same bar shape as tests/integration_tests/test_margin_lifecycle.py, defined
# independently here (an example script should stand on its own, not import
# test code) — see that test's module docstring for the full derivation.
_FLAT_PRICE = 100.0
_CRASH_PRICE = 60.0  # a 40% single-bar drop
_FLAT_BAR_COUNT = 50
_TAIL_BAR_COUNT = 4
_TOTAL_BARS = _FLAT_BAR_COUNT + 1 + _TAIL_BAR_COUNT
_RESET_START_T = 47

_INITIAL_CASH = 1000.0
_REQUESTED_UNITS = 100.0  # deliberately far beyond the 3.0x cap (would be 10.0x)


def _build_market_data() -> MarketData:
    closes = [_FLAT_PRICE] * _FLAT_BAR_COUNT + [_CRASH_PRICE] * (1 + _TAIL_BAR_COUNT)
    idx = pd_date_range("2024-01-01", periods=_TOTAL_BARS, freq="h")
    frame = pd_DataFrame(
        {
            "timestamp": idx,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [10_000.0] * _TOTAL_BARS,
        }
    )
    return MarketData(frame)


def _print_state(label: str, cash: float, units: float, equity: float, leverage: float, done: bool) -> None:
    print(
        f"  {label:<28} cash={cash:>10.2f}  units={units:>7.2f}  equity={equity:>10.2f}  "
        f"leverage={leverage:>5.2f}x  done={done}"
    )


def _encode(action: Action) -> NDArray[np_float64]:
    """Encode one Action as a single-row step_many matrix, [1, 4]."""
    side_code = {"hold": 0.0, "buy": 1.0, "sell": -1.0}[action.side]
    order_code = 1.0 if action.order_type == "limit" else 0.0
    limit = np_nan if action.limit_price is None else float(action.limit_price)
    return np_asarray([[side_code, float(action.units), order_code, limit]], dtype=np_float64)


def _step_once(env: MarketEnv, action: Action) -> tuple[list[float], float, bool, StepTraceRow]:
    """Advance one bar via step_many (single-row batch) — see module docstring for why."""
    observations, rewards, dones, trace_rows = env.step_many(_encode(action), include_trace=True)
    portfolio_vector: list[float] = observations["portfolio_vector"][0].tolist()
    return portfolio_vector, float(rewards[0]), bool(dones[0]), trace_rows[0]


def main() -> None:
    print("=" * 78)
    print("Margin scenario: over-leverage attempt -> clip observed; crash -> liquidation")
    print("=" * 78)

    data = _build_market_data()
    cfg = GameConfig(
        use_numba=False,
        initial_cash=_INITIAL_CASH,
        # max_leverage left at its default (3.0) — the point of this scenario is
        # to exercise that documented default cap, not a custom one.
        fee_bps=0.0,
        slippage_bps=0.0,
        market_latency_bars=0,
        partial_fill_min=1.0,
        partial_fill_max=1.0,
        seed=1,
    )
    print(
        f"\nConfig: initial_cash={cfg.initial_cash:.2f}  max_leverage={cfg.max_leverage:.1f}x  "
        f"fee_bps={cfg.fee_bps}  slippage_bps={cfg.slippage_bps}"
    )
    print(
        f"Market: {_FLAT_BAR_COUNT} bars flat at {_FLAT_PRICE:.2f}, then a 40% drop to "
        f"{_CRASH_PRICE:.2f} at bar {_FLAT_BAR_COUNT}, then {_TAIL_BAR_COUNT} more flat bars."
    )

    with TemporaryDirectory(prefix="margin-scenario-") as tmp_dir:
        trace_path = Path(tmp_dir) / "trace.jsonl"
        env = MarketEnv(data=data, cfg=cfg, trace_sink=TraceJsonlWriter(trace_path))
        try:
            state = env.reset(seed=1, start_t=_RESET_START_T)
            print(f"\nreset(seed=1, start_t={_RESET_START_T}) -> bar {state.t}")
            _print_state("after reset", state.cash, state.units, state.equity, state.leverage, state.done)

            # --- 1. Fill clipping ---------------------------------------------------
            print(
                f"\nStep 1: submit a market buy for {_REQUESTED_UNITS:.0f} units "
                f"(would be {_REQUESTED_UNITS * _FLAT_PRICE / _INITIAL_CASH:.1f}x leverage — "
                f"well past the {cfg.max_leverage:.1f}x cap)."
            )
            portfolio, _reward, done, trace = _step_once(
                env, Action(side="buy", units=_REQUESTED_UNITS, order_type="market")
            )
            cash, units, equity, leverage = portfolio
            print(
                f"  -> exchange filled only {units:.2f} units, not the requested {_REQUESTED_UNITS:.0f} — "
                f"clipped to land exactly at the {cfg.max_leverage:.1f}x cap."
            )
            print(f"  -> trace row: has_filled_order={trace.has_filled_order}  exec_price={trace.exec_price}")
            _print_state("after clipped buy", cash, units, equity, leverage, done)
            # Sanity checks, not the point of this script (the narration above is):
            # the fill was clipped strictly below the request, and it landed
            # exactly on the leverage cap.
            assert units < _REQUESTED_UNITS, "expected the fill to be clipped below the request"
            assert math_isclose(leverage, cfg.max_leverage, rel_tol=1e-9)

            # --- Hold through the flat bars up to the crash -------------------------
            print("\nStep 2: hold through the remaining flat bars up to the crash bar...")
            while env.observe().t < _FLAT_BAR_COUNT - 1:
                portfolio, _reward, done, _trace = _step_once(env, Action(side="hold"))
                cash, units, equity, leverage = portfolio
                _print_state(f"  hold -> bar {env.observe().t}", cash, units, equity, leverage, done)

            # --- 2. Insolvency liquidation -------------------------------------------
            print(f"\nStep 3: one more hold crosses into the crash bar (close={_CRASH_PRICE:.2f}).")
            portfolio, reward, done, trace = _step_once(env, Action(side="hold"))
            cash, units, equity, leverage = portfolio
            _print_state("after crash mark-to-market", cash, units, equity, leverage, done)
            print(
                f"  -> trace row: has_filled_order={trace.has_filled_order}  fills={trace.fills}  reward={reward:.2f}"
            )
            if done and units == 0.0:
                print(
                    f"  -> INSOLVENT: equity crossed <= 0, so the whole position was force-liquidated. "
                    f"cash now equals equity exactly ({cash:.2f} == {equity:.2f}), and the episode is done."
                )
            else:  # pragma: no cover — narration-only guard, not expected on this scenario
                print("  -> still solvent (unexpected for this scenario's numbers).")

            # --- Terminal state does not drift --------------------------------------
            print("\nStep 4: one further hold on the now-terminal episode...")
            portfolio, _reward, done, _trace = _step_once(env, Action(side="hold"))
            cash, units, equity, leverage = portfolio
            _print_state("  further hold (terminal)", cash, units, equity, leverage, done)
            print("  -> equity is unchanged: a terminal state never drifts on later steps.")
        finally:
            env.close()  # flushes/closes the TraceJsonlWriter sink

        print(f"\nTrace file ({trace_path.name}), as a STOCK_SIM_TRACE_JSONL consumer would see it:")
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            row = json_loads(line)
            compact = {
                key: row[key]
                for key in ("t", "fills", "has_filled_order", "exec_price", "position_units", "equity", "done")
            }
            print(f"  {json_dumps(compact)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
