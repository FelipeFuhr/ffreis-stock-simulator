"""Narrated walkthrough of margin enforcement: the two-tier model, end to end.

Run it with:

    uv run python examples/margin_scenario.py

No extras are required — this script only touches the core simulation engine
(`stock_simulator.env.MarketEnv`), not the HTTP/gRPC transports, so the base
project dependencies (already installed by a plain `uv sync`) are enough.

It runs two scenarios back to back. The first, over a synthetic 55-bar market (50
flat bars at 100.0, then a single 40% crash bar down to 60.0, then 4 more flat
bars so the crash is never coincidentally the last bar of data), demonstrates:

1. **Fill clipping (initial margin)** — requesting a buy far beyond
   `GameConfig.max_leverage`'s default 3.0x cap gets silently sized down to the
   largest fill that lands exactly at the cap, not rejected outright (a real
   broker declines the extra margin, it does not error). This is the *order-time*
   check, and the only place `max_leverage` is ever consulted.
2. **The insolvency backstop** — that 40% single-bar crash is violent enough to
   gap clean past the maintenance-margin threshold (which would otherwise have
   closed the book at a mark of ~66.99) and land at *negative* equity in one move.
   `equity <= 0` catches it there. That is exactly the deep-tail case the backstop
   exists for; scenario 2 shows the check that normally fires first.
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

The second scenario replays the same opening trade over a gentler decline and
demonstrates the check that fires in the ordinary case:

4. **Leverage legitimately drifting above the cap** — a mild dip to 80.0 leaves
   the book at 6.0x, twice the 3.0x order-time cap, and *nothing happens*. Real
   exchanges do not continuously re-clip an open position; the cap governs new
   orders only.
5. **Maintenance-margin liquidation** — a further drop to 66.75 puts equity at
   +2.50 against a 10.0125 requirement (0.5% of the 2002.5 notional), and the
   account is closed out **holding positive equity** — the earlier, more sensitive
   intervention a real exchange makes, long before the balance reaches zero.

For the complementary case — a buy sized *just under* the cap that does NOT get
clipped, then liquidation verified through the real HTTP `/v1/step_many` surface
with exact assertions — see
`tests/integration_tests/test_margin_lifecycle.py`. That test and scenario 1
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
_RESET_START_T = 47

_INITIAL_CASH = 1000.0
_REQUESTED_UNITS = 100.0  # deliberately far beyond the 3.0x cap (would be 10.0x)

# Scenario 2: the same opening trade, declining gently enough for the maintenance
# check to be the thing that fires. With 30 units against -2000 cash, liquidation
# triggers below a mark of 2000 / (30 - 30 * 0.005) = 66.998...: the dip bar sits
# well above it (and merely drifts leverage to 6.0x), the next bar sits just under.
_DIP_PRICE = 80.0
_MARGIN_CALL_PRICE = 66.75


def _build_market_data(
    crash_price: float = _CRASH_PRICE,
    dip_price: float | None = None,
) -> MarketData:
    closes = [_FLAT_PRICE] * _FLAT_BAR_COUNT
    if dip_price is not None:
        closes = closes + [dip_price]
    closes = closes + [crash_price] * (1 + _TAIL_BAR_COUNT)
    idx = pd_date_range("2024-01-01", periods=len(closes), freq="h")
    frame = pd_DataFrame(
        {
            "timestamp": idx,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [10_000.0] * len(closes),
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


def _scenario_config() -> GameConfig:
    return GameConfig(
        use_numba=False,
        initial_cash=_INITIAL_CASH,
        # max_leverage (3.0) and maintenance_margin_rate (0.005) both left at their
        # documented defaults — the point is to exercise those, not custom values.
        fee_bps=0.0,
        slippage_bps=0.0,
        market_latency_bars=0,
        partial_fill_min=1.0,
        partial_fill_max=1.0,
        seed=1,
    )


def _maintenance_margin_walkthrough() -> None:
    """Scenario 2: leverage drifts past the cap untouched, then maintenance margin fires."""
    print("\n" + "=" * 78)
    print("Scenario 2: leverage drift above the cap is fine — maintenance margin is not")
    print("=" * 78)

    cfg = _scenario_config()
    env = MarketEnv(data=_build_market_data(crash_price=_MARGIN_CALL_PRICE, dip_price=_DIP_PRICE), cfg=cfg)
    try:
        env.reset(seed=1, start_t=_RESET_START_T)
        print(
            f"\nMarket: flat at {_FLAT_PRICE:.2f}, a dip to {_DIP_PRICE:.2f}, then "
            f"{_MARGIN_CALL_PRICE:.2f}. Same opening buy as scenario 1."
        )
        portfolio, _reward, done, _trace = _step_once(
            env, Action(side="buy", units=_REQUESTED_UNITS, order_type="market")
        )
        cash, units, equity, leverage = portfolio
        _print_state("after clipped buy", cash, units, equity, leverage, done)

        while env.observe().t < _FLAT_BAR_COUNT - 1:
            portfolio, _reward, done, _trace = _step_once(env, Action(side="hold"))

        print(f"\nStep A: hold into the dip bar (close={_DIP_PRICE:.2f}).")
        portfolio, _reward, done, _trace = _step_once(env, Action(side="hold"))
        cash, units, equity, leverage = portfolio
        _print_state("after the dip", cash, units, equity, leverage, done)
        print(
            f"  -> leverage is now {leverage:.2f}x, well ABOVE the {cfg.max_leverage:.1f}x cap, and nothing "
            f"happened. The cap is an ORDER-TIME check; an open position is never re-clipped. Equity "
            f"{equity:.2f} is still far above the maintenance requirement "
            f"({abs(units) * _DIP_PRICE * cfg.maintenance_margin_rate:.2f})."
        )
        assert leverage > cfg.max_leverage, "expected leverage to drift above the order-time cap"
        assert not done and units > 0.0, "expected the position to survive the dip"

        print(f"\nStep B: one more hold drops the mark to {_MARGIN_CALL_PRICE:.2f}.")
        held_units = units
        portfolio, _reward, done, _trace = _step_once(env, Action(side="hold"))
        cash, units, equity, leverage = portfolio
        _print_state("after the margin call", cash, units, equity, leverage, done)
        notional = held_units * _MARGIN_CALL_PRICE
        print(
            f"  -> LIQUIDATED with {equity:+.2f} equity STILL POSITIVE: the margin balance fell under the "
            f"{notional * cfg.maintenance_margin_rate:.4f} maintenance requirement "
            f"({cfg.maintenance_margin_rate:.1%} of the {notional:.2f} notional). The `equity <= 0` backstop "
            f"would have carried this book all the way down to a mark of 66.67 before acting."
        )
        assert done and units == 0.0, "expected a maintenance-margin liquidation"
        assert equity > 0.0, "expected the account to be closed out with equity still positive"
    finally:
        env.close()


def main() -> None:
    print("=" * 78)
    print("Scenario 1: over-leverage attempt -> clip observed; gap crash -> backstop")
    print("=" * 78)

    data = _build_market_data()
    cfg = _scenario_config()
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

            # --- 2. The insolvency backstop ------------------------------------------
            print(f"\nStep 3: one more hold crosses into the crash bar (close={_CRASH_PRICE:.2f}).")
            portfolio, reward, done, trace = _step_once(env, Action(side="hold"))
            cash, units, equity, leverage = portfolio
            _print_state("after crash mark-to-market", cash, units, equity, leverage, done)
            print(
                f"  -> trace row: has_filled_order={trace.has_filled_order}  fills={trace.fills}  reward={reward:.2f}"
            )
            if done and units == 0.0:
                print(
                    f"  -> LIQUIDATED at {equity:.2f} equity. This bar gapped clean past the maintenance-margin "
                    f"threshold (~66.99 on this book) straight through zero, so it is the `equity <= 0` BACKSTOP "
                    f"that caught it — the deep-tail case it exists for. cash now equals equity exactly "
                    f"({cash:.2f} == {equity:.2f}), and the episode is done."
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

    _maintenance_margin_walkthrough()

    print("\nDone.")


if __name__ == "__main__":
    main()
