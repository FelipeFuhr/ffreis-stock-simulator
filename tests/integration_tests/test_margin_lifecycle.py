"""Full margin-enforced episode lifecycle, driven through the real HTTP surface.

`test_margin_enforcement.py` and `test_trace_writer.py` each pin ONE mechanism in
isolation (fill clipping, insolvency liquidation, JSONL trace recording) by calling
`core.py`/`env.py` functions directly. This test ties all three together the way a
real client (the RL agent) actually experiences them: through `/v1/reset` and
`/v1/step_many` on a live FastAPI app (`fastapi.testclient.TestClient`, the same
pattern `test_server_module.py`/`test_market_window.py` use), with a server-startup
`STOCK_SIM_TRACE_JSONL` sink wired exactly as `server.py::_load_engine` wires it in
production (real `MARKET_DATA_CSV` + `STOCK_SIM_*` env vars, not a hand-built
`MarketEnv` injected via a monkeypatched `_load_engine`).

Scenario (all prices/units are exact — no fee, no slippage, no partial fills, so
every number below is deterministic, not merely "expected in distribution"):

* 55 synthetic hourly bars: 50 flat bars at close=100.0 (index 0..49), then one
  bar with a 40% single-bar drop to close=60.0 (index 50), then 4 more flat bars
  at 60.0 (index 51..54) so the crash bar is never also the last bar of data —
  `done=True` at liquidation is attributable to insolvency alone, not a data-
  exhaustion coincidence (`next_t=50` vs `n-1=54`).
* `GameConfig`: `initial_cash=1000.0`, `max_leverage` left at its **default 3.0**
  (deliberately not overridden — the point is to exercise the documented default
  cap), `fee_bps=0.0`, `slippage_bps=0.0`, `market_latency_bars=0` (a market order
  submitted on bar `t` fills on bar `t`, at `open[t]`), `partial_fill_min =
  partial_fill_max = 1.0` (fills are exactly the requested size — no random
  fraction to reason about).
* `reset(seed=1, start_t=47)` — 47 bars into the 50-bar flat run, demonstrating the
  non-zero `start_t` warm-up feature (AGENTS.md) rather than always starting at 0.
* Buy 29 units market, at t=47: notional 29*100=2900 against 1000 equity is a
  2.9x leverage ask — *under* the 3.0x cap (30 units would be exactly 3.0x), so
  `clip_units_for_leverage` leaves it untouched: the fill is NOT clipped. Cash
  1000 - 2900 = -1900.0, units 29.0, equity unchanged at 1000.0 (fair-value entry,
  zero fee). State advances to t=48.
* Two holds step the episode from t=48 -> t=49 -> t=50. Both of bars 48 and 49
  are still flat at 100.0, so the first hold changes nothing (equity stays
  1000.0). The second hold's mark-to-market check (`settle_insolvency`, run
  against the *next* bar's close inside the same `step_core` call) evaluates
  equity at the crash bar's close=60.0: -1900 + 29*60 = -1900 + 1740 = -160.0,
  which is <= 0 — insolvent. The whole position is force-liquidated at that
  instant: units -> 0.0, cash -> exactly the pre-liquidation equity (-160.0), and
  `done=True` fires on this exact step (not the hold before it, not "eventually").
* The liquidating step is a *hold*, not a fill — `settle_insolvency` needs no
  order to trigger, it only marks the existing position to the new bar's close.
  So its own trace row has `has_filled_order=False`/`fills=0`; the fill that
  later gets liquidated is the earlier buy at t=48, and that row's trace line is
  what part (d) of the task ("has_filled_order=True and a sane exec_price for the
  liquidating fill") verifies below — the buy is "the [fill that ends up]
  liquidat[ed]", not a fill that happens on the liquidating step itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pandas import DataFrame as pd_DataFrame
from pandas import date_range as pd_date_range
from pytest import MonkeyPatch

from stock_simulator import server as server_mod

# ---------------------------------------------------------------------------
# Scenario constants — see the module docstring for the worked-out math behind
# every one of these numbers.
# ---------------------------------------------------------------------------
_FLAT_PRICE = 100.0
_CRASH_PRICE = 60.0  # exactly a 40% drop from _FLAT_PRICE
_FLAT_BAR_COUNT = 50  # indices 0..49
_CRASH_BAR_INDEX = _FLAT_BAR_COUNT  # index 50 — the single crash bar
_TAIL_BAR_COUNT = 4  # indices 51..54, flat at _CRASH_PRICE
_TOTAL_BARS = _FLAT_BAR_COUNT + 1 + _TAIL_BAR_COUNT  # 55
_RESET_START_T = 47  # 47 bars into the flat run — "past the warm-up"

_INITIAL_CASH = 1000.0
_BUY_UNITS = 29.0  # 29 * 100 / 1000 = 2.9x — just under the 3.0x default cap

_HOLD_ACTION = {"side_code": 0, "units": 0.0, "order_type_code": 0, "has_limit_price": False}
_BUY_ACTION = {"side_code": 1, "units": _BUY_UNITS, "order_type_code": 0, "has_limit_price": False}


def _write_scenario_market_csv(path: Path) -> None:
    """Write the 55-bar flat-then-crash-then-flat OHLCV series described above."""
    closes = [_FLAT_PRICE] * _FLAT_BAR_COUNT + [_CRASH_PRICE] * (1 + _TAIL_BAR_COUNT)
    assert len(closes) == _TOTAL_BARS
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
    frame.to_csv(path, index=False)


def _build_test_client(monkeypatch: MonkeyPatch, tmp_path: Path) -> tuple[TestClient, Path]:
    """Wire the real FastAPI app against the scenario market data + trace sink.

    Deliberately goes through the *real* `server.py::_load_engine` (real
    `MARKET_DATA_CSV`/`STOCK_SIM_TRACE_JSONL`/`STOCK_SIM_*` env vars), not a
    hand-built `MarketEnv` injected via a monkeypatched `_load_engine` — this is
    the same object graph a production server startup with `STOCK_SIM_TRACE_JSONL`
    set would build, so the trace file this test reads is proof the wiring works
    end to end, not just that `MarketEnv(trace_sink=...)` works in isolation.
    """
    market_csv = tmp_path / "margin_scenario_market.csv"
    _write_scenario_market_csv(market_csv)
    trace_path = tmp_path / "margin_scenario_trace.jsonl"

    monkeypatch.setenv("ENGINE_ENABLED", "true")
    monkeypatch.setenv("MARKET_DATA_CSV", str(market_csv))
    monkeypatch.setenv("STOCK_SIM_TRACE_JSONL", str(trace_path))
    monkeypatch.delenv("STOCK_SIM_CONFIG_YAML", raising=False)
    monkeypatch.delenv("STOCK_SIM_MAX_LEVERAGE", raising=False)  # keep the 3.0x default
    monkeypatch.setenv("STOCK_SIM_USE_NUMBA", "false")
    monkeypatch.setenv("STOCK_SIM_INITIAL_CASH", str(_INITIAL_CASH))
    monkeypatch.setenv("STOCK_SIM_FEE_BPS", "0.0")
    monkeypatch.setenv("STOCK_SIM_SLIPPAGE_BPS", "0.0")
    monkeypatch.setenv("STOCK_SIM_MARKET_LATENCY_BARS", "0")
    monkeypatch.setenv("STOCK_SIM_PARTIAL_FILL_MIN", "1.0")
    monkeypatch.setenv("STOCK_SIM_PARTIAL_FILL_MAX", "1.0")
    monkeypatch.setenv("STOCK_SIM_SEED", "1")

    app = server_mod.create_app()
    return TestClient(app), trace_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_margin_enforced_episode_lifecycle_via_http(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, trace_path = _build_test_client(monkeypatch, tmp_path)

    with client:
        # 1-2. Reset past the flat section's warm-up (see module docstring).
        reset_response = client.post("/v1/reset", json={"seed": 1, "start_t": _RESET_START_T})
        assert reset_response.status_code == 200
        reset_state = reset_response.json()["state"]
        assert reset_state == {
            "t": _RESET_START_T,
            "cash": pytest.approx(_INITIAL_CASH),
            "units": pytest.approx(0.0),
            "equity": pytest.approx(_INITIAL_CASH),
            "leverage": pytest.approx(0.0),
            "open_orders": 0,
            "done": False,
        }

        # 3. A market buy of 29 units at price 100 on 1000 equity requests 2.9x
        # leverage — under the 3.0x default cap (30 units would be exactly 3.0x).
        buy_response = client.post(
            "/v1/step_many",
            json={"actions": [_BUY_ACTION], "include_trace": True},
        )
        assert buy_response.status_code == 200
        buy_payload = buy_response.json()

        # 4. NOT clipped: filled the full requested 29 units, at the full 2.9x —
        # `clip_units_for_leverage` only ever shrinks a fill that would breach the
        # cap, and 2.9x doesn't.
        buy_observation = buy_payload["observations"][0]
        buy_cash, buy_units, buy_equity, buy_leverage = buy_observation["portfolio_vector"]
        assert buy_units == pytest.approx(_BUY_UNITS)
        assert buy_cash == pytest.approx(_INITIAL_CASH - _BUY_UNITS * _FLAT_PRICE)  # -1900.0
        assert buy_equity == pytest.approx(_INITIAL_CASH)  # fair-value entry, zero fee: unchanged
        assert buy_leverage == pytest.approx(2.9, rel=1e-12)
        assert buy_payload["dones"] == [False]
        assert buy_payload["rewards"] == [pytest.approx(0.0)]
        assert buy_observation["market_window_handle"]["t"] == _RESET_START_T + 1  # 48

        buy_trace = buy_payload["trace"][0]
        assert buy_trace["has_filled_order"] is True
        assert buy_trace["fills"] == 1
        assert buy_trace["exec_price"] == pytest.approx(_FLAT_PRICE)  # sane: no slippage, fills at open
        assert buy_trace["filled_order_slot"] == 0

        # The sink writes every step_many step regardless of include_trace (N10) —
        # one line so far, for the buy.
        trace_after_buy = _read_jsonl(trace_path)
        assert len(trace_after_buy) == 1
        assert trace_after_buy[0]["t"] == _RESET_START_T + 1
        assert trace_after_buy[0]["has_filled_order"] is True
        assert trace_after_buy[0]["exec_price"] == pytest.approx(_FLAT_PRICE)

        # 5. Step through the drop: two holds carry the episode from t=48 (still
        # flat) through t=49 (still flat) into t=50, the crash bar.
        drop_response = client.post(
            "/v1/step_many",
            json={"actions": [_HOLD_ACTION, _HOLD_ACTION], "include_trace": True},
        )
        assert drop_response.status_code == 200
        drop_payload = drop_response.json()

        # 6(a)/(b). Liquidated on the EXACT step equity crosses <= 0 — not before:
        # the first hold (bar 48 -> 49, still flat at 100) leaves the position
        # untouched and solvent; only the second hold (bar 49 -> 50, the crash)
        # flips `done` True.
        assert drop_payload["dones"] == [False, True]
        pre_crash_observation = drop_payload["observations"][0]
        pre_crash_cash, pre_crash_units, pre_crash_equity, pre_crash_leverage = pre_crash_observation[
            "portfolio_vector"
        ]
        assert pre_crash_units == pytest.approx(_BUY_UNITS)  # position still fully open
        assert pre_crash_equity == pytest.approx(_INITIAL_CASH)  # still flat, unchanged
        assert pre_crash_leverage == pytest.approx(2.9, rel=1e-12)

        liquidated_observation = drop_payload["observations"][1]
        liq_cash, liq_units, liq_equity, liq_leverage = liquidated_observation["portfolio_vector"]
        expected_liquidated_cash = _INITIAL_CASH - _BUY_UNITS * _FLAT_PRICE + _BUY_UNITS * _CRASH_PRICE  # -160.0
        assert liq_units == pytest.approx(0.0)  # whole position force-closed
        # 6(c). Equity lands EXACTLY on the liquidated cash value.
        assert liq_cash == pytest.approx(expected_liquidated_cash)
        assert liq_equity == pytest.approx(expected_liquidated_cash)
        assert liq_equity == pytest.approx(liq_cash)
        assert liq_leverage == pytest.approx(0.0)  # flat book: leverage is always 0.0, whatever the equity sign
        assert liquidated_observation["done"] is True
        assert liquidated_observation["market_window_handle"]["t"] == _CRASH_BAR_INDEX
        assert drop_payload["rewards"][1] == pytest.approx(
            expected_liquidated_cash - _INITIAL_CASH
        )  # -1160.0 equity_delta

        # 6(d) + Task 3 gap-fill: the liquidating step itself is a HOLD (no order
        # placed), so its own trace row correctly has no fill — see module
        # docstring for why. The trace line that DOES carry `has_filled_order`
        # for this position is the earlier buy's (already asserted above via
        # `buy_trace`); assert it again here through the file-based sink for the
        # full end-to-end path, and pin the liquidating row's shape too so a
        # future regression in "what the trace looks like on a fill-less
        # liquidation step" is caught.
        pre_crash_trace, liquidation_trace = drop_payload["trace"]
        assert pre_crash_trace["has_filled_order"] is False
        assert pre_crash_trace["fills"] == 0
        assert liquidation_trace["has_filled_order"] is False
        assert liquidation_trace["fills"] == 0
        assert liquidation_trace["done"] is True
        assert liquidation_trace["position_units"] == pytest.approx(0.0)
        assert liquidation_trace["cash"] == pytest.approx(expected_liquidated_cash)
        assert liquidation_trace["equity"] == pytest.approx(expected_liquidated_cash)
        assert liquidation_trace["leverage"] == pytest.approx(0.0)
        assert liquidation_trace["market_price"] == pytest.approx(_CRASH_PRICE)

        # File-based verification: 3 lines total now (1 buy + 2 holds). NOTE:
        # `index` is the per-`step_many`-call ordinal (resets to 0 on every call,
        # see `env.py::_build_trace_row`'s `index=i` loop variable) — it is NOT a
        # global step counter across calls, so identifying a specific row across
        # multiple step_many requests must use `t` (the true bar index), not
        # `index`.
        trace_after_drop = _read_jsonl(trace_path)
        assert len(trace_after_drop) == 3
        assert [row["t"] for row in trace_after_drop] == [48, 49, 50]
        liquidation_row = trace_after_drop[-1]
        assert liquidation_row["has_filled_order"] is False
        assert liquidation_row["done"] is True
        assert liquidation_row["cash"] == pytest.approx(expected_liquidated_cash)
        assert liquidation_row["position_units"] == pytest.approx(0.0)

        # Terminal state does not drift on further steps (mirrors
        # test_margin_enforcement.py::test_terminal_state_does_not_drift_on_further_steps,
        # now verified through the HTTP surface).
        further_hold = client.post("/v1/step_many", json={"actions": [_HOLD_ACTION]})
        assert further_hold.status_code == 200
        further_payload = further_hold.json()
        assert further_payload["dones"] == [True]
        further_cash, further_units, further_equity, _further_leverage = further_payload["observations"][0][
            "portfolio_vector"
        ]
        assert further_units == pytest.approx(0.0)
        assert further_cash == pytest.approx(expected_liquidated_cash)
        assert further_equity == pytest.approx(expected_liquidated_cash)
