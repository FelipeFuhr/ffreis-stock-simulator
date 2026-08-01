"""STOCK_SIM_TRACE_JSONL (N10): server-side JSONL trace sink.

A debugging/verification aid, distinct from the structured Recorder/RecordedStep
episode-replay port (see tests/unit_tests/test_recorder.py) — the two mechanisms
cover different concerns; see trace_writer.py's module docstring for why they are
not merged into one.
"""

from __future__ import annotations

from collections.abc import Callable
from json import loads as json_loads
from pathlib import Path

import pytest
from numpy import float64 as np_float64
from numpy.typing import NDArray

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.trace_writer import TraceJsonlWriter
from stock_simulator.types import Action, StepTraceRow


def _row(index: int, *, filled_order_slot: int | None = None, exec_price: float | None = None) -> StepTraceRow:
    return StepTraceRow(
        index=index,
        side_code=1,
        requested_units=1.0,
        order_type_code=0,
        has_limit_price=False,
        limit_price=None,
        fills=1 if filled_order_slot is not None else 0,
        reward=0.5,
        done=False,
        t=index,
        cash=100.0,
        position_units=1.0,
        equity=101.0,
        leverage=0.1,
        open_orders=0,
        market_price=100.0,
        filled_order_slot=filled_order_slot,
        exec_price=exec_price,
    )


class TestTraceJsonlWriter:
    def test_writes_one_jsonl_line_per_row(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        writer = TraceJsonlWriter(path)
        for i in range(3):
            writer.write(_row(i, filled_order_slot=0, exec_price=100.0 + i))
        writer.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        parsed = [json_loads(line) for line in lines]
        assert [p["index"] for p in parsed] == [0, 1, 2]
        assert parsed[1]["exec_price"] == 101.0
        assert parsed[1]["filled_order_slot"] == 0

    def test_appends_across_separate_writer_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        first = TraceJsonlWriter(path)
        first.write(_row(0))
        first.close()

        second = TraceJsonlWriter(path)
        second.write(_row(1))
        second.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "trace.jsonl"
        writer = TraceJsonlWriter(path)
        writer.write(_row(0))
        writer.close()

        assert path.parent.is_dir()
        assert path.exists()

    def test_row_without_a_fill_serializes_null_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        writer = TraceJsonlWriter(path)
        writer.write(_row(0))
        writer.close()

        parsed = json_loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert parsed["filled_order_slot"] is None
        assert parsed["exec_price"] is None
        assert parsed["fills"] == 0


class TestMarketEnvTraceSinkWiring:
    def test_step_many_writes_every_step_regardless_of_include_trace(
        self,
        tmp_path: Path,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
    ) -> None:
        path = tmp_path / "trace.jsonl"
        writer = TraceJsonlWriter(path)
        env = MarketEnv(
            data=market_data_factory(n=8, slope=0.1, spread=0.5, volume=10_000.0),
            cfg=GameConfig(seed=7, use_numba=False, market_latency_bars=0),
            trace_sink=writer,
        )
        env.reset(seed=7)
        actions = [
            Action(side="hold"),
            Action(side="buy", units=1.0, order_type="market"),
            Action(side="hold"),
        ]
        encoded = encode_actions(actions)

        # include_trace defaults to False — the response carries no trace rows...
        _, _, _, trace_rows = env.step_many(encoded)
        assert trace_rows == ()
        env.close()

        # ...but the sink still received one JSONL line per step, unconditionally.
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(actions)
        parsed = [json_loads(line) for line in lines]
        assert [p["index"] for p in parsed] == [0, 1, 2]
        assert parsed[1]["fills"] == 1
        assert parsed[1]["filled_order_slot"] == 0
        assert parsed[1]["exec_price"] is not None

    def test_no_sink_configured_writes_no_file(
        self,
        tmp_path: Path,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
    ) -> None:
        path = tmp_path / "should-not-exist.jsonl"
        env = MarketEnv(
            data=market_data_factory(n=8),
            cfg=GameConfig(seed=1, use_numba=False),
        )
        env.reset(seed=1)
        env.step_many(encode_actions([Action(side="hold")]))
        env.close()  # no-op: no sink was configured

        assert not path.exists()

    def test_sink_also_receives_rows_when_include_trace_is_true(
        self,
        tmp_path: Path,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
    ) -> None:
        path = tmp_path / "trace.jsonl"
        writer = TraceJsonlWriter(path)
        env = MarketEnv(
            data=market_data_factory(n=8),
            cfg=GameConfig(seed=3, use_numba=False),
            trace_sink=writer,
        )
        env.reset(seed=3)
        actions = [Action(side="hold"), Action(side="hold")]
        encoded = encode_actions(actions)

        _, _, _, trace_rows = env.step_many(encoded, include_trace=True)
        env.close()

        assert len(trace_rows) == 2
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_liquidating_step_trace_row_has_no_fill_but_reflects_liquidated_state(
        self,
        tmp_path: Path,
        market_data_factory: Callable[..., MarketData],
        encode_actions: Callable[[list[Action]], NDArray[np_float64]],
    ) -> None:
        """Margin enforcement + trace recording, exercised together (gap found while
        building the full HTTP lifecycle integration test — see
        tests/integration_tests/test_margin_lifecycle.py).

        A mark-to-market liquidation (`settle_insolvency` in core.py) force-closes
        the book on a step where no order is placed or filled — it only marks the
        existing position to the new bar's close. So the liquidating step's own
        trace row must show `has_filled_order=False`/`fills=0`/`filled_order_slot
        is None`/`exec_price is None` even though `done=True` and the portfolio
        fields already reflect the *post*-liquidation state (`position_units=0.0`,
        `cash == equity`, `leverage=0.0`). Neither test_margin_enforcement.py
        (which never reads the trace sink) nor the rest of this file's
        TestMarketEnvTraceSinkWiring (which never opens a leveraged/liquidated
        position) exercised this combination before.
        """
        path = tmp_path / "trace.jsonl"
        writer = TraceJsonlWriter(path)
        # Bar 0->1 flat at 100, bar 2 halves to 50 — the same shape as
        # test_margin_enforcement.py::TestInsolvencyTermination's mark-to-market
        # wipeout case, driven here through step_many instead of step so the
        # trace sink (only wired into step_many) actually has something to record.
        env = MarketEnv(
            data=market_data_factory(close=[100.0, 100.0, 50.0, 50.0, 50.0]),
            cfg=GameConfig(
                seed=1,
                use_numba=False,
                market_latency_bars=0,
                initial_cash=1000.0,
                max_leverage=3.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                partial_fill_min=1.0,
                partial_fill_max=1.0,
            ),
            trace_sink=writer,
        )
        env.reset(seed=1)

        # Open exactly at the 3.0x cap: 30 units at 100 on 1000 equity (not
        # clipped — 30*100 == 3.0*1000, the boundary is inclusive).
        _, _, opened_dones, _ = env.step_many(encode_actions([Action(side="buy", units=30.0, order_type="market")]))
        assert opened_dones.tolist() == [False]

        # Price halves (100 -> 50): equity = -2000 + 30*50 = -500 <= 0 -> liquidated.
        _, _, liquidated_dones, _ = env.step_many(encode_actions([Action(side="hold")]))
        assert liquidated_dones.tolist() == [True]
        env.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        liquidation_row = json_loads(lines[-1])
        assert liquidation_row["fills"] == 0
        assert liquidation_row["has_filled_order"] is False
        assert liquidation_row["filled_order_slot"] is None
        assert liquidation_row["exec_price"] is None
        assert liquidation_row["done"] is True
        assert liquidation_row["position_units"] == pytest.approx(0.0)
        assert liquidation_row["cash"] == pytest.approx(-500.0)
        assert liquidation_row["equity"] == pytest.approx(-500.0)
        assert liquidation_row["equity"] == pytest.approx(liquidation_row["cash"])
        assert liquidation_row["leverage"] == pytest.approx(0.0)
