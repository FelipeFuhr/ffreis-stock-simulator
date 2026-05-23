"""Direct unit tests for the execution summary module."""

from __future__ import annotations

import numpy as np
import pytest

from stock_simulator.core import CoreState, CoreStepOutput, initial_core_state
from stock_simulator.execution import ExecutionSummary, summarize_execution


def _state(*, done: bool) -> CoreState:
    s = initial_core_state(initial_cash=1000.0, max_orders=4)
    return CoreState(
        t=s.t,
        done=done,
        portfolio=s.portfolio,
        order_active=s.order_active,
        order_side=s.order_side,
        order_type=s.order_type,
        order_units=s.order_units,
        order_limit_price=s.order_limit_price,
        order_eligible_t=s.order_eligible_t,
        order_ttl=s.order_ttl,
    )


class TestSummarizeExecution:
    def test_passes_through_fills(self) -> None:
        out = CoreStepOutput(state=_state(done=False), fills=3, equity_delta=12.5)
        summary = summarize_execution(out)
        assert summary.fills == 3

    def test_coerces_equity_delta_to_float(self) -> None:
        # CoreStepOutput.equity_delta may be a numpy float; summary should be plain.
        out = CoreStepOutput(
            state=_state(done=False),
            fills=0,
            equity_delta=np.float64(5.5),
        )
        summary = summarize_execution(out)
        assert isinstance(summary.equity_delta, float)
        assert summary.equity_delta == pytest.approx(5.5)

    def test_propagates_done_flag(self) -> None:
        for done in (True, False):
            out = CoreStepOutput(state=_state(done=done), fills=0, equity_delta=0.0)
            assert summarize_execution(out).done is done

    def test_negative_equity_delta(self) -> None:
        out = CoreStepOutput(state=_state(done=False), fills=1, equity_delta=-10.0)
        summary = summarize_execution(out)
        assert summary.equity_delta == pytest.approx(-10.0)

    def test_zero_fills_and_zero_delta(self) -> None:
        out = CoreStepOutput(state=_state(done=False), fills=0, equity_delta=0.0)
        assert summarize_execution(out) == ExecutionSummary(fills=0, equity_delta=0.0, done=False)
