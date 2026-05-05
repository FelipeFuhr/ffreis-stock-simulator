"""Execution summaries derived from low-level core step output."""

from __future__ import annotations

from dataclasses import dataclass

from .core import CoreStepOutput


@dataclass(frozen=True)
class ExecutionSummary:
    """Aggregated execution information for one environment step."""

    fills: int
    equity_delta: float
    done: bool


def summarize_execution(output: CoreStepOutput) -> ExecutionSummary:
    """Convert core engine output into a compact execution summary."""
    return ExecutionSummary(
        fills=output.fills,
        equity_delta=float(output.equity_delta),
        done=output.state.done,
    )
