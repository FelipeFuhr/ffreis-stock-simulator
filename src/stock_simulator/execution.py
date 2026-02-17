from __future__ import annotations

from dataclasses import dataclass

from .core import CoreStepOutput


@dataclass(frozen=True)
class ExecutionSummary:
    fills: int
    equity_delta: float
    done: bool


def summarize_execution(output: CoreStepOutput) -> ExecutionSummary:
    return ExecutionSummary(
        fills=output.fills,
        equity_delta=float(output.equity_delta),
        done=output.state.done,
    )
