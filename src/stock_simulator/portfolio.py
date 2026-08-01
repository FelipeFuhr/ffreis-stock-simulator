from __future__ import annotations

from dataclasses import dataclass

from numpy import asarray as np_asarray
from numpy import float64 as np_float64
from numpy.typing import NDArray

from .core import CoreState, leverage_ratio

# Reported leverage is always finite. An open position carried against non-positive
# equity has mathematically unbounded leverage, and reporting `inf` there used to
# serialize to JSON `null` (both FastAPI and the JSONL replay writer), which crashed
# clients that expected a number. The engine now liquidates such a book at the step
# boundary, so this ceiling is only reachable from a hand-built state — but the value
# stays finite so no transport can ever emit `null` for `leverage`.
INSOLVENT_LEVERAGE: float = 1e9


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    units: float
    equity: float
    leverage: float

    def to_vector(self) -> NDArray[np_float64]:
        return np_asarray(
            [self.cash, self.units, self.equity, self.leverage],
            dtype=np_float64,
        )


def snapshot_from_state(state: CoreState, price: float) -> PortfolioSnapshot:
    cash = float(state.portfolio[0])
    units = float(state.portfolio[1])
    equity = cash + units * price
    leverage = min(leverage_ratio(units, price, equity), INSOLVENT_LEVERAGE)
    return PortfolioSnapshot(cash=cash, units=units, equity=equity, leverage=leverage)
