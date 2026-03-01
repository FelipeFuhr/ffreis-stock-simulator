from __future__ import annotations

from dataclasses import dataclass

from numpy import asarray as np_asarray, float64 as np_float64, inf as np_inf
from numpy.typing import NDArray

from .core import CoreState


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
    leverage = float(np_inf) if equity <= 0 else abs(units * price) / equity
    return PortfolioSnapshot(cash=cash, units=units, equity=equity, leverage=leverage)
