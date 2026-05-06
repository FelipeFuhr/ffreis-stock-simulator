from __future__ import annotations

from dataclasses import dataclass

from .core import MarketArrays
from .types import MarketWindowViewHandle


@dataclass(frozen=True)
class MarketPortal:
    market_arrays: MarketArrays
    observation_window: int

    def current_price(self, t: int) -> float:
        return float(self.market_arrays.close[t])

    def view_handle(self, t: int) -> MarketWindowViewHandle:
        start = max(0, t + 1 - self.observation_window)
        end = t + 1
        return MarketWindowViewHandle(
            start=start,
            end=end,
            t=t,
            current_price=self.current_price(t),
        )
