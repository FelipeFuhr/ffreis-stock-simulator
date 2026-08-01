from __future__ import annotations

from dataclasses import dataclass

from numpy import float32 as np_float32
from numpy.typing import NDArray

from .core import MarketArrays
from .types import MarketWindowContent, MarketWindowViewHandle

# Hard cap on rows returned by ``window_content`` so an explicit start/end range
# cannot produce an accidentally huge payload. Callers requesting a wider span
# receive the most recent ``_MAX_WINDOW_ROWS`` bars.
_MAX_WINDOW_ROWS = 100_000


def _to_float_tuple(values: NDArray[np_float32]) -> tuple[float, ...]:
    return tuple(float(value) for value in values.tolist())


@dataclass(frozen=True)
class MarketPortal:
    market_arrays: MarketArrays
    observation_window: int
    volume: NDArray[np_float32] | None = None

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

    def window_content(
        self,
        t: int,
        start: int | None = None,
        end: int | None = None,
        max_rows: int = _MAX_WINDOW_ROWS,
    ) -> MarketWindowContent:
        """Return raw OHLCV rows for the resolved window ``[start, end)``.

        With no bounds, mirrors :meth:`view_handle` (the current observation
        window). With explicit ``start``/``end``, the range is clamped so no bar
        with index greater than ``t`` is ever served (no future leakage) and the
        payload length is capped at ``max_rows`` (keeping the most recent bars).

        Parameters
        ----------
        t
            Current bar index. Establishes the upper bound ``end <= t + 1``.
        start, end
            Optional inclusive/exclusive bounds. Negative or out-of-range values
            are clamped into ``[0, t + 1)``.
        max_rows
            Maximum number of rows to return.

        Returns
        -------
        MarketWindowContent
            Window metadata plus per-bar open/high/low/close/volume values.
        """
        upper = t + 1
        if start is None and end is None:
            resolved_start = max(0, upper - self.observation_window)
            resolved_end = upper
        else:
            resolved_start = 0 if start is None else max(0, start)
            resolved_end = upper if end is None else min(end, upper)
        resolved_end = max(0, min(resolved_end, upper))
        resolved_start = max(0, min(resolved_start, resolved_end))
        if resolved_end - resolved_start > max_rows:
            resolved_start = resolved_end - max_rows

        if self.volume is not None:
            volume = _to_float_tuple(self.volume[resolved_start:resolved_end])
        else:
            volume = tuple(0.0 for _ in range(resolved_end - resolved_start))

        return MarketWindowContent(
            start=resolved_start,
            end=resolved_end,
            t=t,
            open=_to_float_tuple(self.market_arrays.open[resolved_start:resolved_end]),
            high=_to_float_tuple(self.market_arrays.high[resolved_start:resolved_end]),
            low=_to_float_tuple(self.market_arrays.low[resolved_start:resolved_end]),
            close=_to_float_tuple(self.market_arrays.close[resolved_start:resolved_end]),
            volume=volume,
        )
