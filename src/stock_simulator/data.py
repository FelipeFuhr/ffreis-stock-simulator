from __future__ import annotations

from numpy import float32 as np_float32, object_ as np_object_
from pandas import DataFrame as pd_DataFrame, read_csv as pd_read_csv
from numpy.typing import NDArray


class MarketData:
    """Container with market time series arrays used by the simulator.

    Parameters
    ----------
    df
        Input frame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, and ``volume``.
    """

    def __init__(self, df: pd_DataFrame):
        self.ts: NDArray[np_object_] = df["timestamp"].to_numpy()
        self.open: NDArray[np_float32] = df["open"].to_numpy(dtype=np_float32)
        self.high: NDArray[np_float32] = df["high"].to_numpy(dtype=np_float32)
        self.low: NDArray[np_float32] = df["low"].to_numpy(dtype=np_float32)
        self.close: NDArray[np_float32] = df["close"].to_numpy(dtype=np_float32)
        self.volume: NDArray[np_float32] = df["volume"].to_numpy(dtype=np_float32)
        self.n: int = len(df)

    @classmethod
    def from_csv(cls, path: str) -> MarketData:
        """Load market data from CSV.

        Parameters
        ----------
        path
            CSV file path.

        Returns
        -------
        MarketData
            Parsed market data container.
        """
        df = pd_read_csv(path)
        return cls(df)
