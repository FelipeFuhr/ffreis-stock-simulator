from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class MarketData:
    """Container with market time series arrays used by the simulator.

    Parameters
    ----------
    df
        Input frame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, and ``volume``.
    """

    def __init__(self, df: pd.DataFrame):
        self.ts: NDArray[np.object_] = df["timestamp"].to_numpy()
        self.open: NDArray[np.float32] = df["open"].to_numpy(dtype=np.float32)
        self.high: NDArray[np.float32] = df["high"].to_numpy(dtype=np.float32)
        self.low: NDArray[np.float32] = df["low"].to_numpy(dtype=np.float32)
        self.close: NDArray[np.float32] = df["close"].to_numpy(dtype=np.float32)
        self.volume: NDArray[np.float32] = df["volume"].to_numpy(dtype=np.float32)
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
        df = pd.read_csv(path)
        return cls(df)
