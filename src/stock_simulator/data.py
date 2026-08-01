from __future__ import annotations

from numpy import float32 as np_float32
from numpy import object_ as np_object_
from numpy.typing import NDArray
from pandas import DataFrame as pd_DataFrame
from pandas import read_csv as pd_read_csv
from pandas import read_parquet as pd_read_parquet

_TIMESTAMP_COLUMN = "timestamp"
_OPEN_TIME_ALIAS = "open_time"


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

    @classmethod
    def from_parquet(cls, path: str) -> MarketData:
        """Load market data from a Parquet file.

        The Binance-style BTC parquets label the bar timestamp ``open_time``
        rather than ``timestamp``; that column is renamed on load so the same
        OHLCV contract as :meth:`from_csv` applies. A frame missing any required
        column raises ``KeyError`` from :class:`MarketData` construction, matching
        the CSV loader's behaviour.

        Parameters
        ----------
        path
            Parquet file path. Requires a Parquet engine (``pyarrow``).

        Returns
        -------
        MarketData
            Parsed market data container.
        """
        df = pd_read_parquet(path)
        if _TIMESTAMP_COLUMN not in df.columns and _OPEN_TIME_ALIAS in df.columns:
            df = df.rename(columns={_OPEN_TIME_ALIAS: _TIMESTAMP_COLUMN})
        return cls(df)
