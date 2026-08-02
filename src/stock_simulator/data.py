from __future__ import annotations

from numpy import float32 as np_float32
from numpy import object_ as np_object_
from numpy.typing import NDArray
from pandas import DataFrame as pd_DataFrame
from pandas import read_csv as pd_read_csv
from pandas import read_parquet as pd_read_parquet

_TIMESTAMP_COLUMN = "timestamp"
_OPEN_TIME_ALIAS = "open_time"
_TAKER_BUY_VOLUME_COLUMN = "taker_buy_base_volume"


# scan-fix(pylint:R0902): 8 attrs is the natural shape of an OHLCV+volume+
# taker_buy_volume+timestamp+count data container, not accumulated complexity —
# splitting it would fight the class's purpose, not simplify it.
# pylint: disable-next=too-many-instance-attributes
class MarketData:
    """Container with market time series arrays used by the simulator.

    Parameters
    ----------
    df
        Input frame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, and ``volume``. An optional ``taker_buy_base_volume`` column
        (present on Binance-style kline parquets) is picked up when available;
        ``taker_buy_volume`` is ``None`` otherwise.
    """

    def __init__(self, df: pd_DataFrame):
        self.ts: NDArray[np_object_] = df["timestamp"].to_numpy()
        self.open: NDArray[np_float32] = df["open"].to_numpy(dtype=np_float32)
        self.high: NDArray[np_float32] = df["high"].to_numpy(dtype=np_float32)
        self.low: NDArray[np_float32] = df["low"].to_numpy(dtype=np_float32)
        self.close: NDArray[np_float32] = df["close"].to_numpy(dtype=np_float32)
        self.volume: NDArray[np_float32] = df["volume"].to_numpy(dtype=np_float32)
        self.taker_buy_volume: NDArray[np_float32] | None = self._optional_column(df, _TAKER_BUY_VOLUME_COLUMN)
        self.n: int = len(df)

    @staticmethod
    def _optional_column(df: pd_DataFrame, name: str) -> NDArray[np_float32] | None:
        return df[name].to_numpy(dtype=np_float32) if name in df.columns else None

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
