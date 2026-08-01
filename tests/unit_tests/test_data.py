"""MarketData unit tests, including from_csv parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame

from stock_simulator.data import MarketData


def _csv(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "market.csv"
    f.write_text(content)
    return f


VALID_CSV = (
    "timestamp,open,high,low,close,volume\n"
    "2024-01-01 00:00:00,100.0,101.5,99.5,100.5,1000\n"
    "2024-01-01 01:00:00,100.5,102.0,100.0,101.0,1100\n"
    "2024-01-01 02:00:00,101.0,103.0,100.8,102.5,1200\n"
)


class TestMarketDataFromDataFrame:
    def test_simple_construction(self) -> None:
        df = DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02"],
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.5, 11.5],
                "volume": [100.0, 200.0],
            }
        )
        md = MarketData(df)
        assert md.n == 2
        assert md.close.dtype == np.float32
        assert md.high.dtype == np.float32
        assert md.low.dtype == np.float32
        assert md.open.dtype == np.float32
        assert md.volume.dtype == np.float32


class TestMarketDataFromCsv:
    def test_parses_valid_csv(self, tmp_path: Path) -> None:
        path = _csv(tmp_path, VALID_CSV)
        md = MarketData.from_csv(str(path))
        assert md.n == 3
        assert md.close[0] == pytest.approx(100.5)
        assert md.close[-1] == pytest.approx(102.5)
        assert md.volume[1] == pytest.approx(1100.0)

    def test_missing_column_raises_key_error(self, tmp_path: Path) -> None:
        bad = "timestamp,open,high,low,close\n2024-01-01,100,101,99,100\n"
        path = _csv(tmp_path, bad)
        with pytest.raises(KeyError):
            MarketData.from_csv(str(path))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MarketData.from_csv(str(tmp_path / "nope.csv"))

    def test_dtype_coercion_from_string_numbers(self, tmp_path: Path) -> None:
        # pandas reads numeric strings into floats automatically.
        path = _csv(tmp_path, VALID_CSV)
        md = MarketData.from_csv(str(path))
        # Each column is downcast to float32.
        assert all(md.open.dtype == np.float32 for _ in range(1))

    def test_empty_csv_raises(self, tmp_path: Path) -> None:
        # Only header — no rows.
        path = _csv(tmp_path, "timestamp,open,high,low,close,volume\n")
        md = MarketData.from_csv(str(path))
        assert md.n == 0


class TestMarketDataFromParquet:
    def test_parses_valid_parquet(self, tmp_path: Path) -> None:
        path = tmp_path / "market.parquet"
        DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
                "open": [100.0, 100.5, 101.0],
                "high": [101.5, 102.0, 103.0],
                "low": [99.5, 100.0, 100.8],
                "close": [100.5, 101.0, 102.5],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        ).to_parquet(path)
        md = MarketData.from_parquet(str(path))
        assert md.n == 3
        assert md.close[0] == pytest.approx(100.5)
        assert md.close[-1] == pytest.approx(102.5)
        assert md.volume[1] == pytest.approx(1100.0)

    def test_maps_open_time_to_timestamp(self, tmp_path: Path) -> None:
        # Binance-style BTC parquets label the bar timestamp ``open_time``.
        path = tmp_path / "btc.parquet"
        DataFrame(
            {
                "open_time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                "open": [42_000.0, 42_100.0],
                "high": [42_500.0, 42_600.0],
                "low": [41_800.0, 41_900.0],
                "close": [42_100.0, 42_050.0],
                "volume": [12.5, 9.75],
            }
        ).to_parquet(path)
        md = MarketData.from_parquet(str(path))
        assert md.n == 2
        assert md.close[0] == pytest.approx(42_100.0)
        assert md.volume[0] == pytest.approx(12.5)
        # The renamed frame exposes the mapped timestamp values.
        assert str(md.ts[0]).startswith("2024-01-01")

    def test_missing_column_raises_key_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.parquet"
        DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                # volume intentionally omitted
            }
        ).to_parquet(path)
        with pytest.raises(KeyError):
            MarketData.from_parquet(str(path))
