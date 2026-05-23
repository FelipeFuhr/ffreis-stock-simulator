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
