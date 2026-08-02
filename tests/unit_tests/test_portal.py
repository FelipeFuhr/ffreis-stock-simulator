"""Direct unit tests for the market portal observation window."""

from __future__ import annotations

import numpy as np
import pytest

from stock_simulator.core import MarketArrays
from stock_simulator.portal import MarketPortal


def _arrays(closes: list[float]) -> MarketArrays:
    arr = np.array(closes, dtype=np.float32)
    return MarketArrays(
        open=arr,
        high=arr,
        low=arr,
        close=arr,
        n=len(closes),
    )


def _ohlcv_portal(n: int, observation_window: int) -> MarketPortal:
    """Portal with distinct per-column ramps so column mapping is verifiable."""
    base = np.arange(n, dtype=np.float32)
    return MarketPortal(
        market_arrays=MarketArrays(
            open=base + 100.0,
            high=base + 200.0,
            low=base + 50.0,
            close=base + 150.0,
            n=n,
        ),
        observation_window=observation_window,
        volume=(base + 1_000.0).astype(np.float32),
        taker_buy_volume=(base + 2_000.0).astype(np.float32),
    )


class TestCurrentPrice:
    def test_returns_close_at_index(self) -> None:
        portal = MarketPortal(_arrays([10.0, 20.0, 30.0]), observation_window=2)
        assert portal.current_price(0) == pytest.approx(10.0)
        assert portal.current_price(2) == pytest.approx(30.0)

    def test_returns_python_float(self) -> None:
        portal = MarketPortal(_arrays([10.0]), observation_window=1)
        result = portal.current_price(0)
        assert type(result) is float


class TestViewHandle:
    def test_initial_window_clamps_start_to_zero(self) -> None:
        # At t=0 with window=5, the start would be negative; should clamp to 0.
        portal = MarketPortal(_arrays([1.0] * 10), observation_window=5)
        handle = portal.view_handle(0)
        assert handle.start == 0
        assert handle.end == 1
        assert handle.t == 0

    def test_full_window_when_history_available(self) -> None:
        portal = MarketPortal(_arrays([float(i) for i in range(10)]), observation_window=4)
        handle = portal.view_handle(t=5)
        # start = max(0, 5+1-4) = 2; end = 6.
        assert handle.start == 2
        assert handle.end == 6
        assert handle.t == 5
        # current_price = close[5] = 5.0
        assert handle.current_price == pytest.approx(5.0)

    def test_window_at_last_bar(self) -> None:
        portal = MarketPortal(_arrays([float(i) for i in range(10)]), observation_window=4)
        handle = portal.view_handle(t=9)
        assert handle.start == 6
        assert handle.end == 10
        assert handle.t == 9
        assert handle.current_price == pytest.approx(9.0)

    @pytest.mark.parametrize("window", [1, 2, 5, 32])
    def test_window_size_constraint(self, window: int) -> None:
        portal = MarketPortal(_arrays([float(i) for i in range(64)]), observation_window=window)
        handle = portal.view_handle(t=50)
        # The slice (start, end) should be at most `window` wide.
        assert handle.end - handle.start <= window
        # When fully populated it should equal window.
        if 50 + 1 >= window:
            assert handle.end - handle.start == window

    def test_window_of_one(self) -> None:
        portal = MarketPortal(_arrays([1.0, 2.0, 3.0]), observation_window=1)
        handle = portal.view_handle(t=2)
        assert handle.start == 2
        assert handle.end == 3
        assert handle.current_price == pytest.approx(3.0)


class TestWindowContent:
    def test_default_window_matches_view_handle_and_maps_columns(self) -> None:
        portal = _ohlcv_portal(n=32, observation_window=4)
        handle = portal.view_handle(t=10)
        content = portal.window_content(t=10)
        assert (content.start, content.end, content.t) == (handle.start, handle.end, 10)
        # start = max(0, 10+1-4) = 7; end = 11.
        assert content.start == 7
        assert content.end == 11
        assert content.open == pytest.approx([107.0, 108.0, 109.0, 110.0])
        assert content.high == pytest.approx([207.0, 208.0, 209.0, 210.0])
        assert content.low == pytest.approx([57.0, 58.0, 59.0, 60.0])
        assert content.close == pytest.approx([157.0, 158.0, 159.0, 160.0])
        assert content.volume == pytest.approx([1007.0, 1008.0, 1009.0, 1010.0])
        assert content.taker_buy_volume == pytest.approx([2007.0, 2008.0, 2009.0, 2010.0])

    def test_episode_start_clamps_to_zero(self) -> None:
        # t (2) smaller than the observation window (8): start clamps to 0.
        portal = _ohlcv_portal(n=32, observation_window=8)
        content = portal.window_content(t=2)
        assert content.start == 0
        assert content.end == 3
        assert content.close == pytest.approx([150.0, 151.0, 152.0])
        assert content.volume == pytest.approx([1000.0, 1001.0, 1002.0])

    def test_explicit_bounds_return_history_slice(self) -> None:
        portal = _ohlcv_portal(n=64, observation_window=4)
        content = portal.window_content(t=40, start=5, end=9)
        assert content.start == 5
        assert content.end == 9
        assert content.close == pytest.approx([155.0, 156.0, 157.0, 158.0])

    def test_explicit_end_clamped_to_no_future_leakage(self) -> None:
        # Requesting far beyond t must never return rows with index > t.
        portal = _ohlcv_portal(n=64, observation_window=4)
        content = portal.window_content(t=10, start=0, end=10_000)
        assert content.end == 11  # t + 1, not 10_000
        assert len(content.close) == 11
        assert content.close[-1] == pytest.approx(160.0)  # close[10]

    def test_start_beyond_current_bar_returns_empty_window(self) -> None:
        portal = _ohlcv_portal(n=64, observation_window=4)
        content = portal.window_content(t=10, start=99, end=200)
        assert content.start == content.end == 11
        assert content.close == ()
        assert content.volume == ()
        assert content.taker_buy_volume == ()

    def test_negative_start_clamps_to_zero(self) -> None:
        portal = _ohlcv_portal(n=64, observation_window=4)
        content = portal.window_content(t=5, start=-10, end=3)
        assert content.start == 0
        assert content.end == 3

    def test_max_rows_cap_keeps_most_recent(self) -> None:
        portal = _ohlcv_portal(n=64, observation_window=4)
        content = portal.window_content(t=50, start=0, end=51, max_rows=5)
        # Capped to the most recent 5 bars ending at index 50.
        assert content.end == 51
        assert content.start == 46
        assert len(content.close) == 5
        assert content.close[-1] == pytest.approx(200.0)  # close[50] = 50 + 150

    def test_missing_volume_yields_zero_filled_column(self) -> None:
        # A portal constructed without a volume array (legacy path) still returns
        # a correctly-sized, zero-filled volume column.
        portal = MarketPortal(_arrays([float(i) for i in range(10)]), observation_window=3)
        content = portal.window_content(t=5)
        assert content.start == 3
        assert content.end == 6
        assert content.volume == (0.0, 0.0, 0.0)
        assert content.close == pytest.approx([3.0, 4.0, 5.0])

    def test_missing_taker_buy_volume_yields_zero_filled_column(self) -> None:
        # A portal constructed without a taker_buy_volume array (e.g. CSV-backed
        # data, which doesn't carry the column) still returns a correctly-sized,
        # zero-filled taker_buy_volume column.
        portal = MarketPortal(_arrays([float(i) for i in range(10)]), observation_window=3)
        content = portal.window_content(t=5)
        assert content.start == 3
        assert content.end == 6
        assert content.taker_buy_volume == (0.0, 0.0, 0.0)
        assert content.close == pytest.approx([3.0, 4.0, 5.0])
