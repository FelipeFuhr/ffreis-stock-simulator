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
