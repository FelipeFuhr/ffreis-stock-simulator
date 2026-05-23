"""Direct unit tests for the order summary module."""

from __future__ import annotations

import numpy as np
import pytest

from stock_simulator.core import CoreState, initial_core_state
from stock_simulator.orders import OrderSummary, summarize_orders


def _state_with_orders(active_sides: list[int], max_orders: int = 8) -> CoreState:
    """Build a CoreState whose first len(active_sides) order slots are active.

    active_sides values: 1=buy, -1=sell. Padding slots are inactive.
    """
    state = initial_core_state(initial_cash=1000.0, max_orders=max_orders)
    n = len(active_sides)
    state.order_active[:n] = 1
    state.order_side[:n] = np.array(active_sides, dtype=np.int8)
    return state


class TestSummarizeOrders:
    def test_empty_book(self) -> None:
        state = initial_core_state(initial_cash=1000.0, max_orders=8)
        summary = summarize_orders(state)
        assert summary == OrderSummary(open_orders=0, buy_open_orders=0, sell_open_orders=0)

    def test_all_buys(self) -> None:
        state = _state_with_orders([1, 1, 1])
        summary = summarize_orders(state)
        assert summary.open_orders == 3
        assert summary.buy_open_orders == 3
        assert summary.sell_open_orders == 0

    def test_all_sells(self) -> None:
        state = _state_with_orders([-1, -1])
        summary = summarize_orders(state)
        assert summary.open_orders == 2
        assert summary.buy_open_orders == 0
        assert summary.sell_open_orders == 2

    def test_mixed_buy_sell(self) -> None:
        state = _state_with_orders([1, -1, 1, -1, 1])
        summary = summarize_orders(state)
        assert summary.open_orders == 5
        assert summary.buy_open_orders == 3
        assert summary.sell_open_orders == 2

    def test_inactive_orders_excluded(self) -> None:
        # Two active orders, then two inactive padding slots flagged with side
        # but order_active=0; these should not count.
        state = initial_core_state(initial_cash=1000.0, max_orders=4)
        state.order_active[:2] = 1
        state.order_side[:2] = np.array([1, -1], dtype=np.int8)
        # Leave indices 2-3 with side set but active=0.
        state.order_side[2:] = np.array([1, -1], dtype=np.int8)
        summary = summarize_orders(state)
        assert summary.open_orders == 2
        assert summary.buy_open_orders == 1
        assert summary.sell_open_orders == 1


class TestOrderSummaryVector:
    def test_to_vector_layout(self) -> None:
        v = OrderSummary(open_orders=5, buy_open_orders=3, sell_open_orders=2).to_vector()
        assert v.shape == (3,)
        assert v.dtype == np.float64
        assert v.tolist() == [5.0, 3.0, 2.0]

    def test_to_vector_zero_state(self) -> None:
        v = OrderSummary(0, 0, 0).to_vector()
        assert v.tolist() == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize(
        "open_total,buys,sells",
        [(0, 0, 0), (1, 1, 0), (1, 0, 1), (10, 4, 6), (3, 3, 0)],
    )
    def test_to_vector_round_trip(self, open_total: int, buys: int, sells: int) -> None:
        summary = OrderSummary(open_total, buys, sells)
        v = summary.to_vector()
        assert v[0] == open_total
        assert v[1] == buys
        assert v[2] == sells
