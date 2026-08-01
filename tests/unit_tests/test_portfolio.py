"""Direct unit tests for portfolio snapshot module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_simulator.core import CoreState, initial_core_state
from stock_simulator.portfolio import INSOLVENT_LEVERAGE, PortfolioSnapshot, snapshot_from_state


def _state(cash: float, units: float) -> CoreState:
    s = initial_core_state(initial_cash=cash, max_orders=4)
    s.portfolio[0] = cash
    s.portfolio[1] = units
    return s


class TestSnapshotFromState:
    def test_flat_position_no_leverage(self) -> None:
        state = _state(cash=1000.0, units=0.0)
        snap = snapshot_from_state(state, price=100.0)
        assert snap.cash == pytest.approx(1000.0)
        assert snap.units == pytest.approx(0.0)
        assert snap.equity == pytest.approx(1000.0)
        assert snap.leverage == pytest.approx(0.0)

    def test_long_position_leverage(self) -> None:
        # 500 cash + 10 units @ 100 = 1000 equity; exposure = 1000; leverage = 1.0
        state = _state(cash=500.0, units=5.0)
        snap = snapshot_from_state(state, price=100.0)
        assert snap.equity == pytest.approx(1000.0)
        assert snap.leverage == pytest.approx(0.5)

    def test_short_position_leverage_uses_absolute_exposure(self) -> None:
        # cash=1500, units=-5 at price=100 → equity = 1500 - 500 = 1000; exposure 500.
        state = _state(cash=1500.0, units=-5.0)
        snap = snapshot_from_state(state, price=100.0)
        assert snap.equity == pytest.approx(1000.0)
        assert snap.leverage == pytest.approx(0.5)

    def test_negative_equity_with_open_position_reports_finite_ceiling(self) -> None:
        # Massive short: cash=100, units=-10 at price=100 → equity = -900.
        # Leverage is mathematically unbounded here; the reported value is the finite
        # INSOLVENT_LEVERAGE ceiling so JSON transports never emit `null`.
        state = _state(cash=100.0, units=-10.0)
        snap = snapshot_from_state(state, price=100.0)
        assert snap.equity < 0.0
        assert math.isfinite(snap.leverage)
        assert snap.leverage == pytest.approx(INSOLVENT_LEVERAGE)

    def test_zero_equity_flat_book_has_zero_leverage(self) -> None:
        # cash=0 and units=0 yields equity=0 — but with no exposure there is nothing
        # to be levered, so leverage is 0.0 rather than an unbounded value.
        state = _state(cash=0.0, units=0.0)
        snap = snapshot_from_state(state, price=100.0)
        assert snap.equity == pytest.approx(0.0)
        assert snap.leverage == pytest.approx(0.0)

    def test_liquidated_book_with_negative_cash_has_zero_leverage(self) -> None:
        # Post-liquidation shape: units zeroed, cash carries the realized loss.
        state = _state(cash=-500.0, units=0.0)
        snap = snapshot_from_state(state, price=100.0)
        assert snap.equity == pytest.approx(-500.0)
        assert snap.leverage == pytest.approx(0.0)

    def test_to_vector_layout(self) -> None:
        snap = PortfolioSnapshot(cash=1.0, units=2.0, equity=3.0, leverage=4.0)
        v = snap.to_vector()
        assert v.shape == (4,)
        assert v.dtype == np.float64
        assert v.tolist() == [1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("price", [0.01, 1.0, 50.0, 1_000.0, 10_000.0])
    def test_leverage_scales_with_exposure(self, price: float) -> None:
        state = _state(cash=10_000.0, units=10.0)
        snap = snapshot_from_state(state, price=price)
        expected_exposure = abs(10.0 * price)
        expected_equity = 10_000.0 + 10.0 * price
        if expected_equity > 0:
            assert snap.leverage == pytest.approx(expected_exposure / expected_equity)
