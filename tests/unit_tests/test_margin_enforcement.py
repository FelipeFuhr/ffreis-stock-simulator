"""Margin enforcement: the two-tier margin model and its liquidation triggers.

The engine separates the two margin concepts a real exchange keeps apart:

* **Initial margin** — `GameConfig.max_leverage`, applied once per order by
  `clip_units_for_leverage`. It used to be parsed and never read, so an agent could
  accumulate unbounded leverage.
* **Maintenance margin** — `GameConfig.maintenance_margin_rate`/`maintenance_amount`,
  re-evaluated at every mark by `settle_maintenance_margin`. Far more permissive than
  the order-time cap, and the trigger that force-closes an already-open position while
  it still has positive equity.

`settle_insolvency` (`equity <= 0`) remains behind both as the deep-tail backstop for a
single bar that gaps clean past the maintenance threshold. These tests pin all three
pieces, the fact that leverage may legitimately drift above `max_leverage` between
fills without anything firing, and the python/numba parity of every code path involved.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import textwrap
from collections.abc import Callable
from typing import Any

import pytest
from numpy import asarray as np_asarray
from numpy import float32 as np_float32
from numpy import float64 as np_float64

from stock_simulator.config import MAINTENANCE_MARGIN_TIERS, GameConfig
from stock_simulator.core import (
    CoreState,
    CoreStepOutput,
    MarketArrays,
    _clip_units_for_leverage_jit,
    _is_below_maintenance_margin_jit,
    _liquidate_position_jit,
    _maintenance_margin_jit,
    _settle_insolvency_jit,
    _settle_maintenance_margin_jit,
    clip_units_for_leverage,
    initial_core_state,
    is_below_maintenance_margin,
    leverage_ratio,
    liquidate_position,
    maintenance_margin,
    settle_insolvency,
    settle_maintenance_margin,
    step_core,
    step_core_numba,
)
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.portfolio import INSOLVENT_LEVERAGE, snapshot_from_state
from stock_simulator.types import Action, EnvStateModel, MarketWindowViewHandleModel, ObservationModel

_FULL_FILL = np_asarray([1.0, 1.0], dtype=np_float64)


def _market(closes: list[float], spread: float = 0.5) -> MarketArrays:
    close = np_asarray(closes, dtype=np_float32)
    return MarketArrays(
        open=close.copy(),
        high=(close + spread).astype(np_float32),
        low=(close - spread).astype(np_float32),
        close=close,
        n=len(closes),
    )


def _cfg(**overrides: float | bool | int) -> GameConfig:
    base: dict[str, float | bool | int] = {
        "use_numba": False,
        "market_latency_bars": 0,
        "limit_ttl_bars": 10,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "max_leverage": 2.0,
    }
    base.update(overrides)
    return GameConfig(**base)  # type: ignore[arg-type]


def _state(cash: float, units: float = 0.0) -> CoreState:
    state = initial_core_state(initial_cash=cash, max_orders=2)
    state.portfolio[1] = units
    return state


class TestFillClipping:
    def test_over_leveraged_buy_is_clipped_to_hand_computed_size(self) -> None:
        # cash=1000, flat book, cap=2.0, no fee, no slippage, price 100.
        # A 100-unit buy is 10_000 notional against 1000 equity → leverage 10.0.
        # Clip solves (0 + a) * 100 == 2.0 * (1000 - a * 100 * 0.0) → a = 2000 / 100 = 20.
        result = step_core(
            _state(cash=1000.0),
            Action(side="buy", units=100.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert result.fills == 1
        assert float(result.state.portfolio[1]) == 20.0
        assert float(result.state.portfolio[0]) == -1000.0
        # Equity 1000, exposure 2000 → exactly at the cap, not over it.
        assert leverage_ratio(20.0, 100.0, 1000.0) == pytest.approx(2.0, rel=1e-12)
        assert result.state.done is False

    def test_clip_accounts_for_the_fee_charged_on_the_clipped_notional(self) -> None:
        # Same setup with fee_bps=10 (0.001). The fee is charged on the clipped
        # notional and shrinks equity, so the cap solves against a smaller book:
        # a * 100 == 2.0 * (1000 - a * 100 * 0.001) → a = 2000 / 100.2.
        result = step_core(
            _state(cash=1000.0),
            Action(side="buy", units=100.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=2.0, fee_bps=10.0),
            _FULL_FILL,
        )
        units = float(result.state.portfolio[1])
        cash = float(result.state.portfolio[0])
        assert units == pytest.approx(19.96007984031936, rel=1e-12)
        assert cash == pytest.approx(-998.0039920159678, rel=1e-12)
        assert leverage_ratio(units, 100.0, cash + units * 100.0) == pytest.approx(2.0, rel=1e-12)

    def test_clip_uses_the_slipped_execution_price(self) -> None:
        # slippage_bps=100 → a buy executes at 101, not the 100 fill price. The cap
        # is solved at the execution price: a = 2.0 * 1000 / 101.
        result = step_core(
            _state(cash=1000.0),
            Action(side="buy", units=100.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=2.0, slippage_bps=100.0),
            _FULL_FILL,
        )
        units = float(result.state.portfolio[1])
        cash = float(result.state.portfolio[0])
        assert units == pytest.approx(19.801980198019802, rel=1e-12)
        assert leverage_ratio(units, 101.0, cash + units * 101.0) == pytest.approx(2.0, rel=1e-12)

    def test_order_exactly_at_the_cap_fills_the_full_requested_size(self) -> None:
        # 20 units at 100 is exactly 2.0x on 1000 equity — the boundary is inclusive.
        result = step_core(
            _state(cash=1000.0),
            Action(side="buy", units=20.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert float(result.state.portfolio[1]) == 20.0
        assert float(result.state.portfolio[0]) == -1000.0

    def test_order_just_under_the_cap_fills_the_full_requested_size(self) -> None:
        result = step_core(
            _state(cash=1000.0),
            Action(side="buy", units=19.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert float(result.state.portfolio[1]) == 19.0

    def test_reducing_order_is_never_clipped_even_when_already_over_the_cap(self) -> None:
        # Book is 20 units at 100 on 1000 equity → leverage 2.0, against a 1.0 cap.
        # Selling 1 unit still leaves the account over the cap, but shrinking exposure
        # must never be blocked — a real margin system always lets you de-risk.
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="sell", units=1.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=1.0),
            _FULL_FILL,
        )
        assert result.fills == 1
        assert float(result.state.portfolio[1]) == 19.0
        assert float(result.state.portfolio[0]) == -900.0
        assert leverage_ratio(19.0, 100.0, 1000.0) == pytest.approx(1.9, rel=1e-12)

    def test_closing_the_whole_position_is_never_clipped(self) -> None:
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="sell", units=20.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=1.0),
            _FULL_FILL,
        )
        assert float(result.state.portfolio[1]) == 0.0
        assert float(result.state.portfolio[0]) == 1000.0

    def test_adding_to_a_position_already_over_the_cap_fills_zero_units(self) -> None:
        # allowed = (1.0 * 1000 - 20 * 100) / 100 = -10 → clipped to zero. The order
        # still "fills" (for nothing) rather than raising; the book is untouched.
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="buy", units=5.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=1.0),
            _FULL_FILL,
        )
        assert result.fills == 1
        assert float(result.state.portfolio[1]) == 20.0
        assert float(result.state.portfolio[0]) == -1000.0
        assert int(result.state.order_active[0]) == 0

    def test_short_side_is_clipped_symmetrically(self) -> None:
        # Flat book, cap 2.0, price 100 → a short is capped at 2000 notional too.
        result = step_core(
            _state(cash=1000.0),
            Action(side="sell", units=100.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert float(result.state.portfolio[1]) == -20.0
        assert float(result.state.portfolio[0]) == 3000.0

    def test_flipping_a_short_into_a_long_is_clipped_at_the_far_side_cap(self) -> None:
        # Short 5 units at 100 on 1500 cash → equity 1000. Buying 100 units flips the
        # book long: allowed = (2.0 * 1000 + 5 * 100) / 100 = 25 → ends at +20 units.
        result = step_core(
            _state(cash=1500.0, units=-5.0),
            Action(side="buy", units=100.0, order_type="market"),
            _market([100.0] * 4),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert float(result.state.portfolio[1]) == 20.0
        assert float(result.state.portfolio[0]) == -1000.0


class TestClipHelper:
    def test_zero_size_request_is_returned_unchanged(self) -> None:
        # Falls out of the de-risking gate: exposure is unchanged, so nothing to clip.
        assert clip_units_for_leverage(1000.0, 0.0, 0.0, 100.0, 0.0, 2.0) == 0.0

    def test_non_positive_execution_price_skips_the_check(self) -> None:
        # Guarded so the pure-python path cannot raise ZeroDivisionError where the
        # numba mirror would silently produce inf.
        assert clip_units_for_leverage(1000.0, 0.0, 5.0, 0.0, 0.0, 2.0) == 5.0

    def test_zero_cap_forbids_opening_any_exposure(self) -> None:
        assert clip_units_for_leverage(1000.0, 0.0, 5.0, 100.0, 0.0, 0.0) == 0.0

    def test_zero_cap_still_allows_closing(self) -> None:
        assert clip_units_for_leverage(-1000.0, 20.0, -20.0, 100.0, 0.0, 0.0) == -20.0

    def test_clip_never_upsizes_a_small_request(self) -> None:
        assert clip_units_for_leverage(1000.0, 0.0, 1.0, 100.0, 0.0, 2.0) == 1.0


class TestMaintenanceMarginFormula:
    """`notional * rate - amount`, floored at zero — one bracket of an exchange table."""

    def test_requirement_is_the_configured_fraction_of_notional(self) -> None:
        assert maintenance_margin(1002.5, 0.005, 0.0) == pytest.approx(5.0125, rel=1e-12)

    def test_maintenance_amount_is_deducted_from_the_requirement(self) -> None:
        # The bracket deduction that keeps a multi-tier table continuous; the shipped
        # single tier has none, but the field is honored so a table can be added later.
        assert maintenance_margin(1000.0, 0.01, 4.0) == pytest.approx(6.0, rel=1e-12)

    def test_requirement_is_floored_at_zero(self) -> None:
        # A deduction larger than the raw requirement cannot make the requirement negative.
        assert maintenance_margin(1000.0, 0.005, 500.0) == 0.0

    def test_flat_book_requires_nothing(self) -> None:
        assert maintenance_margin(0.0, 0.005, 0.0) == 0.0

    def test_short_notional_is_treated_by_gross_size(self) -> None:
        assert maintenance_margin(-1000.0, 0.005, 0.0) == maintenance_margin(1000.0, 0.005, 0.0)

    def test_equity_above_the_requirement_is_not_below_it(self) -> None:
        assert is_below_maintenance_margin(5.0125, 1002.5, 0.005, 0.0) is False

    def test_equity_exactly_on_the_requirement_is_not_below_it(self) -> None:
        # Boundary is inclusive on the surviving side, matching the initial-margin cap.
        assert is_below_maintenance_margin(5.0, 1000.0, 0.005, 0.0) is False

    def test_equity_under_the_requirement_is_below_it(self) -> None:
        assert is_below_maintenance_margin(2.5, 1002.5, 0.005, 0.0) is True

    def test_flat_book_is_never_below_maintenance_even_with_negative_equity(self) -> None:
        # No position to maintain. A negative flat balance is the insolvency
        # backstop's case, not a margin call.
        assert is_below_maintenance_margin(-500.0, 0.0, 0.005, 0.0) is False

    def test_shipped_defaults_match_the_documented_single_tier(self) -> None:
        # The default rate/amount are the lowest bracket of the documented Binance
        # approximation — one tier, floored at zero notional.
        assert len(MAINTENANCE_MARGIN_TIERS) == 1
        tier = MAINTENANCE_MARGIN_TIERS[0]
        assert tier.notional_floor == 0.0
        cfg = GameConfig()
        assert cfg.maintenance_margin_rate == tier.rate == 0.005
        assert cfg.maintenance_amount == tier.amount == 0.0

    def test_liquidate_position_realizes_the_book_into_cash(self) -> None:
        portfolio = np_asarray([-1000.0, 20.0], dtype=np_float64)
        liquidate_position(portfolio, 50.125)
        assert portfolio.tolist() == [2.5, 0.0]


class TestMaintenanceMarginLiquidation:
    """The maintenance check fires earlier than `equity <= 0`, on an open position."""

    def test_position_comfortably_above_maintenance_survives_a_drawdown(self) -> None:
        # 15 units at 100 against 1000 equity (1.5x, inside the 2.0x cap). The mark
        # drops 10% to 90: equity 850 against a 6.75 requirement — untouched.
        result = step_core(
            _state(cash=-500.0, units=15.0),
            Action(side="hold"),
            _market([100.0, 90.0, 90.0, 90.0]),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert result.state.done is False
        assert float(result.state.portfolio[1]) == 15.0
        assert float(result.state.portfolio[0]) == -500.0
        assert maintenance_margin(1350.0, 0.005, 0.0) == pytest.approx(6.75, rel=1e-12)

    def test_leverage_drifting_far_above_max_leverage_is_not_liquidated(self) -> None:
        # THE point of the two-tier model. 20 units bought at the 2.0x cap, then the
        # mark falls to 60: equity 200, exposure 1200 → 6.0x, three times the ORDER-time
        # cap. A real exchange does not continuously re-clip an open position, and
        # neither does this engine: 200 is still far above the 6.0 maintenance
        # requirement, so the book is untouched and the episode continues.
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="hold"),
            _market([100.0, 60.0, 60.0, 60.0]),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert result.state.done is False
        assert float(result.state.portfolio[1]) == 20.0
        assert float(result.state.portfolio[0]) == -1000.0
        assert leverage_ratio(20.0, 60.0, 200.0) == pytest.approx(6.0, rel=1e-12)
        assert is_below_maintenance_margin(200.0, 1200.0, 0.005, 0.0) is False

    def test_crossing_below_maintenance_liquidates_with_positive_residual_equity(self) -> None:
        # 20 units carried into a mark of 50.125: equity = -1000 + 1002.5 = 2.50, still
        # POSITIVE, so the `equity <= 0` backstop would not have fired. The maintenance
        # requirement is 1002.5 * 0.005 = 5.0125, so the account is closed out here,
        # keeping $2.50 — exactly the earlier intervention a real exchange makes.
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="hold"),
            _market([100.0, 50.125, 50.125, 50.125]),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert result.state.done is True
        assert float(result.state.portfolio[1]) == 0.0
        assert float(result.state.portfolio[0]) == 2.5
        assert float(result.state.portfolio[0]) > 0.0
        assert maintenance_margin(1002.5, 0.005, 0.0) == pytest.approx(5.0125, rel=1e-12)

    def test_liquidation_can_trigger_on_the_filling_bar(self) -> None:
        # The check runs at both points, not only at the next bar's mark. A limit buy
        # fills 10 units at 300 (exactly the 3.0x cap on 1000 cash) while this bar's
        # close is 201: equity = -2000 + 2010 = 10.00 against a 10.05 requirement. The
        # next bar closes at 700, which would put the book far back above water, so only
        # the fill-bar check can catch this.
        result = step_core(
            _state(cash=1000.0),
            Action(side="buy", units=10.0, order_type="limit", limit_price=300.0),
            _market([201.0, 700.0, 700.0, 700.0]),
            _cfg(max_leverage=3.0),
            _FULL_FILL,
        )
        assert result.fills == 1
        assert result.state.done is True
        assert float(result.state.portfolio[1]) == 0.0
        assert float(result.state.portfolio[0]) == 10.0
        assert maintenance_margin(2010.0, 0.005, 0.0) == pytest.approx(10.05, rel=1e-12)

    def test_a_higher_maintenance_rate_liquidates_earlier_and_leaves_more_equity(self) -> None:
        # The rate is configurable, and the whole point of it: at a 10% maintenance rate
        # the same book is closed out at a mark of 55 with $100 left — where the default
        # 0.5% rate would have carried it down to ~50.25, and the bare insolvency
        # backstop all the way to 50.
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="hold"),
            _market([100.0, 55.0, 55.0, 55.0]),
            _cfg(max_leverage=2.0, maintenance_margin_rate=0.10),
            _FULL_FILL,
        )
        assert result.state.done is True
        assert float(result.state.portfolio[1]) == 0.0
        assert float(result.state.portfolio[0]) == 100.0

    def test_maintenance_amount_deduction_keeps_a_position_alive(self) -> None:
        # Same 55.0 mark and 10% rate, but a 50 bracket deduction drops the requirement
        # to 60, under the 100 equity — so the identical book survives. Pins that
        # `maintenance_amount` is really threaded through to the engine.
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="hold"),
            _market([100.0, 55.0, 55.0, 55.0]),
            _cfg(max_leverage=2.0, maintenance_margin_rate=0.10, maintenance_amount=50.0),
            _FULL_FILL,
        )
        assert result.state.done is False
        assert float(result.state.portfolio[1]) == 20.0

    def test_single_bar_gap_past_maintenance_lands_below_zero_equity(self) -> None:
        # Deep-tail gap risk: the mark falls 100 → 40 in one bar, jumping clean past the
        # 50.25 maintenance threshold to negative equity. Still liquidated at that price
        # and terminal — realistic for a volatile bar, not a bug.
        result = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="hold"),
            _market([100.0, 40.0, 40.0, 40.0]),
            _cfg(max_leverage=2.0),
            _FULL_FILL,
        )
        assert result.state.done is True
        assert float(result.state.portfolio[1]) == 0.0
        assert float(result.state.portfolio[0]) == -200.0

    def test_insolvency_backstop_still_terminates_with_the_maintenance_tier_disabled(self) -> None:
        # With rate and amount both zero the maintenance requirement is zero for any
        # notional, so `equity <= 0` is the only trigger left — proving the backstop is
        # load-bearing rather than shadowed by the new check.
        cfg = _cfg(max_leverage=2.0, maintenance_margin_rate=0.0, maintenance_amount=0.0)
        survives = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="hold"),
            _market([100.0, 50.125, 50.125, 50.125]),
            cfg,
            _FULL_FILL,
        )
        assert survives.state.done is False
        assert float(survives.state.portfolio[1]) == 20.0

        wiped = step_core(
            _state(cash=-1000.0, units=20.0),
            Action(side="hold"),
            _market([100.0, 40.0, 40.0, 40.0]),
            cfg,
            _FULL_FILL,
        )
        assert wiped.state.done is True
        assert float(wiped.state.portfolio[0]) == -200.0

    def test_settle_helper_leaves_a_healthy_book_untouched(self) -> None:
        portfolio = np_asarray([-1000.0, 20.0], dtype=np_float64)
        assert settle_maintenance_margin(portfolio, 60.0, 0.005, 0.0) is False
        assert portfolio.tolist() == [-1000.0, 20.0]

    def test_settle_helper_closes_a_book_under_its_requirement(self) -> None:
        portfolio = np_asarray([-1000.0, 20.0], dtype=np_float64)
        assert settle_maintenance_margin(portfolio, 50.125, 0.005, 0.0) is True
        assert portfolio.tolist() == [2.5, 0.0]

    def test_env_reports_the_liquidated_book_through_a_full_step(
        self,
        market_data_factory: Callable[..., MarketData],
    ) -> None:
        # End-to-end through MarketEnv: buy at the 3.0x cap on 1000 cash (30 units at
        # 100, cash -2000), then a mark of 66.75 leaves equity 2.50 against a 10.0125
        # requirement — liquidated with positive equity, and reported as flat.
        cfg = _cfg(max_leverage=3.0, initial_cash=1000.0)
        env = MarketEnv(data=market_data_factory(close=[100.0, 100.0, 66.75, 66.75, 66.75]), cfg=cfg)
        env.reset(seed=1)

        opened = env.step(Action(side="buy", units=100.0, order_type="market"))
        assert opened.state.units == 30.0
        assert opened.done is False

        closed = env.step(Action(side="hold"))
        assert closed.done is True
        assert closed.state.units == 0.0
        assert closed.state.cash == pytest.approx(2.5)
        assert closed.state.equity == pytest.approx(2.5)
        assert closed.state.equity > 0.0
        assert closed.state.leverage == 0.0


class TestInsolvencyTermination:
    def test_mark_to_market_wipeout_liquidates_and_terminates(
        self,
        market_data_factory: Callable[..., MarketData],
    ) -> None:
        # Buy at the 3.0x cap (30 units at 100 on 1000 equity), then the price halves.
        # Book would be cash -2000 + 30 * 50 = -500 equity while still holding units.
        cfg = _cfg(max_leverage=3.0, initial_cash=1000.0)
        env = MarketEnv(data=market_data_factory(close=[100.0, 100.0, 50.0, 50.0, 50.0]), cfg=cfg)
        env.reset(seed=1)

        opened = env.step(Action(side="buy", units=100.0, order_type="market"))
        assert opened.state.units == 30.0
        assert opened.state.cash == -2000.0
        assert opened.done is False

        wiped = env.step(Action(side="hold"))
        assert wiped.done is True
        assert wiped.state.units == 0.0
        assert wiped.state.cash == pytest.approx(-500.0)
        # Equity lands exactly on the liquidated cash value and stops there.
        assert wiped.state.equity == pytest.approx(wiped.state.cash)
        assert math.isfinite(wiped.state.leverage)
        assert wiped.state.leverage == 0.0

    def test_terminal_state_does_not_drift_on_further_steps(
        self,
        market_data_factory: Callable[..., MarketData],
    ) -> None:
        cfg = _cfg(max_leverage=3.0, initial_cash=1000.0)
        env = MarketEnv(data=market_data_factory(close=[100.0, 100.0, 50.0, 10.0, 1.0]), cfg=cfg)
        env.reset(seed=1)
        env.step(Action(side="buy", units=100.0, order_type="market"))
        wiped = env.step(Action(side="hold"))
        assert wiped.done is True

        # Prices keep collapsing; a stale unit count would keep draining equity.
        for _ in range(3):
            later = env.step(Action(side="hold"))
            assert later.done is True
            assert later.state.equity == pytest.approx(wiped.state.equity)

    def test_fill_driven_wipeout_liquidates_on_the_filling_bar(self) -> None:
        # A limit buy that fills far above the bar's close: 3 units (the 3.0x cap at
        # the 1000 execution price) are worth only 300 marked at close 100, so equity
        # is -1700 on the filling bar. The next bar's close of 700 would put the book
        # back above water, so only the same-bar solvency check can catch this.
        result = step_core(
            _state(cash=1000.0),
            Action(side="buy", units=10.0, order_type="limit", limit_price=1000.0),
            _market([100.0, 700.0, 700.0, 700.0]),
            _cfg(max_leverage=3.0),
            _FULL_FILL,
        )
        assert result.fills == 1
        assert result.state.done is True
        assert float(result.state.portfolio[1]) == 0.0
        assert float(result.state.portfolio[0]) == pytest.approx(-1700.0)

    def test_solvent_book_is_left_untouched(self) -> None:
        portfolio = np_asarray([1000.0, 5.0], dtype=np_float64)
        assert settle_insolvency(portfolio, 100.0) is False
        assert portfolio.tolist() == [1000.0, 5.0]

    def test_flat_book_with_negative_cash_is_insolvent_but_unchanged(self) -> None:
        portfolio = np_asarray([-500.0, 0.0], dtype=np_float64)
        assert settle_insolvency(portfolio, 100.0) is True
        assert portfolio.tolist() == [-500.0, 0.0]

    def test_data_exhaustion_still_terminates(self, market_data_factory: Callable[..., MarketData]) -> None:
        # Insolvency is an additional terminal condition, not a replacement.
        cfg = _cfg(initial_cash=1000.0)
        env = MarketEnv(data=market_data_factory(close=[100.0, 100.0, 100.0]), cfg=cfg)
        env.reset(seed=1)
        env.step(Action(side="hold"))
        final = env.step(Action(side="hold"))
        assert final.done is True
        assert final.state.equity == pytest.approx(1000.0)


class TestNumbaMirrorsAreIdenticalSource:
    """The numba mirrors are hand-copied; pin them so they cannot silently drift."""

    # A mirror that delegates to another helper has to call that helper's `_jit`
    # name, so a delegating body can never dump identically without renaming.
    # Only these exact identifiers are normalized — every other difference,
    # including any call to a helper not listed here, still shows up as a diff.
    _JIT_ALIASES = {
        "_maintenance_margin_jit": "maintenance_margin",
        "_is_below_maintenance_margin_jit": "is_below_maintenance_margin",
        "_liquidate_position_jit": "liquidate_position",
    }

    @classmethod
    def _body_ast(cls, func: Any) -> str:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in cls._JIT_ALIASES:
                node.id = cls._JIT_ALIASES[node.id]
        body = tree.body[0].body  # type: ignore[attr-defined]
        first = body[0]
        is_docstring = isinstance(first, ast.Expr) and isinstance(getattr(first.value, "value", None), str)
        return "\n".join(ast.dump(node) for node in (body[1:] if is_docstring else body))

    def test_clip_helper_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(clip_units_for_leverage) == self._body_ast(_clip_units_for_leverage_jit.py_func)

    def test_settle_helper_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(settle_insolvency) == self._body_ast(_settle_insolvency_jit.py_func)

    def test_maintenance_margin_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(maintenance_margin) == self._body_ast(_maintenance_margin_jit.py_func)

    def test_maintenance_predicate_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(is_below_maintenance_margin) == self._body_ast(_is_below_maintenance_margin_jit.py_func)

    def test_liquidate_helper_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(liquidate_position) == self._body_ast(_liquidate_position_jit.py_func)

    def test_maintenance_settle_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(settle_maintenance_margin) == self._body_ast(_settle_maintenance_margin_jit.py_func)

    def test_alias_map_only_renames_the_mirrors_it_claims_to(self) -> None:
        # Guard on the guard: a typo'd alias would silently stop normalizing and the
        # comparison above would fail loudly, but an alias for a name that no longer
        # exists would rot unnoticed. Every key must be a real numba mirror.
        import stock_simulator.core as core_module

        for jit_name, pure_name in self._JIT_ALIASES.items():
            assert hasattr(core_module, jit_name)
            assert hasattr(core_module, pure_name)


class TestNumbaParity:
    @staticmethod
    def _assert_identical(py_out: CoreStepOutput, nb_out: CoreStepOutput) -> None:
        # Bit-identical, not approximate: the two engines must never diverge.
        assert py_out.state.t == nb_out.state.t
        assert py_out.state.done == nb_out.state.done
        assert py_out.fills == nb_out.fills
        assert py_out.state.portfolio.tolist() == nb_out.state.portfolio.tolist()
        assert py_out.equity_delta == nb_out.equity_delta
        # N9: the fill-slot/exec-price observability fields must also match
        # bit-for-bit — the numba mirror decodes its -1/NaN sentinels back to the
        # same Optional values the pure-Python match loop produces directly.
        assert py_out.filled_order_slot == nb_out.filled_order_slot
        assert py_out.exec_price == nb_out.exec_price

    def test_clipped_fill_is_identical_in_both_engines(self) -> None:
        cfg = _cfg(max_leverage=2.0, fee_bps=10.0, slippage_bps=5.0)
        market = _market([100.0] * 4)
        action = Action(side="buy", units=100.0, order_type="market")
        self._assert_identical(
            step_core(_state(cash=1000.0), action, market, cfg, _FULL_FILL),
            step_core_numba(_state(cash=1000.0), action, market, cfg, _FULL_FILL),
        )

    def test_zero_size_clip_is_identical_in_both_engines(self) -> None:
        cfg = _cfg(max_leverage=1.0)
        market = _market([100.0] * 4)
        action = Action(side="buy", units=5.0, order_type="market")
        self._assert_identical(
            step_core(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL),
            step_core_numba(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL),
        )

    def test_reducing_fill_is_identical_in_both_engines(self) -> None:
        cfg = _cfg(max_leverage=1.0, fee_bps=4.0, slippage_bps=1.0)
        market = _market([100.0] * 4)
        action = Action(side="sell", units=1.0, order_type="market")
        self._assert_identical(
            step_core(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL),
            step_core_numba(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL),
        )

    def test_fill_driven_liquidation_is_identical_in_both_engines(self) -> None:
        cfg = _cfg(max_leverage=3.0)
        market = _market([100.0, 700.0, 700.0, 700.0])
        action = Action(side="buy", units=10.0, order_type="limit", limit_price=1000.0)
        self._assert_identical(
            step_core(_state(cash=1000.0), action, market, cfg, _FULL_FILL),
            step_core_numba(_state(cash=1000.0), action, market, cfg, _FULL_FILL),
        )

    def test_delayed_limit_fill_slot_and_price_are_identical_in_both_engines(self) -> None:
        # N9: a limit order that does NOT fill on the bar it's submitted (low stays
        # above the buy limit for several bars) and only fills once the price drops
        # through it several steps later. Both engines must report the identical
        # fill slot and execution price on the filling step, not just identical
        # portfolio/equity_delta.
        closes = [100.0, 100.0, 100.0, 100.0, 80.0, 80.0]
        market = _market(closes, spread=0.5)
        cfg = _cfg(max_leverage=10.0)
        submit = Action(side="buy", units=10.0, order_type="limit", limit_price=85.0)
        hold = Action(side="hold")

        def _run(step_fn: Callable[..., CoreStepOutput]) -> list[CoreStepOutput]:
            # bar t=0: submit (no fill, low stays above the limit). bars t=1..3:
            # hold (still no fill). bar t=4: hold submitted, but the OLD order
            # fills once close/low drops through the limit price.
            outputs = [step_fn(_state(cash=1000.0), submit, market, cfg, _FULL_FILL)]
            for _ in range(4):
                outputs.append(step_fn(outputs[-1].state, hold, market, cfg, _FULL_FILL))
            return outputs

        py_outputs = _run(step_core)
        nb_outputs = _run(step_core_numba)

        for i in range(4):
            self._assert_identical(py_outputs[i], nb_outputs[i])
            assert py_outputs[i].fills == 0
            assert py_outputs[i].filled_order_slot is None
            assert py_outputs[i].exec_price is None

        self._assert_identical(py_outputs[4], nb_outputs[4])
        assert py_outputs[4].fills == 1
        assert py_outputs[4].filled_order_slot == 0
        assert py_outputs[4].exec_price == pytest.approx(85.0)

    def test_maintenance_margin_liquidation_is_identical_in_both_engines(self) -> None:
        # A liquidation driven by the maintenance check, NOT by `equity <= 0`: equity
        # lands on +2.50 against a 5.0125 requirement. Both engines must produce the
        # identical liquidated book, bit for bit.
        cfg = _cfg(max_leverage=2.0)
        market = _market([100.0, 50.125, 50.125, 50.125])
        action = Action(side="hold")
        py_out = step_core(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL)
        nb_out = step_core_numba(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL)
        self._assert_identical(py_out, nb_out)
        assert py_out.state.done is True
        assert py_out.state.portfolio.tolist() == [2.5, 0.0]

    def test_fill_bar_maintenance_liquidation_is_identical_in_both_engines(self) -> None:
        # The other check point: the maintenance call fires on the filling bar.
        cfg = _cfg(max_leverage=3.0)
        market = _market([201.0, 700.0, 700.0, 700.0])
        action = Action(side="buy", units=10.0, order_type="limit", limit_price=300.0)
        py_out = step_core(_state(cash=1000.0), action, market, cfg, _FULL_FILL)
        nb_out = step_core_numba(_state(cash=1000.0), action, market, cfg, _FULL_FILL)
        self._assert_identical(py_out, nb_out)
        assert py_out.state.done is True
        assert py_out.state.portfolio.tolist() == [10.0, 0.0]

    def test_custom_maintenance_parameters_are_identical_in_both_engines(self) -> None:
        # Both new config fields reach `step_core_jit`; a mis-threaded parameter would
        # leave the numba engine on its defaults and diverge here.
        cfg = _cfg(max_leverage=2.0, maintenance_margin_rate=0.10, maintenance_amount=50.0)
        market = _market([100.0, 55.0, 55.0, 55.0])
        action = Action(side="hold")
        py_out = step_core(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL)
        nb_out = step_core_numba(_state(cash=-1000.0, units=20.0), action, market, cfg, _FULL_FILL)
        self._assert_identical(py_out, nb_out)
        assert py_out.state.done is False

    def test_maintenance_liquidation_through_the_env_is_identical_in_both_engines(
        self,
        market_data_factory: Callable[..., MarketData],
    ) -> None:
        # Same scenario driven through MarketEnv.step (which reaches the numba engine
        # via `step_core_numba`). The other call site, `MarketEnv._advance_one_step`,
        # is covered by test_use_numba_parity.py.
        closes = [100.0, 100.0, 66.75, 66.75, 66.75]
        actions = [Action(side="buy", units=100.0, order_type="market"), Action(side="hold")]
        trajectories = []
        for use_numba in (False, True):
            cfg = _cfg(max_leverage=3.0, initial_cash=1000.0, use_numba=use_numba)
            env = MarketEnv(data=market_data_factory(close=closes), cfg=cfg)
            env.reset(seed=5)
            trajectories.append([env.step(action).state for action in actions])
        for py_state, nb_state in zip(*trajectories, strict=True):
            assert py_state == nb_state
        assert trajectories[0][-1].done is True
        assert trajectories[0][-1].cash == pytest.approx(2.5)

    def test_mark_to_market_liquidation_is_identical_in_both_engines(
        self,
        market_data_factory: Callable[..., MarketData],
    ) -> None:
        closes = [100.0, 100.0, 50.0, 50.0, 50.0]
        actions = [Action(side="buy", units=100.0, order_type="market"), Action(side="hold")]
        trajectories = []
        for use_numba in (False, True):
            cfg = _cfg(max_leverage=3.0, initial_cash=1000.0, use_numba=use_numba)
            env = MarketEnv(data=market_data_factory(close=closes), cfg=cfg)
            env.reset(seed=5)
            trajectories.append([env.step(action).state for action in actions])
        for py_state, nb_state in zip(*trajectories, strict=True):
            assert py_state == nb_state


class TestLeverageIsJsonSafe:
    @staticmethod
    def _serialized_portfolio_vector(portfolio_vector: list[float]) -> list[float | None]:
        model = ObservationModel(
            market_window_handle=MarketWindowViewHandleModel(start=0, end=1, t=0, current_price=100.0),
            portfolio_vector=portfolio_vector,
            order_summary_vector=[0.0, 0.0, 0.0],
            done=True,
        )
        payload = json.loads(model.model_dump_json())
        return list(payload["portfolio_vector"])

    def test_insolvent_snapshot_serializes_leverage_as_a_finite_number(self) -> None:
        # Regression: leverage was `inf` here, which pydantic/FastAPI emit as JSON
        # `null`, and clients crashed with a TypeError on the missing number.
        state = _state(cash=100.0, units=-10.0)
        snapshot = snapshot_from_state(state, price=100.0)
        assert snapshot.equity < 0.0
        assert snapshot.leverage == INSOLVENT_LEVERAGE

        leverage = self._serialized_portfolio_vector(list(snapshot.to_vector()))[3]
        assert leverage is not None
        assert math.isfinite(leverage)

    def test_liquidated_episode_serializes_without_nulls(
        self,
        market_data_factory: Callable[..., MarketData],
    ) -> None:
        cfg = _cfg(max_leverage=3.0, initial_cash=1000.0)
        env = MarketEnv(data=market_data_factory(close=[100.0, 100.0, 50.0, 50.0, 50.0]), cfg=cfg)
        env.reset(seed=1)
        env.step(Action(side="buy", units=100.0, order_type="market"))
        wiped = env.step(Action(side="hold"))
        assert wiped.done is True

        state_payload = json.loads(EnvStateModel.from_dataclass(wiped.state).model_dump_json())
        assert state_payload["leverage"] is not None
        assert math.isfinite(state_payload["leverage"])

        vector = self._serialized_portfolio_vector(list(wiped.observation.portfolio_vector))
        assert all(value is not None for value in vector)
