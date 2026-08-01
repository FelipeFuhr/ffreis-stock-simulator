"""Margin enforcement: max_leverage fill clipping and insolvency liquidation.

`GameConfig.max_leverage` used to be parsed and never read, so an agent could
accumulate unbounded leverage, and `done` was driven purely by end-of-data, so an
episode kept executing steps against negative equity. These tests pin both halves of
the fix and the python/numba parity of the two code paths that implement it.
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

from stock_simulator.config import GameConfig
from stock_simulator.core import (
    CoreState,
    CoreStepOutput,
    MarketArrays,
    _clip_units_for_leverage_jit,
    _settle_insolvency_jit,
    clip_units_for_leverage,
    initial_core_state,
    leverage_ratio,
    settle_insolvency,
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

    @staticmethod
    def _body_ast(func: Any) -> str:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        body = tree.body[0].body  # type: ignore[attr-defined]
        first = body[0]
        is_docstring = isinstance(first, ast.Expr) and isinstance(getattr(first.value, "value", None), str)
        return "\n".join(ast.dump(node) for node in (body[1:] if is_docstring else body))

    def test_clip_helper_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(clip_units_for_leverage) == self._body_ast(_clip_units_for_leverage_jit.py_func)

    def test_settle_helper_body_matches_its_numba_mirror(self) -> None:
        assert self._body_ast(settle_insolvency) == self._body_ast(_settle_insolvency_jit.py_func)


class TestNumbaParity:
    @staticmethod
    def _assert_identical(py_out: CoreStepOutput, nb_out: CoreStepOutput) -> None:
        # Bit-identical, not approximate: the two engines must never diverge.
        assert py_out.state.t == nb_out.state.t
        assert py_out.state.done == nb_out.state.done
        assert py_out.fills == nb_out.fills
        assert py_out.state.portfolio.tolist() == nb_out.state.portfolio.tolist()
        assert py_out.equity_delta == nb_out.equity_delta

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
