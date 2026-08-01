from __future__ import annotations

from dataclasses import dataclass
from math import inf as math_inf

from numba import njit
from numpy import array as np_array
from numpy import float32 as np_float32
from numpy import float64 as np_float64
from numpy import int8 as np_int8
from numpy import int32 as np_int32
from numpy import zeros as np_zeros
from numpy.typing import NDArray

from .config import GameConfig
from .types import Action

_BUY_SIDE = np_int8(1)
_SELL_SIDE = np_int8(-1)
_ORDER_MARKET = np_int8(0)
_ORDER_LIMIT = np_int8(1)


@dataclass(frozen=True)
class CoreState:
    t: int
    done: bool
    portfolio: NDArray[np_float64]
    order_active: NDArray[np_int8]
    order_side: NDArray[np_int8]
    order_type: NDArray[np_int8]
    order_units: NDArray[np_float64]
    order_limit_price: NDArray[np_float64]
    order_eligible_t: NDArray[np_int32]
    order_ttl: NDArray[np_int32]


@dataclass(frozen=True)
class MarketArrays:
    open: NDArray[np_float32]
    high: NDArray[np_float32]
    low: NDArray[np_float32]
    close: NDArray[np_float32]
    n: int


@dataclass(frozen=True)
class CoreStepOutput:
    state: CoreState
    fills: int
    equity_delta: float


def leverage_ratio(units: float, price: float, equity: float) -> float:
    """Account leverage: gross exposure over equity.

    Single definition of leverage for the whole engine — the reporting path
    (``portfolio.snapshot_from_state``) calls this directly, and the fill-time
    margin check below is an algebraic rearrangement of the same expression.

    A book with no exposure has ``0.0`` leverage regardless of the sign of equity.
    Exposure carried against non-positive equity has undefined (unbounded) leverage
    and reports ``inf``; callers that serialize the value are responsible for mapping
    it onto a finite, JSON-safe convention.
    """
    exposure = abs(units * price)
    if equity > 0.0:
        return exposure / equity
    return math_inf if exposure > 0.0 else 0.0


def clip_units_for_leverage(
    cash: float,
    units: float,
    signed_units: float,
    exec_price: float,
    fee_bps: float,
    max_leverage: float,
) -> float:
    """Clip a fill down to the largest size that keeps leverage within ``max_leverage``.

    Acts as one more upper bound on fill size, alongside the ``partial_fill_min`` /
    ``partial_fill_max`` draw — never a separate all-or-nothing rejection path. The
    order still fills, just for fewer units; a request that cannot add any exposure
    at all fills for zero units (a real broker declines the extra margin, it does not
    error).

    Two cases are always filled at the requested size:

    * **De-risking** — the fill shrinks gross exposure (``abs(units_after) <=
      abs(units)``). Closing or reducing a position is allowed even when the account
      is already above the cap, exactly like a real margin system.
    * **Within the cap** — leverage after the fill is at or under ``max_leverage``.
      The boundary is inclusive, so an order landing exactly on the cap fills whole.

    Otherwise the fill is clipped to ``a`` units, the solution of

    ``(direction * units + a) * exec_price == max_leverage * (equity - a * exec_price * fee_rate)``

    which is :func:`leverage_ratio` at the post-fill book, set equal to the cap and
    solved for size (equity is marked at ``exec_price``; the fee shrinks it, hence the
    ``1 + cap * fee_rate`` denominator). The comparisons are written multiplied-through
    rather than as divisions so this and its numba mirror agree bit-for-bit.
    """
    if exec_price <= 0.0:
        return signed_units

    fee_rate = fee_bps / 10_000.0
    equity_before = cash + units * exec_price
    equity_after = equity_before - abs(signed_units) * exec_price * fee_rate
    units_after = units + signed_units

    if abs(units_after) <= abs(units):
        return signed_units
    if equity_after > 0.0 and abs(units_after) * exec_price <= max_leverage * equity_after:
        return signed_units

    cap = max_leverage if max_leverage > 0.0 else 0.0
    direction = 1.0 if signed_units > 0.0 else -1.0
    allowed = (cap * equity_before - direction * units * exec_price) / (exec_price * (1.0 + cap * fee_rate))
    if allowed <= 0.0:
        return 0.0
    # min() keeps this a clip and never an upsize: direction * abs(signed_units)
    # reproduces signed_units exactly.
    return direction * min(allowed, abs(signed_units))


def settle_insolvency(portfolio: NDArray[np_float64], price: float) -> bool:
    """Force-close the book when equity is non-positive at ``price``.

    Realizes the whole position into cash at ``price`` so equity becomes exactly cash
    and cannot drift further negative from a stale unit count on later steps. Returns
    whether the account was insolvent, which the caller turns into a terminal step.
    """
    equity = portfolio[0] + portfolio[1] * price
    if equity > 0.0:
        return False
    portfolio[0] = equity
    portfolio[1] = 0.0
    return True


def initial_core_state(initial_cash: float, max_orders: int, start_t: int = 0) -> CoreState:
    return CoreState(
        t=start_t,
        done=False,
        portfolio=np_array([initial_cash, 0.0], dtype=np_float64),
        order_active=np_zeros(max_orders, dtype=np_int8),
        order_side=np_zeros(max_orders, dtype=np_int8),
        order_type=np_zeros(max_orders, dtype=np_int8),
        order_units=np_zeros(max_orders, dtype=np_float64),
        order_limit_price=np_zeros(max_orders, dtype=np_float64),
        order_eligible_t=np_zeros(max_orders, dtype=np_int32),
        order_ttl=np_zeros(max_orders, dtype=np_int32),
    )


def step_core(
    state: CoreState,
    action: Action,
    market_arrays: MarketArrays,
    config: GameConfig,
    random_draws: NDArray[np_float64],
) -> CoreStepOutput:
    if state.done:
        return CoreStepOutput(state=state, fills=0, equity_delta=0.0)

    prev_equity = _equity(state.portfolio, float(market_arrays.close[state.t]))
    next_state = _copy_state(state)

    _apply_action(next_state, action, config)
    fills = _match_orders(next_state, market_arrays, config, random_draws)
    # Solvency is checked twice per step, against both ways equity can be wiped out:
    # a fill on this bar, then the mark-to-market move into the next bar.
    insolvent_on_fill = settle_insolvency(next_state.portfolio, float(market_arrays.close[state.t]))

    next_t = state.t + 1
    if next_t >= market_arrays.n:
        next_t = market_arrays.n - 1
    insolvent_on_mark = settle_insolvency(next_state.portfolio, float(market_arrays.close[next_t]))
    done = next_t >= market_arrays.n - 1 or insolvent_on_fill or insolvent_on_mark

    advanced_state = CoreState(
        t=next_t,
        done=done,
        portfolio=next_state.portfolio,
        order_active=next_state.order_active,
        order_side=next_state.order_side,
        order_type=next_state.order_type,
        order_units=next_state.order_units,
        order_limit_price=next_state.order_limit_price,
        order_eligible_t=next_state.order_eligible_t,
        order_ttl=next_state.order_ttl,
    )

    next_equity = _equity(advanced_state.portfolio, float(market_arrays.close[next_t]))
    return CoreStepOutput(
        state=advanced_state,
        fills=fills,
        equity_delta=next_equity - prev_equity,
    )


@njit(cache=True)  # pragma: no cover
def _clip_units_for_leverage_jit(
    cash: float,
    units: float,
    signed_units: float,
    exec_price: float,
    fee_bps: float,
    max_leverage: float,
) -> float:
    # Numba mirror of clip_units_for_leverage — the two bodies must stay identical
    # statement for statement; tests/unit_tests/test_margin_enforcement.py pins parity.
    if exec_price <= 0.0:
        return signed_units

    fee_rate = fee_bps / 10_000.0
    equity_before = cash + units * exec_price
    equity_after = equity_before - abs(signed_units) * exec_price * fee_rate
    units_after = units + signed_units

    if abs(units_after) <= abs(units):
        return signed_units
    if equity_after > 0.0 and abs(units_after) * exec_price <= max_leverage * equity_after:
        return signed_units

    cap = max_leverage if max_leverage > 0.0 else 0.0
    direction = 1.0 if signed_units > 0.0 else -1.0
    allowed = (cap * equity_before - direction * units * exec_price) / (exec_price * (1.0 + cap * fee_rate))
    if allowed <= 0.0:
        return 0.0
    # min() keeps this a clip and never an upsize: direction * abs(signed_units)
    # reproduces signed_units exactly.
    return direction * min(allowed, abs(signed_units))


@njit(cache=True)  # pragma: no cover
def _settle_insolvency_jit(portfolio: NDArray[np_float64], price: float) -> bool:
    # Numba mirror of settle_insolvency — keep both bodies identical.
    equity = portfolio[0] + portfolio[1] * price
    if equity > 0.0:
        return False
    portfolio[0] = equity
    portfolio[1] = 0.0
    return True


@njit(cache=True)  # pragma: no cover
def step_core_jit(
    t: int,  # NOSONAR — Numba @njit requires flat scalar parameters; dataclass wrappers are unsupported
    done: bool,
    portfolio: NDArray[np_float64],
    order_active: NDArray[np_int8],
    order_side: NDArray[np_int8],
    order_type: NDArray[np_int8],
    order_units: NDArray[np_float64],
    order_limit_price: NDArray[np_float64],
    order_eligible_t: NDArray[np_int32],
    order_ttl: NDArray[np_int32],
    action_side: np_int8,
    action_units: float,
    action_order_type: np_int8,
    action_limit_price: float,
    market_open: NDArray[np_float32],
    market_high: NDArray[np_float32],
    market_low: NDArray[np_float32],
    market_close: NDArray[np_float32],
    market_n: int,
    market_latency_bars: int,
    limit_ttl_bars: int,
    random_draws: NDArray[np_float64],
    fee_bps: float,
    slippage_bps: float,
    max_leverage: float,
) -> tuple[
    int,
    bool,
    NDArray[np_float64],
    NDArray[np_int8],
    NDArray[np_int8],
    NDArray[np_int8],
    NDArray[np_float64],
    NDArray[np_float64],
    NDArray[np_int32],
    NDArray[np_int32],
    int,
    float,
]:
    if done:
        return (
            t,
            True,
            portfolio.copy(),
            order_active.copy(),
            order_side.copy(),
            order_type.copy(),
            order_units.copy(),
            order_limit_price.copy(),
            order_eligible_t.copy(),
            order_ttl.copy(),
            0,
            0.0,
        )

    portfolio_next = portfolio.copy()
    order_active_next = order_active.copy()
    order_side_next = order_side.copy()
    order_type_next = order_type.copy()
    order_units_next = order_units.copy()
    order_limit_price_next = order_limit_price.copy()
    order_eligible_t_next = order_eligible_t.copy()
    order_ttl_next = order_ttl.copy()

    prev_equity = portfolio[0] + portfolio[1] * float(market_close[t])

    if action_side != np_int8(0):
        if action_units <= 0:
            raise ValueError("units must be > 0 for buy/sell actions")
        slot = -1
        free_slot = -1
        for i in range(order_active_next.size):
            if order_active_next[i] == np_int8(1) and order_side_next[i] == action_side:
                slot = i
                break
            if free_slot < 0 and order_active_next[i] == np_int8(0):
                free_slot = i
        if slot < 0:
            slot = free_slot
        if slot < 0:
            raise ValueError("order capacity exceeded")

        order_active_next[slot] = np_int8(1)
        order_side_next[slot] = action_side
        order_type_next[slot] = action_order_type
        order_units_next[slot] = action_units
        order_limit_price_next[slot] = action_limit_price
        order_eligible_t_next[slot] = np_int32(t + market_latency_bars)
        order_ttl_next[slot] = np_int32(limit_ttl_bars)

    open_price = float(market_open[t])
    high = float(market_high[t])
    low = float(market_low[t])
    fills = 0

    for i in range(order_active_next.size):
        if order_active_next[i] == np_int8(0):
            continue
        if t < int(order_eligible_t_next[i]):
            continue

        side = order_side_next[i]
        otype = order_type_next[i]
        limit_price = float(order_limit_price_next[i])
        fill_price = open_price
        filled = False

        if otype == _ORDER_MARKET:
            filled = True
        elif (side == _BUY_SIDE and low <= limit_price) or (side == _SELL_SIDE and high >= limit_price):
            filled = True
            fill_price = limit_price

        if filled:
            fraction = random_draws[i]
            requested_units = float(side) * float(order_units_next[i]) * float(fraction)
            slip = slippage_bps / 10_000.0
            exec_price = fill_price * (1 + slip) if requested_units > 0 else fill_price * (1 - slip)
            signed_units = _clip_units_for_leverage_jit(
                portfolio_next[0],
                portfolio_next[1],
                requested_units,
                exec_price,
                fee_bps,
                max_leverage,
            )
            notional = abs(signed_units) * exec_price
            fee = notional * (fee_bps / 10_000.0)
            portfolio_next[0] -= signed_units * exec_price + fee
            portfolio_next[1] += signed_units
            order_active_next[i] = np_int8(0)
            order_ttl_next[i] = np_int32(0)
            fills += 1
        else:
            ttl = int(order_ttl_next[i]) - 1
            order_ttl_next[i] = np_int32(ttl)
            if ttl <= 0:
                order_active_next[i] = np_int8(0)
                order_ttl_next[i] = np_int32(0)

    # Mirrors step_core: solvency is settled against this bar's fills, then against
    # the mark-to-market move into the next bar.
    insolvent_on_fill = _settle_insolvency_jit(portfolio_next, float(market_close[t]))

    next_t = t + 1
    if next_t >= market_n:
        next_t = market_n - 1
    insolvent_on_mark = _settle_insolvency_jit(portfolio_next, float(market_close[next_t]))
    done_out = next_t >= market_n - 1 or insolvent_on_fill or insolvent_on_mark
    next_equity = portfolio_next[0] + portfolio_next[1] * float(market_close[next_t])
    equity_delta = next_equity - prev_equity

    return (
        next_t,
        done_out,
        portfolio_next,
        order_active_next,
        order_side_next,
        order_type_next,
        order_units_next,
        order_limit_price_next,
        order_eligible_t_next,
        order_ttl_next,
        fills,
        equity_delta,
    )


def step_core_numba(
    state: CoreState,
    action: Action,
    market_arrays: MarketArrays,
    config: GameConfig,
    random_draws: NDArray[np_float64],
) -> CoreStepOutput:
    action_side = _action_side_code(action)
    action_order_type = _ORDER_LIMIT if action.order_type == "limit" else _ORDER_MARKET
    action_limit_price = float(action.limit_price) if action.limit_price is not None else 0.0
    (
        next_t,
        next_done,
        next_portfolio,
        next_order_active,
        next_order_side,
        next_order_type,
        next_order_units,
        next_order_limit_price,
        next_order_eligible_t,
        next_order_ttl,
        fills,
        equity_delta,
    ) = step_core_jit(
        t=state.t,
        done=state.done,
        portfolio=state.portfolio,
        order_active=state.order_active,
        order_side=state.order_side,
        order_type=state.order_type,
        order_units=state.order_units,
        order_limit_price=state.order_limit_price,
        order_eligible_t=state.order_eligible_t,
        order_ttl=state.order_ttl,
        action_side=action_side,
        action_units=float(action.units),
        action_order_type=action_order_type,
        action_limit_price=action_limit_price,
        market_open=market_arrays.open,
        market_high=market_arrays.high,
        market_low=market_arrays.low,
        market_close=market_arrays.close,
        market_n=market_arrays.n,
        market_latency_bars=config.market_latency_bars,
        limit_ttl_bars=config.limit_ttl_bars,
        random_draws=random_draws,
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
        max_leverage=config.max_leverage,
    )
    return CoreStepOutput(
        state=CoreState(
            t=next_t,
            done=next_done,
            portfolio=next_portfolio,
            order_active=next_order_active,
            order_side=next_order_side,
            order_type=next_order_type,
            order_units=next_order_units,
            order_limit_price=next_order_limit_price,
            order_eligible_t=next_order_eligible_t,
            order_ttl=next_order_ttl,
        ),
        fills=fills,
        equity_delta=equity_delta,
    )


def _copy_state(state: CoreState) -> CoreState:
    return CoreState(
        t=state.t,
        done=state.done,
        portfolio=state.portfolio.copy(),
        order_active=state.order_active.copy(),
        order_side=state.order_side.copy(),
        order_type=state.order_type.copy(),
        order_units=state.order_units.copy(),
        order_limit_price=state.order_limit_price.copy(),
        order_eligible_t=state.order_eligible_t.copy(),
        order_ttl=state.order_ttl.copy(),
    )


def _action_side_code(action: Action) -> np_int8:
    if action.side == "buy":
        return _BUY_SIDE
    if action.side == "sell":
        return _SELL_SIDE
    return np_int8(0)


def _apply_action(state: CoreState, action: Action, config: GameConfig) -> None:
    if action.side == "hold":
        return
    if action.units <= 0:
        raise ValueError("units must be > 0 for buy/sell actions")

    side = _BUY_SIDE if action.side == "buy" else _SELL_SIDE
    order_type = _ORDER_LIMIT if action.order_type == "limit" else _ORDER_MARKET
    limit_price = float(action.limit_price) if action.limit_price is not None else 0.0
    slot = _find_side_slot_or_free(state, side)
    if slot < 0:
        raise ValueError(f"order capacity exceeded (max={state.order_active.size})")

    state.order_active[slot] = np_int8(1)
    state.order_side[slot] = side
    state.order_type[slot] = order_type
    state.order_units[slot] = float(action.units)
    state.order_limit_price[slot] = limit_price
    state.order_eligible_t[slot] = np_int32(state.t + config.market_latency_bars)
    state.order_ttl[slot] = np_int32(config.limit_ttl_bars)


def _find_side_slot_or_free(state: CoreState, side: np_int8) -> int:
    max_orders = state.order_active.size
    free_slot = -1
    for i in range(max_orders):
        if state.order_active[i] == 1 and state.order_side[i] == side:
            return i
        if free_slot < 0 and state.order_active[i] == 0:
            free_slot = i
    return free_slot


def _match_orders(
    state: CoreState,
    market_arrays: MarketArrays,
    config: GameConfig,
    random_draws: NDArray[np_float64],
) -> int:
    open_price = float(market_arrays.open[state.t])
    high = float(market_arrays.high[state.t])
    low = float(market_arrays.low[state.t])
    fills = 0

    for i in range(state.order_active.size):
        if state.order_active[i] == 0:
            continue
        if state.t < int(state.order_eligible_t[i]):
            continue

        side = state.order_side[i]
        order_type = state.order_type[i]
        limit_price = float(state.order_limit_price[i])
        fill_price = open_price
        filled = False

        if order_type == _ORDER_MARKET:
            filled = True
        elif (side == _BUY_SIDE and low <= limit_price) or (side == _SELL_SIDE and high >= limit_price):
            filled = True
            fill_price = limit_price

        if filled:
            fraction = float(random_draws[i])
            requested_units = float(side) * float(state.order_units[i]) * fraction
            exec_price = _execution_price(fill_price, requested_units, config.slippage_bps)
            signed_units = clip_units_for_leverage(
                float(state.portfolio[0]),
                float(state.portfolio[1]),
                requested_units,
                exec_price,
                config.fee_bps,
                config.max_leverage,
            )
            _execute_trade(state.portfolio, signed_units, exec_price, config.fee_bps)
            state.order_active[i] = np_int8(0)
            state.order_ttl[i] = np_int32(0)
            fills += 1
            continue

        ttl = int(state.order_ttl[i]) - 1
        state.order_ttl[i] = np_int32(ttl)
        if ttl <= 0:
            state.order_active[i] = np_int8(0)
            state.order_ttl[i] = np_int32(0)

    return fills


def _execution_price(fill_price: float, signed_units: float, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    return fill_price * (1 + slip) if signed_units > 0 else fill_price * (1 - slip)


def _execute_trade(
    portfolio: NDArray[np_float64],
    signed_units: float,
    exec_price: float,
    fee_bps: float,
) -> None:
    notional = abs(signed_units) * exec_price
    fee = notional * (fee_bps / 10_000.0)
    portfolio[0] -= signed_units * exec_price + fee
    portfolio[1] += signed_units


def _equity(portfolio: NDArray[np_float64], price: float) -> float:
    return float(portfolio[0] + portfolio[1] * price)
