from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy import asarray as np_asarray
from numpy import float64 as np_float64
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

Side = Literal["buy", "sell", "hold"]
OrderType = Literal["market", "limit"]
type SerializedObservation = dict[str, int | float | bool | list[float] | dict[str, int | float]]
type SerializedStepTraceRow = dict[str, int | float | bool | None]


@dataclass(frozen=True)
class Action:
    """Action issued to the environment for one step.

    Parameters
    ----------
    side
        Trade direction.
    units
        Requested quantity. Must be zero for ``"hold"``.
    order_type
        Execution type, either immediate market order or price-constrained limit order.
    limit_price
        Limit trigger price. Required for non-hold limit orders.
    """

    side: Side
    units: float = 0.0
    order_type: OrderType = "market"
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell", "hold"}:
            raise ValueError("side must be one of: buy, sell, hold")
        if self.order_type not in {"market", "limit"}:
            raise ValueError("order_type must be one of: market, limit")
        if self.units < 0:
            raise ValueError("units must be >= 0")
        if self.side == "hold" and self.units != 0:
            raise ValueError("units must be 0 for hold actions")
        if self.side == "hold" and self.limit_price is not None:
            raise ValueError("limit_price is not valid for hold actions")
        if self.order_type == "limit" and self.side != "hold" and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")


@dataclass(frozen=True)
class MarketWindowViewHandle:
    """Index-based view metadata for the active market observation window.

    Parameters
    ----------
    start
        Inclusive start index of the window.
    end
        Exclusive end index of the window.
    t
        Current bar index.
    current_price
        Current close price at ``t``.
    """

    start: int
    end: int
    t: int
    current_price: float

    def to_numpy(self) -> NDArray[np_float64]:
        """Convert the view handle to a dense NumPy vector.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(4,)`` ordered as ``[start, end, t, current_price]``.
        """
        return np_asarray(
            [self.start, self.end, self.t, self.current_price],
            dtype=np_float64,
        )


@dataclass(frozen=True)
class MarketWindowContent:
    """Raw OHLCV rows for a resolved market window ``[start, end)``.

    Unlike :class:`MarketWindowViewHandle` (index metadata only), this carries the
    actual per-bar values, including ``volume`` which is intentionally absent from
    the numba step core's ``MarketArrays``. All sequences share the same length,
    ``end - start``.

    Parameters
    ----------
    start
        Inclusive start index of the window.
    end
        Exclusive end index of the window. Never exceeds ``t + 1`` (no lookahead).
    t
        Current bar index the window was resolved against.
    open, high, low, close, volume, taker_buy_volume
        Per-bar values for indices ``[start, end)``. ``taker_buy_volume`` is
        zero-filled when the underlying data source doesn't carry it (e.g. CSV
        fixtures) — see :class:`~stock_simulator.data.MarketData`.
    """

    start: int
    end: int
    t: int
    open: tuple[float, ...]
    high: tuple[float, ...]
    low: tuple[float, ...]
    close: tuple[float, ...]
    volume: tuple[float, ...]
    taker_buy_volume: tuple[float, ...]


@dataclass(frozen=True)
class Observation:
    """Environment observation exposed to agents.

    Parameters
    ----------
    market
        Handle describing the market slice used for this step.
    portfolio_vector
        Dense vector ``[cash, units, equity, leverage]``.
    order_summary_vector
        Dense vector ``[open_orders, buy_open_orders, sell_open_orders]``.
    done
        Terminal flag.
    """

    market: MarketWindowViewHandle
    portfolio_vector: NDArray[np_float64]
    order_summary_vector: NDArray[np_float64]
    done: bool

    @property
    def t(self) -> int:
        return self.market.t

    @property
    def price(self) -> float:
        return self.market.current_price

    @property
    def cash(self) -> float:
        return float(self.portfolio_vector[0])

    @property
    def units(self) -> float:
        return float(self.portfolio_vector[1])

    @property
    def equity(self) -> float:
        return float(self.portfolio_vector[2])

    @property
    def leverage(self) -> float:
        return float(self.portfolio_vector[3])

    @property
    def open_orders(self) -> int:
        return int(self.order_summary_vector[0])

    def to_numpy_tensors(self) -> dict[str, NDArray[np_float64]]:
        """Convert observation into tensor-ready NumPy values.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary containing market handle, portfolio vector, order summary
            vector, and done flag tensor.
        """
        return {
            "market_window_handle": self.market.to_numpy(),
            "portfolio_vector": self.portfolio_vector.copy(),
            "order_summary_vector": self.order_summary_vector.copy(),
            "done": np_asarray([1.0 if self.done else 0.0], dtype=np_float64),
        }

    def to_serializable(self) -> SerializedObservation:
        """Convert observation into JSON-serializable primitives.

        Returns
        -------
        SerializedObservation
            Dictionary with plain Python scalars, lists, and nested dictionaries.
        """
        return {
            "market_window_handle": {
                "start": self.market.start,
                "end": self.market.end,
                "t": self.market.t,
                "current_price": self.market.current_price,
            },
            "portfolio_vector": self.portfolio_vector.tolist(),
            "order_summary_vector": self.order_summary_vector.tolist(),
            "done": self.done,
        }


@dataclass(frozen=True)
class EnvState:
    """Compact environment state after reset/step."""

    t: int
    cash: float
    units: float
    equity: float
    leverage: float
    open_orders: int
    done: bool


@dataclass(frozen=True)
class StepTraceRow:
    """Line-by-line execution trace row for one processed action.

    ``has_filled_order``/``filled_order_slot``/``exec_price`` are additive (N9)
    fields recovering the actually-filled order's real execution price —
    distinct from ``limit_price``, which is the *submitted* action's requested
    limit, not necessarily the price an older queued order filled at on this
    step. ``has_filled_order`` mirrors the existing ``has_limit_price`` /
    ``limit_price`` presence-flag convention; all three are falsy/``None`` when
    ``fills == 0``, and describe the *last* fill processed this step when
    ``fills > 0`` (see :class:`stock_simulator.core.CoreStepOutput`).
    """

    index: int
    side_code: int
    requested_units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float | None
    fills: int
    reward: float
    done: bool
    t: int
    cash: float
    position_units: float
    equity: float
    leverage: float
    open_orders: int
    market_price: float
    has_filled_order: bool = False
    filled_order_slot: int | None = None
    exec_price: float | None = None

    def to_serializable(self) -> SerializedStepTraceRow:
        """Convert trace row into JSON-serializable primitives."""
        return {
            "index": self.index,
            "side_code": self.side_code,
            "requested_units": self.requested_units,
            "order_type_code": self.order_type_code,
            "has_limit_price": self.has_limit_price,
            "limit_price": self.limit_price,
            "fills": self.fills,
            "reward": self.reward,
            "done": self.done,
            "t": self.t,
            "cash": self.cash,
            "position_units": self.position_units,
            "equity": self.equity,
            "leverage": self.leverage,
            "open_orders": self.open_orders,
            "market_price": self.market_price,
            "has_filled_order": self.has_filled_order,
            "filled_order_slot": self.filled_order_slot,
            "exec_price": self.exec_price,
        }


@dataclass(frozen=True)
class StepResult:
    """Result container returned by :meth:`stock_simulator.env.MarketEnv.step`."""

    state: EnvState
    observation: Observation
    done: bool


class ActionModel(BaseModel):
    """Pydantic transport model for :class:`Action`."""

    model_config = ConfigDict(extra="forbid")

    side: Side
    units: float = 0.0
    order_type: OrderType = "market"
    limit_price: float | None = None

    def to_action(self) -> Action:
        """Convert validated transport data into the dataclass action."""
        return Action(
            side=self.side,
            units=self.units,
            order_type=self.order_type,
            limit_price=self.limit_price,
        )


class MarketWindowViewHandleModel(BaseModel):
    """Pydantic transport model for :class:`MarketWindowViewHandle`."""

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    t: int
    current_price: float

    @classmethod
    def from_dataclass(cls, value: MarketWindowViewHandle) -> MarketWindowViewHandleModel:
        """Create model from dataclass value."""
        return cls(
            start=value.start,
            end=value.end,
            t=value.t,
            current_price=value.current_price,
        )


class MarketWindowRowsModel(BaseModel):
    """Per-bar OHLCV column arrays for a market window."""

    model_config = ConfigDict(extra="forbid")

    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    taker_buy_volume: list[float]


class MarketWindowContentModel(BaseModel):
    """Pydantic transport model for :class:`MarketWindowContent`."""

    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    t: int
    rows: MarketWindowRowsModel

    @classmethod
    def from_dataclass(cls, value: MarketWindowContent) -> MarketWindowContentModel:
        """Create model from dataclass value."""
        return cls(
            start=value.start,
            end=value.end,
            t=value.t,
            rows=MarketWindowRowsModel(
                open=list(value.open),
                high=list(value.high),
                low=list(value.low),
                close=list(value.close),
                volume=list(value.volume),
                taker_buy_volume=list(value.taker_buy_volume),
            ),
        )


class ObservationModel(BaseModel):
    """Pydantic transport model for :class:`Observation`."""

    model_config = ConfigDict(extra="forbid")

    market_window_handle: MarketWindowViewHandleModel
    portfolio_vector: list[float]
    order_summary_vector: list[float]
    done: bool

    @classmethod
    def from_dataclass(cls, value: Observation) -> ObservationModel:
        """Create model from dataclass value."""
        return cls(
            market_window_handle=MarketWindowViewHandleModel.from_dataclass(value.market),
            portfolio_vector=value.portfolio_vector.tolist(),
            order_summary_vector=value.order_summary_vector.tolist(),
            done=value.done,
        )


class EnvStateModel(BaseModel):
    """Pydantic transport model for :class:`EnvState`."""

    model_config = ConfigDict(extra="forbid")

    t: int
    cash: float
    units: float
    equity: float
    leverage: float
    open_orders: int
    done: bool

    @classmethod
    def from_dataclass(cls, value: EnvState) -> EnvStateModel:
        """Create model from dataclass value."""
        return cls(
            t=value.t,
            cash=value.cash,
            units=value.units,
            equity=value.equity,
            leverage=value.leverage,
            open_orders=value.open_orders,
            done=value.done,
        )


class StepTraceRowModel(BaseModel):
    """Pydantic transport model for :class:`StepTraceRow`."""

    model_config = ConfigDict(extra="forbid")

    index: int
    side_code: int
    requested_units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float | None = None
    fills: int
    reward: float
    done: bool
    t: int
    cash: float
    position_units: float
    equity: float
    leverage: float
    open_orders: int
    market_price: float
    has_filled_order: bool = False
    filled_order_slot: int | None = None
    exec_price: float | None = None

    @classmethod
    def from_dataclass(cls, value: StepTraceRow) -> StepTraceRowModel:
        """Create model from dataclass value."""
        return cls(
            index=value.index,
            side_code=value.side_code,
            requested_units=value.requested_units,
            order_type_code=value.order_type_code,
            has_limit_price=value.has_limit_price,
            limit_price=value.limit_price,
            fills=value.fills,
            reward=value.reward,
            done=value.done,
            t=value.t,
            cash=value.cash,
            position_units=value.position_units,
            equity=value.equity,
            leverage=value.leverage,
            open_orders=value.open_orders,
            market_price=value.market_price,
            has_filled_order=value.has_filled_order,
            filled_order_slot=value.filled_order_slot,
            exec_price=value.exec_price,
        )


class StepResultModel(BaseModel):
    """Pydantic transport model for :class:`StepResult`."""

    model_config = ConfigDict(extra="forbid")

    state: EnvStateModel
    observation: ObservationModel
    done: bool

    @classmethod
    def from_dataclass(cls, value: StepResult) -> StepResultModel:
        """Create model from dataclass value."""
        return cls(
            state=EnvStateModel.from_dataclass(value.state),
            observation=ObservationModel.from_dataclass(value.observation),
            done=value.done,
        )
