from __future__ import annotations

from importlib import import_module as importlib_import_module
from types import ModuleType
from typing import Protocol, cast

from grpc import Channel as grpc_Channel
from grpc import insecure_channel as grpc_insecure_channel
from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from numpy import isnan as np_isnan
from numpy.typing import NDArray

type ObservationDict = dict[
    str,
    bool | list[float] | dict[str, int | float],
]
type StepTraceRowDict = dict[str, int | float | bool | None]


class ProtoMessage(Protocol):
    """Marker protocol for protobuf request/response objects."""


def _load_engine_pb2_module() -> ModuleType | None:
    """Load generated protobuf messages module when present."""
    try:
        return importlib_import_module("stocksim_grpc.engine_pb2")
    except ModuleNotFoundError:
        return None


def _load_engine_pb2_grpc_module() -> ModuleType | None:
    """Load generated gRPC stubs module when present."""
    try:
        return importlib_import_module("stocksim_grpc.engine_pb2_grpc")
    except ModuleNotFoundError:
        return None


engine_pb2 = _load_engine_pb2_module()
engine_pb2_grpc = _load_engine_pb2_grpc_module()


class _PingRequestFactory(Protocol):
    def __call__(self) -> ProtoMessage: ...


class _ResetRequestFactory(Protocol):
    def __call__(
        self,
        *,
        has_seed: bool,
        seed: int,
        has_start_t: bool,
        start_t: int,
    ) -> ProtoMessage: ...


class _ObserveRequestFactory(Protocol):
    def __call__(self) -> ProtoMessage: ...


class _EncodedActionFactory(Protocol):
    def __call__(
        self,
        *,
        side_code: int,
        units: float,
        order_type_code: int,
        has_limit_price: bool,
        limit_price: float,
    ) -> ProtoMessage: ...


class _StepManyRequestFactory(Protocol):
    def __call__(self, *, actions: list[ProtoMessage], include_trace: bool) -> ProtoMessage: ...


class _Pb2Protocol(Protocol):
    PingRequest: _PingRequestFactory
    ResetRequest: _ResetRequestFactory
    ObserveRequest: _ObserveRequestFactory
    EncodedAction: _EncodedActionFactory
    StepManyRequest: _StepManyRequestFactory


class _MarketHandleLike(Protocol):
    start: int
    end: int
    t: int
    current_price: float


class _StateLike(Protocol):
    t: int
    cash: float
    units: float
    equity: float
    leverage: float
    open_orders: int
    done: bool


class _ObservationLike(Protocol):
    market_window_handle: _MarketHandleLike
    portfolio_vector: list[float]
    order_summary_vector: list[float]
    done: bool


class _PingResponseLike(Protocol):
    status: str


class _ResetResponseLike(Protocol):
    state: _StateLike


class _ObserveResponseLike(Protocol):
    observation: _ObservationLike


class _StepManyResponseLike(Protocol):
    observations: list[_ObservationLike]
    rewards: list[float]
    dones: list[bool]
    trace: list[object]


class _StepTraceRowLike(Protocol):
    index: int
    side_code: int
    requested_units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float
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


class _EngineStubProtocol(Protocol):
    def Ping(self, request: ProtoMessage) -> _PingResponseLike: ...

    def Reset(self, request: ProtoMessage) -> _ResetResponseLike: ...

    def Observe(self, request: ProtoMessage) -> _ObserveResponseLike: ...

    def StepMany(self, request: ProtoMessage) -> _StepManyResponseLike: ...


class _EngineServiceStubFactory(Protocol):
    def __call__(self, channel: grpc_Channel) -> _EngineStubProtocol: ...


class _Pb2GrpcProtocol(Protocol):
    EngineServiceStub: _EngineServiceStubFactory


def _require_engine_pb2() -> _Pb2Protocol:
    """Return generated protobuf messages module or raise actionable error."""
    if engine_pb2 is None:
        raise RuntimeError("gRPC protobuf messages are unavailable. Run ./scripts/generate_grpc_stubs.sh first.")
    return cast(_Pb2Protocol, engine_pb2)


def _require_engine_pb2_grpc() -> _Pb2GrpcProtocol:
    """Return generated grpc stubs module or raise actionable error."""
    if engine_pb2_grpc is None:
        raise RuntimeError("gRPC stubs are unavailable. Run ./scripts/generate_grpc_stubs.sh first.")
    return cast(_Pb2GrpcProtocol, engine_pb2_grpc)


class EngineGrpcClient:
    def __init__(self, *, target: str = "127.0.0.1:50051") -> None:
        self._channel = grpc_insecure_channel(target)
        pb2_grpc = _require_engine_pb2_grpc()
        self._stub = pb2_grpc.EngineServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def ping(self) -> str:
        pb2 = _require_engine_pb2()
        response = self._stub.Ping(pb2.PingRequest())
        return str(response.status)

    def reset(self, seed: int | None = None, start_t: int | None = None) -> dict[str, float | int | bool]:
        pb2 = _require_engine_pb2()
        request = pb2.ResetRequest(
            has_seed=seed is not None,
            seed=int(seed or 0),
            has_start_t=start_t is not None,
            start_t=int(start_t or 0),
        )
        response = self._stub.Reset(request)
        state = response.state
        return {
            "t": int(state.t),
            "cash": float(state.cash),
            "units": float(state.units),
            "equity": float(state.equity),
            "leverage": float(state.leverage),
            "open_orders": int(state.open_orders),
            "done": bool(state.done),
        }

    def observe(self) -> ObservationDict:
        pb2 = _require_engine_pb2()
        response = self._stub.Observe(pb2.ObserveRequest())
        observation = response.observation
        return {
            "market_window_handle": {
                "start": int(observation.market_window_handle.start),
                "end": int(observation.market_window_handle.end),
                "t": int(observation.market_window_handle.t),
                "current_price": float(observation.market_window_handle.current_price),
            },
            "portfolio_vector": list(observation.portfolio_vector),
            "order_summary_vector": list(observation.order_summary_vector),
            "done": bool(observation.done),
        }

    def step_many(
        self, actions: NDArray[np_float64]
    ) -> tuple[list[ObservationDict], NDArray[np_float64], NDArray[np_bool_]]:
        observations, rewards, dones, _ = self._step_many_internal(actions, include_trace=False)
        return observations, rewards, dones

    def step_many_with_trace(
        self, actions: NDArray[np_float64]
    ) -> tuple[list[ObservationDict], NDArray[np_float64], NDArray[np_bool_], list[StepTraceRowDict]]:
        return self._step_many_internal(actions, include_trace=True)

    def _step_many_internal(
        self,
        actions: NDArray[np_float64],
        *,
        include_trace: bool,
    ) -> tuple[list[ObservationDict], NDArray[np_float64], NDArray[np_bool_], list[StepTraceRowDict]]:
        pb2 = _require_engine_pb2()
        encoded_actions: list[ProtoMessage] = []
        for row in actions:
            encoded_actions.append(
                pb2.EncodedAction(
                    side_code=int(row[0]),
                    units=float(row[1]),
                    order_type_code=int(row[2]),
                    has_limit_price=not np_isnan(row[3]),
                    limit_price=0.0 if np_isnan(row[3]) else float(row[3]),
                )
            )
        response = self._stub.StepMany(pb2.StepManyRequest(actions=encoded_actions, include_trace=include_trace))
        obs: list[ObservationDict] = []
        for row in response.observations:
            obs.append(
                {
                    "market_window_handle": {
                        "start": int(row.market_window_handle.start),
                        "end": int(row.market_window_handle.end),
                        "t": int(row.market_window_handle.t),
                        "current_price": float(row.market_window_handle.current_price),
                    },
                    "portfolio_vector": list(row.portfolio_vector),
                    "order_summary_vector": list(row.order_summary_vector),
                    "done": bool(row.done),
                }
            )
        trace: list[StepTraceRowDict] = []
        for row in response.trace:
            trace_row = cast(_StepTraceRowLike, row)
            trace.append(
                {
                    "index": int(trace_row.index),
                    "side_code": int(trace_row.side_code),
                    "requested_units": float(trace_row.requested_units),
                    "order_type_code": int(trace_row.order_type_code),
                    "has_limit_price": bool(trace_row.has_limit_price),
                    "limit_price": (float(trace_row.limit_price) if trace_row.has_limit_price else None),
                    "fills": int(trace_row.fills),
                    "reward": float(trace_row.reward),
                    "done": bool(trace_row.done),
                    "t": int(trace_row.t),
                    "cash": float(trace_row.cash),
                    "position_units": float(trace_row.position_units),
                    "equity": float(trace_row.equity),
                    "leverage": float(trace_row.leverage),
                    "open_orders": int(trace_row.open_orders),
                    "market_price": float(trace_row.market_price),
                }
            )
        return (
            obs,
            np_asarray(response.rewards, dtype=np_float64),
            np_asarray(response.dones, dtype=np_bool_),
            trace,
        )
