from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol, TypeAlias, cast

import grpc
import numpy as np
from numpy.typing import NDArray

ObservationDict: TypeAlias = dict[
    str,
    bool | list[float] | dict[str, int | float],
]


class ProtoMessage(Protocol):
    """Marker protocol for protobuf request/response objects."""


def _load_grpc_module(module_name: str) -> ModuleType | None:
    """Load generated grpc module by name when present."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


engine_pb2 = _load_grpc_module("stocksim_grpc.engine_pb2")
engine_pb2_grpc = _load_grpc_module("stocksim_grpc.engine_pb2_grpc")


class _PingRequestFactory(Protocol):
    def __call__(self) -> ProtoMessage: ...


class _ResetRequestFactory(Protocol):
    def __call__(self, *, has_seed: bool, seed: int) -> ProtoMessage: ...


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
    def __call__(self, *, actions: list[ProtoMessage]) -> ProtoMessage: ...


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


class _EngineStubProtocol(Protocol):
    def Ping(self, request: ProtoMessage) -> _PingResponseLike: ...

    def Reset(self, request: ProtoMessage) -> _ResetResponseLike: ...

    def Observe(self, request: ProtoMessage) -> _ObserveResponseLike: ...

    def StepMany(self, request: ProtoMessage) -> _StepManyResponseLike: ...


class _EngineServiceStubFactory(Protocol):
    def __call__(self, channel: grpc.Channel) -> _EngineStubProtocol: ...


class _Pb2GrpcProtocol(Protocol):
    EngineServiceStub: _EngineServiceStubFactory


def _require_engine_pb2() -> _Pb2Protocol:
    """Return generated protobuf messages module or raise actionable error."""
    if engine_pb2 is None:
        raise RuntimeError(
            "gRPC protobuf messages are unavailable. Run ./scripts/generate_grpc_stubs.sh first."
        )
    return cast(_Pb2Protocol, engine_pb2)


def _require_engine_pb2_grpc() -> _Pb2GrpcProtocol:
    """Return generated grpc stubs module or raise actionable error."""
    if engine_pb2_grpc is None:
        raise RuntimeError(
            "gRPC stubs are unavailable. Run ./scripts/generate_grpc_stubs.sh first."
        )
    return cast(_Pb2GrpcProtocol, engine_pb2_grpc)


class EngineGrpcClient:
    def __init__(self, *, target: str = "127.0.0.1:50051") -> None:
        self._channel = grpc.insecure_channel(target)
        pb2_grpc = _require_engine_pb2_grpc()
        self._stub = pb2_grpc.EngineServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def ping(self) -> str:
        pb2 = _require_engine_pb2()
        response = self._stub.Ping(pb2.PingRequest())
        return str(response.status)

    def reset(self, seed: int | None = None) -> dict[str, float | int | bool]:
        pb2 = _require_engine_pb2()
        request = pb2.ResetRequest(has_seed=seed is not None, seed=int(seed or 0))
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
        self, actions: NDArray[np.float64]
    ) -> tuple[list[ObservationDict], NDArray[np.float64], NDArray[np.bool_]]:
        pb2 = _require_engine_pb2()
        encoded_actions: list[ProtoMessage] = []
        for row in actions:
            encoded_actions.append(
                pb2.EncodedAction(
                    side_code=int(row[0]),
                    units=float(row[1]),
                    order_type_code=int(row[2]),
                    has_limit_price=not np.isnan(row[3]),
                    limit_price=0.0 if np.isnan(row[3]) else float(row[3]),
                )
            )
        response = self._stub.StepMany(pb2.StepManyRequest(actions=encoded_actions))
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
        return (
            obs,
            np.asarray(response.rewards, dtype=np.float64),
            np.asarray(response.dones, dtype=np.bool_),
        )
