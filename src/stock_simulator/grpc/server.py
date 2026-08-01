from __future__ import annotations

from argparse import ArgumentParser as argparse_ArgumentParser
from concurrent.futures import ThreadPoolExecutor as concurrentfutures_ThreadPoolExecutor
from importlib import import_module as importlib_import_module
from os import getenv as os_getenv
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from numpy import asarray as np_asarray
from numpy import cumprod as np_cumprod
from numpy import float64 as np_float64
from numpy import maximum as np_maximum
from numpy import minimum as np_minimum
from numpy import nan as np_nan
from numpy import random as np_random
from numpy import roll as np_roll
from numpy.typing import NDArray
from pandas import DataFrame as pd_DataFrame
from pandas import date_range as pd_date_range

from ..config import GameConfig
from ..data import MarketData
from ..env import MarketEnv
from ..types import EnvState, Observation

if TYPE_CHECKING:
    from grpc import Server as grpc_Server
    from grpc import ServicerContext as grpc_ServicerContext
else:
    grpc_Server = object
    grpc_ServicerContext = object


class ProtoMessage(Protocol):
    """Marker protocol for protobuf request/response objects."""


class _GrpcStatusCode(Protocol):
    INVALID_ARGUMENT: object


class _GrpcServerFactory(Protocol):
    def __call__(self, thread_pool: concurrentfutures_ThreadPoolExecutor) -> grpc_Server: ...


class _GrpcModule(Protocol):
    StatusCode: _GrpcStatusCode

    def server(self, thread_pool: concurrentfutures_ThreadPoolExecutor) -> grpc_Server: ...


def _load_grpc_module() -> ModuleType | None:
    """Load grpc runtime module when present."""
    try:
        return importlib_import_module("grpc")
    except ModuleNotFoundError:
        return None


grpc_module = _load_grpc_module()
grpc_server = cast(_GrpcServerFactory | None, getattr(grpc_module, "server", None)) if grpc_module is not None else None
grpc_StatusCode = cast(_GrpcStatusCode | None, getattr(grpc_module, "StatusCode", None)) if grpc_module else None
_GRPC_RUNTIME_UNAVAILABLE = "gRPC runtime is unavailable. Install with `uv sync --extra grpc`."


def _require_grpc_module() -> _GrpcModule:
    """Return grpc runtime module or raise actionable error."""
    if grpc_module is None:
        raise RuntimeError(_GRPC_RUNTIME_UNAVAILABLE)
    return cast(_GrpcModule, grpc_module)


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


class _MarketWindowFactory(Protocol):
    def __call__(self, *, start: int, end: int, t: int, current_price: float) -> ProtoMessage: ...


class _ObservationFactory(Protocol):
    def __call__(
        self,
        *,
        market_window_handle: ProtoMessage,
        portfolio_vector: list[float],
        order_summary_vector: list[float],
        done: bool,
    ) -> ProtoMessage: ...


class _EnvStateFactory(Protocol):
    def __call__(
        self,
        *,
        t: int,
        cash: float,
        units: float,
        equity: float,
        leverage: float,
        open_orders: int,
        done: bool,
    ) -> ProtoMessage: ...


class _PingResponseFactory(Protocol):
    def __call__(self, *, status: str) -> ProtoMessage: ...


class _ResetResponseFactory(Protocol):
    def __call__(self, *, state: ProtoMessage) -> ProtoMessage: ...


class _ObserveResponseFactory(Protocol):
    def __call__(self, *, observation: ProtoMessage) -> ProtoMessage: ...


class _StepManyResponseFactory(Protocol):
    def __call__(
        self,
        *,
        observations: list[ProtoMessage] = ...,
        rewards: list[float] = ...,
        dones: list[bool] = ...,
        trace: list[ProtoMessage] = ...,
    ) -> ProtoMessage: ...


class _StepTraceRowFactory(Protocol):
    def __call__(
        self,  # NOSONAR — parameter count mirrors protobuf-generated StepTraceRow fields; cannot be reduced
        *,
        index: int,
        side_code: int,
        requested_units: float,
        order_type_code: int,
        has_limit_price: bool,
        limit_price: float,
        fills: int,
        reward: float,
        done: bool,
        t: int,
        cash: float,
        position_units: float,
        equity: float,
        leverage: float,
        open_orders: int,
        market_price: float,
    ) -> ProtoMessage: ...


class _Pb2Protocol(Protocol):
    MarketWindowViewHandle: _MarketWindowFactory
    Observation: _ObservationFactory
    EnvState: _EnvStateFactory
    StepTraceRow: _StepTraceRowFactory
    PingResponse: _PingResponseFactory
    ResetResponse: _ResetResponseFactory
    ObserveResponse: _ObserveResponseFactory
    StepManyResponse: _StepManyResponseFactory


class _ActionLike(Protocol):
    side_code: int
    units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float


class _RequestWithSeed(Protocol):
    has_seed: bool
    seed: int
    has_start_t: bool
    start_t: int


class _RequestWithActions(Protocol):
    actions: list[_ActionLike]
    include_trace: bool


class _GrpcStubsProtocol(Protocol):
    def add_EngineServiceServicer_to_server(self, servicer: EngineGrpcService, server: grpc_Server) -> None: ...


def _require_engine_pb2() -> _Pb2Protocol:
    """Return generated protobuf messages module or raise actionable error."""
    if engine_pb2 is None:
        raise RuntimeError("gRPC protobuf messages are unavailable. Run ./scripts/generate_grpc_stubs.sh first.")
    return cast(_Pb2Protocol, engine_pb2)


def _require_engine_pb2_grpc() -> _GrpcStubsProtocol:
    """Return generated grpc stubs module or raise actionable error."""
    if engine_pb2_grpc is None:
        raise RuntimeError("gRPC stubs are unavailable. Run ./scripts/generate_grpc_stubs.sh first.")
    return cast(_GrpcStubsProtocol, engine_pb2_grpc)


def _build_synthetic_market_data(bars: int = 1024, seed: int = 1234) -> MarketData:
    rng = np_random.default_rng(seed)
    returns = rng.normal(loc=0.0001, scale=0.01, size=bars)
    close = 100.0 * np_cumprod(1.0 + returns)
    open_ = np_roll(close, 1)
    open_[0] = close[0]
    spread = rng.uniform(0.0002, 0.01, size=bars)
    high = np_maximum(open_, close) * (1.0 + spread)
    low = np_minimum(open_, close) * (1.0 - spread)
    frame = pd_DataFrame(
        {
            "timestamp": pd_date_range("2024-01-01", periods=bars, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(10_000.0, 100_000.0, size=bars),
        }
    )
    return MarketData(frame)


def _load_market_data() -> MarketData:
    csv_path = os_getenv("MARKET_DATA_CSV", "").strip()
    if csv_path:
        return MarketData.from_csv(csv_path)
    bars = int(os_getenv("IPC_SYNTHETIC_BARS", "1024"))
    seed = int(os_getenv("IPC_SYNTHETIC_SEED", "1234"))
    return _build_synthetic_market_data(bars=bars, seed=seed)


def _observation_to_proto(observation: Observation) -> ProtoMessage:
    pb2 = _require_engine_pb2()
    return pb2.Observation(
        market_window_handle=pb2.MarketWindowViewHandle(
            start=observation.market.start,
            end=observation.market.end,
            t=observation.market.t,
            current_price=observation.market.current_price,
        ),
        portfolio_vector=observation.portfolio_vector.tolist(),
        order_summary_vector=observation.order_summary_vector.tolist(),
        done=observation.done,
    )


def _state_to_proto(state: EnvState) -> ProtoMessage:
    pb2 = _require_engine_pb2()
    return pb2.EnvState(
        t=state.t,
        cash=state.cash,
        units=state.units,
        equity=state.equity,
        leverage=state.leverage,
        open_orders=state.open_orders,
        done=state.done,
    )


class EngineGrpcService:
    def __init__(self, env: MarketEnv) -> None:
        self._env = env

    # gRPC-generated server dispatch expects PascalCase RPC method names.
    def Ping(self, request: ProtoMessage, context: grpc_ServicerContext | None) -> ProtoMessage:  # noqa: N802  # NOSONAR
        _ = (request, context)
        pb2 = _require_engine_pb2()
        return pb2.PingResponse(status="pong")

    def Reset(  # noqa: N802  # NOSONAR
        self, request: _RequestWithSeed, context: grpc_ServicerContext | None
    ) -> ProtoMessage:
        pb2 = _require_engine_pb2()
        seed: int | None = int(request.seed) if request.has_seed else None
        start_t: int = int(request.start_t) if request.has_start_t else 0
        try:
            state = self._env.reset(seed=seed, start_t=start_t)
        except ValueError as exc:
            if context is None:
                raise
            if grpc_StatusCode is None:
                raise RuntimeError(_GRPC_RUNTIME_UNAVAILABLE) from exc
            context.abort(grpc_StatusCode.INVALID_ARGUMENT, str(exc))
            raise AssertionError("context.abort should raise") from exc
        return pb2.ResetResponse(state=_state_to_proto(state))

    def Observe(self, request: ProtoMessage, context: grpc_ServicerContext | None) -> ProtoMessage:  # noqa: N802  # NOSONAR
        _ = (request, context)
        pb2 = _require_engine_pb2()
        observation = self._env.observe()
        return pb2.ObserveResponse(observation=_observation_to_proto(observation))

    def StepMany(  # noqa: N802  # NOSONAR
        self, request: _RequestWithActions, context: grpc_ServicerContext | None
    ) -> ProtoMessage:
        pb2 = _require_engine_pb2()
        if len(request.actions) == 0:
            return pb2.StepManyResponse()

        rows: list[list[float]] = []
        for action in request.actions:
            limit_price = action.limit_price if action.has_limit_price else np_nan
            rows.append(
                [
                    float(action.side_code),
                    float(action.units),
                    float(action.order_type_code),
                    float(limit_price),
                ]
            )
        actions_matrix: NDArray[np_float64] = np_asarray(rows, dtype=np_float64)
        try:
            observations, rewards, dones, trace = self._env.step_many(
                actions_matrix,
                include_trace=bool(request.include_trace),
            )
        except ValueError as exc:
            if context is None:
                raise
            if grpc_StatusCode is None:
                raise RuntimeError(_GRPC_RUNTIME_UNAVAILABLE) from exc
            context.abort(grpc_StatusCode.INVALID_ARGUMENT, str(exc))
            raise AssertionError("context.abort should raise") from exc

        observation_rows: list[ProtoMessage] = []
        num_rows = int(rewards.shape[0])
        for i in range(num_rows):
            observation_rows.append(
                pb2.Observation(
                    market_window_handle=pb2.MarketWindowViewHandle(
                        start=int(observations["market_window_handle"][i, 0]),
                        end=int(observations["market_window_handle"][i, 1]),
                        t=int(observations["market_window_handle"][i, 2]),
                        current_price=float(observations["market_window_handle"][i, 3]),
                    ),
                    portfolio_vector=observations["portfolio_vector"][i].tolist(),
                    order_summary_vector=observations["order_summary_vector"][i].tolist(),
                    done=bool(dones[i]),
                )
            )
        trace_rows: list[ProtoMessage] = []
        for row in trace:
            trace_rows.append(
                pb2.StepTraceRow(
                    index=int(row.index),
                    side_code=int(row.side_code),
                    requested_units=float(row.requested_units),
                    order_type_code=int(row.order_type_code),
                    has_limit_price=bool(row.has_limit_price),
                    limit_price=0.0 if row.limit_price is None else float(row.limit_price),
                    fills=int(row.fills),
                    reward=float(row.reward),
                    done=bool(row.done),
                    t=int(row.t),
                    cash=float(row.cash),
                    position_units=float(row.position_units),
                    equity=float(row.equity),
                    leverage=float(row.leverage),
                    open_orders=int(row.open_orders),
                    market_price=float(row.market_price),
                )
            )
        return pb2.StepManyResponse(
            observations=observation_rows,
            rewards=rewards.tolist(),
            dones=dones.tolist(),
            trace=trace_rows,
        )


def create_server(
    *,
    host: str,
    port: int,
    cfg: GameConfig | None = None,
    data: MarketData | None = None,
    max_workers: int = 16,
) -> grpc_Server:
    config = cfg if cfg is not None else GameConfig.load(yaml_path=os_getenv("STOCK_SIM_CONFIG_YAML"))
    market_data = data if data is not None else _load_market_data()
    env = MarketEnv(data=market_data, cfg=config)
    env.reset(seed=config.seed)

    _require_grpc_module()
    if grpc_server is None:
        raise RuntimeError(_GRPC_RUNTIME_UNAVAILABLE)
    server = grpc_server(concurrentfutures_ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc = _require_engine_pb2_grpc()
    pb2_grpc.add_EngineServiceServicer_to_server(EngineGrpcService(env), server)
    server.add_insecure_port(f"{host}:{port}")
    return server


def main() -> None:
    parser = argparse_ArgumentParser(description="gRPC interface for stock simulator.")
    parser.add_argument("--host", default=os_getenv("GRPC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os_getenv("GRPC_PORT", "50051")))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--use-numba", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = GameConfig(use_numba=args.use_numba, seed=args.seed)
    server = create_server(
        host=args.host,
        port=args.port,
        cfg=cfg,
        max_workers=args.max_workers,
    )
    server.start()
    print(f"stock simulator gRPC listening on {args.host}:{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
