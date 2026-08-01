from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from pytest import MonkeyPatch

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.grpc import server as grpc_server_mod
from stock_simulator.types import EnvState, MarketWindowViewHandle, Observation


@dataclass
class _FakeMarketWindowViewHandle:
    start: int
    end: int
    t: int
    current_price: float


@dataclass
class _FakeObservation:
    market_window_handle: _FakeMarketWindowViewHandle
    portfolio_vector: list[float]
    order_summary_vector: list[float]
    done: bool


@dataclass
class _FakeEnvState:
    t: int
    cash: float
    units: float
    equity: float
    leverage: float
    open_orders: int
    done: bool


@dataclass
class _FakePingResponse:
    status: str


@dataclass
class _FakeResetResponse:
    state: _FakeEnvState


@dataclass
class _FakeObserveResponse:
    observation: _FakeObservation


@dataclass
class _FakeStepManyResponse:
    observations: list[_FakeObservation] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    trace: list[object] = field(default_factory=list)


@dataclass
class _FakeStepTraceRow:
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
    has_filled_order: bool = False
    filled_order_slot: int | None = None
    exec_price: float | None = None


@dataclass
class _FakeAction:
    side_code: int
    units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float


@dataclass
class _FakeResetRequest:
    has_seed: bool
    seed: int
    has_start_t: bool = False
    start_t: int = 0


@dataclass
class _FakeStepManyRequest:
    actions: list[_FakeAction]
    include_trace: bool = False


class _FakePb2:
    MarketWindowViewHandle = _FakeMarketWindowViewHandle  # NOSONAR - protobuf-generated API shape
    Observation = _FakeObservation  # NOSONAR - protobuf-generated API shape
    EnvState = _FakeEnvState  # NOSONAR - protobuf-generated API shape
    PingResponse = _FakePingResponse  # NOSONAR - protobuf-generated API shape
    ResetResponse = _FakeResetResponse  # NOSONAR - protobuf-generated API shape
    ObserveResponse = _FakeObserveResponse  # NOSONAR - protobuf-generated API shape
    StepManyResponse = _FakeStepManyResponse  # NOSONAR - protobuf-generated API shape
    StepTraceRow = _FakeStepTraceRow  # NOSONAR - protobuf-generated API shape


class _FakeEnv:
    def __init__(self, *, raise_on_step: bool = False, raise_on_reset: bool = False) -> None:
        self.raise_on_step = raise_on_step
        self.raise_on_reset = raise_on_reset
        self.last_seed: int | None = None
        self.last_start_t: int = 0

    def reset(self, seed: int | None = None, start_t: int = 0) -> EnvState:
        self.last_seed = seed
        self.last_start_t = start_t
        if self.raise_on_reset:
            raise ValueError(f"start_t must satisfy 0 <= start_t < 10 (market has 10 bars); got {start_t}")
        return EnvState(
            t=start_t,
            cash=1000.0,
            units=0.0,
            equity=1000.0,
            leverage=0.0,
            open_orders=0,
            done=False,
        )

    def observe(self) -> Observation:
        return Observation(
            market=MarketWindowViewHandle(start=1, end=65, t=4, current_price=103.5),
            portfolio_vector=np_asarray([1000.0, 0.0, 1000.0, 0.0], dtype=np_float64),
            order_summary_vector=np_asarray([0.0, 0.0, 0.0], dtype=np_float64),
            done=False,
        )

    def step_many(
        self,
        _matrix: object,
        *,
        include_trace: bool = False,
    ) -> tuple[dict[str, object], object, object, tuple[_FakeStepTraceRow, ...]]:
        if self.raise_on_step:
            raise ValueError("bad action payload")
        observations: dict[str, object] = {
            "market_window_handle": np_asarray([[1.0, 65.0, 5.0, 104.0]], dtype=np_float64),
            "portfolio_vector": np_asarray([[999.0, 1.0, 1103.0, 0.1]], dtype=np_float64),
            "order_summary_vector": np_asarray([[1.0, 1.0, 0.0]], dtype=np_float64),
        }
        rewards = np_asarray([5.5], dtype=np_float64)
        dones = np_asarray([False], dtype=np_bool_)
        trace = (
            (
                _FakeStepTraceRow(
                    index=0,
                    side_code=1,
                    requested_units=1.0,
                    order_type_code=0,
                    has_limit_price=False,
                    limit_price=0.0,
                    fills=1,
                    reward=5.5,
                    done=False,
                    t=5,
                    cash=999.0,
                    position_units=1.0,
                    equity=1103.0,
                    leverage=0.1,
                    open_orders=1,
                    market_price=104.0,
                    has_filled_order=True,
                    filled_order_slot=0,
                    exec_price=104.0,
                ),
            )
            if include_trace
            else ()
        )
        return observations, cast(object, rewards), cast(object, dones), trace


class _FakeContext:
    def abort(self, _status: object, detail: str) -> None:
        raise ValueError(detail)


@dataclass
class _FakeGrpcServer:
    address: str = ""
    started: bool = False
    waited: bool = False

    def add_insecure_port(self, address: str) -> None:
        self.address = address

    def start(self) -> None:
        self.started = True

    def wait_for_termination(self) -> None:
        self.waited = True


class _FakePb2Grpc:
    def __init__(self) -> None:
        self.add_calls = 0

    def add_EngineServiceServicer_to_server(  # noqa: N802  # NOSONAR - gRPC generated name contract
        self,
        _servicer: grpc_server_mod.EngineGrpcService,
        _server: _FakeGrpcServer,
    ) -> None:
        self.add_calls += 1


def test_require_helpers_raise_when_stubs_unavailable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(grpc_server_mod, "engine_pb2", None)
    monkeypatch.setattr(grpc_server_mod, "engine_pb2_grpc", None)

    try:
        grpc_server_mod._require_engine_pb2()
        raise AssertionError("expected _require_engine_pb2 to fail")
    except RuntimeError as exc:
        assert "protobuf messages are unavailable" in str(exc)

    try:
        grpc_server_mod._require_engine_pb2_grpc()
        raise AssertionError("expected _require_engine_pb2_grpc to fail")
    except RuntimeError as exc:
        assert "gRPC stubs are unavailable" in str(exc)


def test_build_synthetic_market_data_and_loader(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    synthetic = grpc_server_mod._build_synthetic_market_data(bars=32, seed=9)
    assert synthetic.n == 32

    marker = object()
    market_csv_path = tmp_path / "market.csv"
    monkeypatch.setenv("MARKET_DATA_CSV", str(market_csv_path))
    monkeypatch.setattr(MarketData, "from_csv", lambda _path: marker)
    loaded = grpc_server_mod._load_market_data()
    assert loaded is marker

    monkeypatch.delenv("MARKET_DATA_CSV", raising=False)
    monkeypatch.setenv("IPC_SYNTHETIC_BARS", "16")
    monkeypatch.setenv("IPC_SYNTHETIC_SEED", "42")
    generated = grpc_server_mod._load_market_data()
    assert generated.n == 16


def test_engine_grpc_service_methods(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(grpc_server_mod, "engine_pb2", _FakePb2())
    env = _FakeEnv()
    service = grpc_server_mod.EngineGrpcService(cast(MarketEnv, env))

    ping = cast(_FakePingResponse, service.Ping(object(), _FakeContext()))
    assert ping.status == "pong"

    reset = cast(
        _FakeResetResponse,
        service.Reset(
            cast(
                grpc_server_mod._RequestWithSeed,
                _FakeResetRequest(has_seed=True, seed=77, has_start_t=True, start_t=150),
            ),
            _FakeContext(),
        ),
    )
    assert reset.state.t == 150
    assert env.last_seed == 77
    assert env.last_start_t == 150

    observe = cast(_FakeObserveResponse, service.Observe(object(), _FakeContext()))
    assert observe.observation.market_window_handle.t == 4

    request = _FakeStepManyRequest(
        actions=[_FakeAction(side_code=1, units=1.0, order_type_code=0, has_limit_price=False, limit_price=0.0)],
        include_trace=True,
    )
    step_many = cast(
        _FakeStepManyResponse,
        service.StepMany(
            cast(grpc_server_mod._RequestWithActions, request),
            _FakeContext(),
        ),
    )
    assert step_many.rewards == [5.5]
    assert step_many.dones == [False]
    assert len(step_many.trace) == 1
    trace_row = cast(_FakeStepTraceRow, step_many.trace[0])
    assert trace_row.has_filled_order is True
    assert trace_row.filled_order_slot == 0
    assert trace_row.exec_price == 104.0


def test_engine_grpc_service_reset_omitted_start_t_defaults_to_zero(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(grpc_server_mod, "engine_pb2", _FakePb2())
    env = _FakeEnv()
    service = grpc_server_mod.EngineGrpcService(cast(MarketEnv, env))

    reset = cast(
        _FakeResetResponse,
        service.Reset(
            cast(grpc_server_mod._RequestWithSeed, _FakeResetRequest(has_seed=True, seed=5)),
            _FakeContext(),
        ),
    )
    assert reset.state.t == 0
    assert env.last_start_t == 0


def test_engine_grpc_service_reset_error_path_aborts_context(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(grpc_server_mod, "engine_pb2", _FakePb2())
    env = _FakeEnv(raise_on_reset=True)
    service = grpc_server_mod.EngineGrpcService(cast(MarketEnv, env))
    try:
        service.Reset(
            cast(
                grpc_server_mod._RequestWithSeed,
                _FakeResetRequest(has_seed=False, seed=0, has_start_t=True, start_t=999),
            ),
            _FakeContext(),
        )
        raise AssertionError("expected Reset to raise")
    except ValueError as exc:
        assert "start_t" in str(exc)


def test_engine_grpc_service_reset_error_path_reraises_without_context(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(grpc_server_mod, "engine_pb2", _FakePb2())
    env = _FakeEnv(raise_on_reset=True)
    service = grpc_server_mod.EngineGrpcService(cast(MarketEnv, env))
    try:
        service.Reset(
            cast(grpc_server_mod._RequestWithSeed, _FakeResetRequest(has_seed=False, seed=0)),
            None,
        )
        raise AssertionError("expected Reset to raise")
    except ValueError as exc:
        assert "start_t" in str(exc)


def test_engine_grpc_service_step_many_error_path(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(grpc_server_mod, "engine_pb2", _FakePb2())
    env = _FakeEnv(raise_on_step=True)
    service = grpc_server_mod.EngineGrpcService(cast(MarketEnv, env))
    try:
        request = _FakeStepManyRequest(
            actions=[_FakeAction(side_code=1, units=1.0, order_type_code=9, has_limit_price=False, limit_price=0.0)]
        )
        service.StepMany(
            cast(grpc_server_mod._RequestWithActions, request),
            _FakeContext(),
        )
        raise AssertionError("expected StepMany to raise")
    except ValueError as exc:
        assert "bad action payload" in str(exc)


def test_create_server_wires_service_and_port(monkeypatch: MonkeyPatch) -> None:
    fake_server = _FakeGrpcServer()
    fake_pb2_grpc = _FakePb2Grpc()
    env = _FakeEnv()

    monkeypatch.setattr(grpc_server_mod, "engine_pb2_grpc", fake_pb2_grpc)
    monkeypatch.setattr(grpc_server_mod, "grpc_server", lambda _executor: fake_server)
    monkeypatch.setattr(grpc_server_mod, "_load_market_data", lambda: object())
    monkeypatch.setattr(grpc_server_mod, "MarketEnv", lambda data, cfg: env)

    server = grpc_server_mod.create_server(host="127.0.0.1", port=50051, cfg=GameConfig(seed=55, use_numba=False))
    assert server is fake_server
    assert fake_server.address == "127.0.0.1:50051"
    assert fake_pb2_grpc.add_calls == 1
    assert env.last_seed == 55


def test_main_starts_and_waits(monkeypatch: MonkeyPatch) -> None:
    created: list[_FakeGrpcServer] = []

    def _fake_create_server(**_kwargs: object) -> _FakeGrpcServer:
        server = _FakeGrpcServer()
        created.append(server)
        return server

    monkeypatch.setattr(grpc_server_mod, "create_server", _fake_create_server)
    monkeypatch.setattr(sys, "argv", ["stock-simulator-grpc"])
    monkeypatch.setenv("GRPC_HOST", "127.0.0.2")
    monkeypatch.setenv("GRPC_PORT", "50052")

    grpc_server_mod.main()

    assert len(created) == 1
    assert created[0].started is True
    assert created[0].waited is True
