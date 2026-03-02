from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import cast

from numpy import asarray as np_asarray
from numpy import float64 as np_float64
from pytest import MonkeyPatch

from stock_simulator.config import GameConfig
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
    observations: list[_FakeObservation]
    rewards: list[float]
    dones: list[bool]


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


@dataclass
class _FakeStepManyRequest:
    actions: list[_FakeAction]


class _FakePb2:
    MarketWindowViewHandle = _FakeMarketWindowViewHandle
    Observation = _FakeObservation
    EnvState = _FakeEnvState
    PingResponse = _FakePingResponse
    ResetResponse = _FakeResetResponse
    ObserveResponse = _FakeObserveResponse
    StepManyResponse = _FakeStepManyResponse


class _FakeEnv:
    def __init__(self, *, raise_on_step: bool = False) -> None:
        self.raise_on_step = raise_on_step
        self.last_seed: int | None = None

    def reset(self, seed: int | None = None) -> EnvState:
        self.last_seed = seed
        return EnvState(
            t=0,
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

    def step_many(self, _matrix: object) -> tuple[dict[str, object], object, object]:
        if self.raise_on_step:
            raise ValueError("bad action payload")
        observations: dict[str, object] = {
            "market_window_handle": np_asarray([[1.0, 65.0, 5.0, 104.0]], dtype=np_float64),
            "portfolio_vector": np_asarray([[999.0, 1.0, 1103.0, 0.1]], dtype=np_float64),
            "order_summary_vector": np_asarray([[1.0, 1.0, 0.0]], dtype=np_float64),
        }
        rewards = np_asarray([5.5], dtype=np_float64)
        dones = np_asarray([False], dtype=np_float64)
        return observations, cast(object, rewards), cast(object, dones)


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

    def add_EngineServiceServicer_to_server(  # noqa: N802
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


def test_build_synthetic_market_data_and_loader(monkeypatch: MonkeyPatch) -> None:
    synthetic = grpc_server_mod._build_synthetic_market_data(bars=32, seed=9)
    assert synthetic.n == 32

    marker = object()
    monkeypatch.setenv("MARKET_DATA_CSV", "/tmp/test.csv")
    monkeypatch.setattr(grpc_server_mod.MarketData, "from_csv", lambda _path: marker)
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
                _FakeResetRequest(has_seed=True, seed=77),
            ),
            _FakeContext(),
        ),
    )
    assert reset.state.t == 0
    assert env.last_seed == 77

    observe = cast(_FakeObserveResponse, service.Observe(object(), _FakeContext()))
    assert observe.observation.market_window_handle.t == 4

    request = _FakeStepManyRequest(
        actions=[_FakeAction(side_code=1, units=1.0, order_type_code=0, has_limit_price=False, limit_price=0.0)]
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
