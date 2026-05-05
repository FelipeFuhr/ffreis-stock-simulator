from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from numpy import testing as np_testing
from pytest import MonkeyPatch
from pytest import approx as pytest_approx

from stock_simulator.grpc import client as client_mod


@dataclass
class _MarketHandle:
    start: int
    end: int
    t: int
    current_price: float


@dataclass
class _State:
    t: int
    cash: float
    units: float
    equity: float
    leverage: float
    open_orders: int
    done: bool


@dataclass
class _Observation:
    market_window_handle: _MarketHandle
    portfolio_vector: list[float]
    order_summary_vector: list[float]
    done: bool


@dataclass
class _PingReply:
    status: str


@dataclass
class _ResetReply:
    state: _State


@dataclass
class _ObserveReply:
    observation: _Observation


@dataclass
class _StepManyReply:
    observations: list[_Observation]
    rewards: list[float]
    dones: list[bool]
    trace: list[object]


@dataclass
class _StepTraceRow:
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


@dataclass
class _EncodedAction:
    side_code: int
    units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float


@dataclass
class _ResetRequest:
    has_seed: bool
    seed: int


@dataclass
class _StepManyRequest:
    actions: list[_EncodedAction]
    include_trace: bool


class _Pb2:
    class PingRequest:
        def __init__(self) -> None:
            # Empty by design: this mirrors generated protobuf request construction.
            pass

    class ObserveRequest:
        def __init__(self) -> None:
            # Empty by design: this mirrors generated protobuf request construction.
            pass

    ResetRequest = _ResetRequest  # NOSONAR - protobuf-generated API shape
    EncodedAction = _EncodedAction  # NOSONAR - protobuf-generated API shape
    StepManyRequest = _StepManyRequest  # NOSONAR - protobuf-generated API shape


class _FakeStub:
    def Ping(self, _request: object) -> _PingReply:  # noqa: N802  # NOSONAR - gRPC method name
        return _PingReply(status="pong")

    def Reset(self, request: _ResetRequest) -> _ResetReply:  # noqa: N802  # NOSONAR - gRPC method name
        if request.has_seed:
            seed_cash = float(request.seed)
        else:
            seed_cash = 0.0
        return _ResetReply(
            state=_State(
                t=0,
                cash=seed_cash,
                units=1.0,
                equity=seed_cash + 1.0,
                leverage=0.1,
                open_orders=2,
                done=False,
            )
        )

    def Observe(self, _request: object) -> _ObserveReply:  # noqa: N802  # NOSONAR - gRPC method name
        return _ObserveReply(
            observation=_Observation(
                market_window_handle=_MarketHandle(start=0, end=8, t=2, current_price=101.0),
                portfolio_vector=[100.0, 1.0, 201.0, 0.2],
                order_summary_vector=[1.0, 1.0, 0.0],
                done=False,
            )
        )

    def StepMany(self, request: _StepManyRequest) -> _StepManyReply:  # noqa: N802  # NOSONAR - gRPC method name
        trace_rows: list[object] = []
        if request.include_trace:
            trace_rows.append(
                _StepTraceRow(
                    index=0,
                    side_code=1,
                    requested_units=1.0,
                    order_type_code=0,
                    has_limit_price=False,
                    limit_price=0.0,
                    fills=1,
                    reward=1.25,
                    done=False,
                    t=3,
                    cash=99.0,
                    position_units=2.0,
                    equity=303.0,
                    leverage=0.3,
                    open_orders=2,
                    market_price=102.0,
                )
            )
        return _StepManyReply(
            observations=[
                _Observation(
                    market_window_handle=_MarketHandle(start=1, end=9, t=3, current_price=102.0),
                    portfolio_vector=[99.0, 2.0, 303.0, 0.3],
                    order_summary_vector=[2.0, 1.0, 1.0],
                    done=False,
                )
            ],
            rewards=[1.25],
            dones=[False],
            trace=trace_rows,
        )


@dataclass
class _FakeChannel:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


class _Pb2Grpc:
    def __init__(self, stub: _FakeStub) -> None:
        self._stub = stub

    def EngineServiceStub(self, _channel: _FakeChannel) -> _FakeStub:  # noqa: N802  # NOSONAR
        return self._stub


def test_require_helpers_raise_when_stubs_unavailable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod, "engine_pb2", None)
    monkeypatch.setattr(client_mod, "engine_pb2_grpc", None)

    try:
        client_mod._require_engine_pb2()
        raise AssertionError("expected _require_engine_pb2 to fail")
    except RuntimeError as exc:
        assert "protobuf messages are unavailable" in str(exc)

    try:
        client_mod._require_engine_pb2_grpc()
        raise AssertionError("expected _require_engine_pb2_grpc to fail")
    except RuntimeError as exc:
        assert "gRPC stubs are unavailable" in str(exc)


def test_client_methods_roundtrip_with_fake_stub(monkeypatch: MonkeyPatch) -> None:
    channel = _FakeChannel()
    stub = _FakeStub()

    monkeypatch.setattr(client_mod, "engine_pb2", _Pb2())
    monkeypatch.setattr(client_mod, "engine_pb2_grpc", _Pb2Grpc(stub))
    monkeypatch.setattr(client_mod, "grpc_insecure_channel", lambda _target: channel)

    client = client_mod.EngineGrpcClient(target="127.0.0.1:50051")

    assert client.ping() == "pong"

    reset_state = client.reset(seed=123)
    assert reset_state["cash"] == pytest_approx(123.0)
    assert reset_state["open_orders"] == 2

    observed = client.observe()
    market = observed["market_window_handle"]
    assert isinstance(market, dict)
    assert market["t"] == 2
    assert observed["done"] is False

    actions = np_asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 2.0, 1.0, 99.0],
        ],
        dtype=np_float64,
    )
    obs_rows, rewards, dones = client.step_many(actions)
    assert len(obs_rows) == 1
    market = cast(dict[str, int | float], obs_rows[0]["market_window_handle"])
    assert market["current_price"] == pytest_approx(102.0)
    np_testing.assert_allclose(rewards, np_asarray([1.25], dtype=np_float64))
    np_testing.assert_array_equal(dones, np_asarray([False], dtype=np_bool_))

    _, _, _, trace_rows = client.step_many_with_trace(actions)
    assert len(trace_rows) == 1
    assert trace_rows[0]["fills"] == 1
    assert trace_rows[0]["requested_units"] == pytest_approx(1.0)

    client.close()
    assert channel.closed is True
