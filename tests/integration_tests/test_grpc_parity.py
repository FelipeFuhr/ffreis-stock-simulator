from __future__ import annotations

from collections.abc import Callable
from importlib import import_module as importlib_import_module
from os import getenv as os_getenv
from typing import Protocol, cast

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy import array as np_array
from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from numpy import isnan as np_isnan
from numpy import nan as np_nan
from numpy import testing as np_testing
from numpy.typing import NDArray
from pytest import mark as pytest_mark
from pytest import skip as pytest_skip

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv
from stock_simulator.server import create_app
from stock_simulator.types import (
    Action,
    EnvStateModel,
    MarketWindowViewHandleModel,
    ObservationModel,
    StepTraceRowModel,
)

try:
    from stock_simulator.grpc.server import EngineGrpcService

    _engine_pb2 = importlib_import_module("stocksim_grpc.engine_pb2")
except ImportError as exc:
    pytest_skip(f"grpc parity dependencies unavailable: {exc}", allow_module_level=True)


class _MethodLike(Protocol):
    name: str


class _ServiceDescriptorLike(Protocol):
    methods: list[_MethodLike]


class _MessageDescriptorLike(Protocol):
    fields_by_name: dict[str, int]


class _RootDescriptorLike(Protocol):
    services_by_name: dict[str, _ServiceDescriptorLike]


class _MessageTypeLike(Protocol):
    DESCRIPTOR: _MessageDescriptorLike


class _EncodedActionLike(Protocol):
    side_code: int
    units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float


class _StepManyRequestLike(Protocol):
    actions: list[_EncodedActionLike]
    include_trace: bool


class _ResetRequestLike(_MessageTypeLike, Protocol):
    has_seed: bool
    seed: int
    has_start_t: bool
    start_t: int


class _ObserveRequestLike(_MessageTypeLike, Protocol):
    pass


class _FactoryEncodedAction(Protocol):
    DESCRIPTOR: _MessageDescriptorLike

    def __call__(
        self,
        *,
        side_code: int,
        units: float,
        order_type_code: int,
        has_limit_price: bool,
        limit_price: float,
    ) -> _EncodedActionLike: ...


class _FactoryStepManyRequest(Protocol):
    def __call__(self, *, actions: list[_EncodedActionLike], include_trace: bool = False) -> _StepManyRequestLike: ...


class _FactoryResetRequest(Protocol):
    def __call__(
        self,
        *,
        has_seed: bool,
        seed: int,
        has_start_t: bool = False,
        start_t: int = 0,
    ) -> _ResetRequestLike: ...


class _FactoryObserveRequest(Protocol):
    def __call__(self) -> _ObserveRequestLike: ...


class _StepManyRequestType(_MessageTypeLike, _FactoryStepManyRequest, Protocol):
    pass


class _Pb2Like(Protocol):
    DESCRIPTOR: _RootDescriptorLike
    EnvState: _MessageTypeLike
    Observation: _MessageTypeLike
    MarketWindowViewHandle: _MessageTypeLike
    EncodedAction: _FactoryEncodedAction
    StepManyRequest: _StepManyRequestType
    ResetRequest: _FactoryResetRequest
    ObserveRequest: _FactoryObserveRequest


class _ReplyMarketHandleLike(Protocol):
    start: float
    end: float
    t: float
    current_price: float


class _ReplyObservationLike(Protocol):
    market_window_handle: _ReplyMarketHandleLike
    portfolio_vector: list[float]
    order_summary_vector: list[float]
    done: bool


class _ReplyEnvStateLike(Protocol):
    t: int
    cash: float
    units: float
    equity: float
    open_orders: int
    done: bool


class _ResetReplyLike(Protocol):
    state: _ReplyEnvStateLike


class _ObserveReplyLike(Protocol):
    observation: _ReplyObservationLike


class _StepManyReplyLike(Protocol):
    observations: list[_ReplyObservationLike]
    rewards: list[float]
    dones: list[bool]
    trace: list[_ReplyTraceRowLike]


class _ReplyTraceRowLike(Protocol):
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


class _DirectTraceRowLike(Protocol):
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


class _EngineGrpcServiceLike(Protocol):
    def Reset(self, request: _ResetRequestLike, context: None) -> _ResetReplyLike: ...  # noqa: N802

    def Observe(self, request: _ObserveRequestLike, context: None) -> _ObserveReplyLike: ...  # noqa: N802

    def StepMany(self, request: _StepManyRequestLike, context: None) -> _StepManyReplyLike: ...  # noqa: N802


engine_pb2 = cast(_Pb2Like, _engine_pb2)

_HTTP_TO_GRPC_SURFACE_MAP: dict[str, str] = {
    "/healthz": "Ping",
    "/v1/reset": "Reset",
    "/v1/observe": "Observe",
    "/v1/step_many": "StepMany",
}
# /v1/market_window is intentionally HTTP-only for now: the equivalent gRPC RPC is
# a deferred follow-up (see PR body). Declared here so the surface-parity guard
# records the intended difference rather than flagging accidental drift.
_UNMAPPED_HTTP_PATHS: set[str] = {"/readyz", "/v1/market_window"}
_UNMAPPED_GRPC_METHODS: set[str] = set()
_HYPOTHESIS_MAX_EXAMPLES = int(os_getenv("HYPOTHESIS_MAX_EXAMPLES", "25"))


def _build_step_many_request(actions: NDArray[np_float64], *, include_trace: bool = False) -> _StepManyRequestLike:
    encoded: list[_EncodedActionLike] = []
    for row in actions:
        encoded.append(
            engine_pb2.EncodedAction(
                side_code=int(row[0]),
                units=float(row[1]),
                order_type_code=int(row[2]),
                has_limit_price=not np_isnan(row[3]),
                limit_price=0.0 if np_isnan(row[3]) else float(row[3]),
            )
        )
    return engine_pb2.StepManyRequest(actions=encoded, include_trace=include_trace)


def _grpc_reply_to_arrays(
    reply: _StepManyReplyLike,
) -> tuple[
    NDArray[np_float64],
    NDArray[np_float64],
    NDArray[np_float64],
    NDArray[np_float64],
    NDArray[np_bool_],
    NDArray[np_float64],
]:
    market = np_asarray(
        [
            [
                float(o.market_window_handle.start),
                float(o.market_window_handle.end),
                float(o.market_window_handle.t),
                float(o.market_window_handle.current_price),
            ]
            for o in reply.observations
        ],
        dtype=np_float64,
    )
    portfolio = np_asarray([o.portfolio_vector for o in reply.observations], dtype=np_float64)
    orders = np_asarray([o.order_summary_vector for o in reply.observations], dtype=np_float64)
    rewards = np_asarray(reply.rewards, dtype=np_float64)
    dones = np_asarray(reply.dones, dtype=np_bool_)
    trace = np_asarray(
        [
            [
                float(row.index),
                float(row.side_code),
                float(row.requested_units),
                float(row.order_type_code),
                1.0 if bool(row.has_limit_price) else 0.0,
                float(row.limit_price) if bool(row.has_limit_price) else np_nan,
                float(row.fills),
                float(row.reward),
                1.0 if bool(row.done) else 0.0,
                float(row.t),
                float(row.cash),
                float(row.position_units),
                float(row.equity),
                float(row.leverage),
                float(row.open_orders),
                float(row.market_price),
            ]
            for row in reply.trace
        ],
        dtype=np_float64,
    )
    return market, portfolio, orders, rewards, dones, trace


def _trace_rows_to_array(rows: tuple[_DirectTraceRowLike, ...]) -> NDArray[np_float64]:
    return np_asarray(
        [
            [
                float(row.index),
                float(row.side_code),
                float(row.requested_units),
                float(row.order_type_code),
                1.0 if bool(row.has_limit_price) else 0.0,
                float(row.limit_price) if row.has_limit_price and row.limit_price is not None else np_nan,
                float(row.fills),
                float(row.reward),
                1.0 if bool(row.done) else 0.0,
                float(row.t),
                float(row.cash),
                float(row.position_units),
                float(row.equity),
                float(row.leverage),
                float(row.open_orders),
                float(row.market_price),
            ]
            for row in rows
        ],
        dtype=np_float64,
    )


def _contract_fields(message: _MessageTypeLike) -> set[str]:
    return set(message.DESCRIPTOR.fields_by_name.keys())


def _grpc_method_names() -> set[str]:
    service = engine_pb2.DESCRIPTOR.services_by_name["EngineService"]
    return {method.name for method in service.methods}


def _http_route_paths() -> set[str]:
    app = create_app()
    return set(app.openapi().get("paths", {}).keys())


def test_surface_parity_map_is_explicit_for_intended_differences() -> None:
    """Guard against accidental HTTP/gRPC surface drift."""
    discovered_http = _http_route_paths()
    discovered_grpc = _grpc_method_names()

    mapped_http = set(_HTTP_TO_GRPC_SURFACE_MAP.keys())
    mapped_grpc = set(_HTTP_TO_GRPC_SURFACE_MAP.values())

    assert discovered_http == mapped_http | _UNMAPPED_HTTP_PATHS
    assert discovered_grpc == mapped_grpc | _UNMAPPED_GRPC_METHODS


def test_contract_parity_for_env_state_and_observation_models() -> None:
    grpc_env_fields = _contract_fields(engine_pb2.EnvState)
    grpc_observation_fields = _contract_fields(engine_pb2.Observation)
    grpc_market_fields = _contract_fields(engine_pb2.MarketWindowViewHandle)

    pydantic_env_fields = set(EnvStateModel.model_fields.keys())
    pydantic_observation_fields = set(ObservationModel.model_fields.keys())
    market_handle_annotation = cast(
        type[MarketWindowViewHandleModel],
        ObservationModel.model_fields["market_window_handle"].annotation,
    )
    pydantic_market_fields = set(market_handle_annotation.model_fields.keys())

    assert pydantic_env_fields == grpc_env_fields
    assert pydantic_observation_fields == grpc_observation_fields
    assert pydantic_market_fields == grpc_market_fields


def test_contract_parity_for_step_many_request_payload() -> None:
    request_fields = _contract_fields(engine_pb2.StepManyRequest)
    response_fields = _contract_fields(engine_pb2.StepManyResponse)
    action_fields = _contract_fields(engine_pb2.EncodedAction)
    trace_fields = _contract_fields(engine_pb2.StepTraceRow)

    assert request_fields == {"actions", "include_trace"}
    assert response_fields == {"observations", "rewards", "dones", "trace"}
    assert action_fields == {
        "side_code",
        "units",
        "order_type_code",
        "has_limit_price",
        "limit_price",
    }
    assert trace_fields == set(StepTraceRowModel.model_fields.keys())


def test_grpc_service_matches_env_behavior(
    market_data_factory: Callable[..., MarketData],
    encode_actions: Callable[[list[Action]], NDArray[np_float64]],
) -> None:
    data = market_data_factory(n=256, slope=0.15, spread=0.2, volume=15_000.0)
    cfg = GameConfig(seed=4242, use_numba=False)

    env_direct = MarketEnv(data=data, cfg=cfg)
    env_grpc = MarketEnv(data=data, cfg=cfg)
    grpc_service = cast(_EngineGrpcServiceLike, EngineGrpcService(env_grpc))

    direct_state = env_direct.reset(seed=4242)
    grpc_state = grpc_service.Reset(
        engine_pb2.ResetRequest(has_seed=True, seed=4242),
        None,
    ).state
    assert direct_state.t == grpc_state.t
    assert direct_state.cash == grpc_state.cash
    assert direct_state.units == grpc_state.units
    assert direct_state.equity == grpc_state.equity
    assert direct_state.open_orders == grpc_state.open_orders
    assert direct_state.done == grpc_state.done

    direct_obs = env_direct.observe()
    grpc_obs = grpc_service.Observe(engine_pb2.ObserveRequest(), None).observation
    grpc_market_vec = np_array(
        [
            grpc_obs.market_window_handle.start,
            grpc_obs.market_window_handle.end,
            grpc_obs.market_window_handle.t,
            grpc_obs.market_window_handle.current_price,
        ],
        dtype=np_float64,
    )
    np_testing.assert_allclose(direct_obs.market.to_numpy(), grpc_market_vec)
    np_testing.assert_allclose(
        direct_obs.portfolio_vector,
        np_asarray(grpc_obs.portfolio_vector, dtype=np_float64),
    )
    np_testing.assert_allclose(
        direct_obs.order_summary_vector,
        np_asarray(grpc_obs.order_summary_vector, dtype=np_float64),
    )
    assert direct_obs.done == grpc_obs.done

    actions = encode_actions(
        [
            Action(side="hold"),
            Action(side="buy", units=2.5, order_type="market"),
            Action(side="sell", units=1.0, order_type="limit", limit_price=105.0),
            Action(side="buy", units=1.5, order_type="limit", limit_price=99.0),
        ]
    )

    direct_obs_stack, direct_rewards, direct_dones, direct_trace = env_direct.step_many(actions)
    grpc_reply = grpc_service.StepMany(_build_step_many_request(actions), None)

    grpc_market, grpc_portfolio, grpc_orders, grpc_rewards, grpc_dones, grpc_trace = _grpc_reply_to_arrays(grpc_reply)

    np_testing.assert_allclose(direct_obs_stack["market_window_handle"], grpc_market)
    np_testing.assert_allclose(direct_obs_stack["portfolio_vector"], grpc_portfolio)
    np_testing.assert_allclose(direct_obs_stack["order_summary_vector"], grpc_orders)
    np_testing.assert_allclose(direct_rewards, grpc_rewards)
    np_testing.assert_array_equal(direct_dones, grpc_dones)
    assert len(direct_trace) == 0
    assert grpc_trace.shape == (0,)

    _ = env_direct.reset(seed=4242)
    _ = grpc_service.Reset(
        engine_pb2.ResetRequest(has_seed=True, seed=4242),
        None,
    )
    direct_obs_stack_trace, direct_rewards_trace, direct_dones_trace, direct_trace_rows = env_direct.step_many(
        actions,
        include_trace=True,
    )
    grpc_reply_trace = grpc_service.StepMany(
        _build_step_many_request(actions, include_trace=True),
        None,
    )
    (
        grpc_market_trace,
        grpc_portfolio_trace,
        grpc_orders_trace,
        grpc_rewards_trace,
        grpc_dones_trace,
        grpc_trace_rows,
    ) = _grpc_reply_to_arrays(grpc_reply_trace)

    np_testing.assert_allclose(direct_obs_stack_trace["market_window_handle"], grpc_market_trace)
    np_testing.assert_allclose(direct_obs_stack_trace["portfolio_vector"], grpc_portfolio_trace)
    np_testing.assert_allclose(direct_obs_stack_trace["order_summary_vector"], grpc_orders_trace)
    np_testing.assert_allclose(direct_rewards_trace, grpc_rewards_trace)
    np_testing.assert_array_equal(direct_dones_trace, grpc_dones_trace)
    np_testing.assert_allclose(_trace_rows_to_array(direct_trace_rows), grpc_trace_rows, rtol=1e-7, atol=1e-9)


@pytest_mark.property
@settings(
    max_examples=_HYPOTHESIS_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    actions=st.lists(
        st.one_of(
            st.just(Action(side="hold")),
            st.builds(
                Action,
                side=st.sampled_from(["buy", "sell"]),
                units=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
                order_type=st.just("market"),
                limit_price=st.none(),
            ),
            st.builds(
                Action,
                side=st.sampled_from(["buy", "sell"]),
                units=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
                order_type=st.just("limit"),
                limit_price=st.floats(min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False),
            ),
        ),
        min_size=1,
        max_size=30,
    )
)
def test_property_based_behavior_parity(
    actions: list[Action],
    market_data_factory: Callable[..., MarketData],
    encode_actions: Callable[[list[Action]], NDArray[np_float64]],
) -> None:
    data = market_data_factory(n=512, slope=0.07, spread=0.25, volume=18_000.0)
    cfg = GameConfig(seed=3030, use_numba=False)

    env_direct = MarketEnv(data=data, cfg=cfg)
    env_grpc = MarketEnv(data=data, cfg=cfg)
    grpc_service = cast(_EngineGrpcServiceLike, EngineGrpcService(env_grpc))

    env_direct.reset(seed=3030)
    grpc_service.Reset(engine_pb2.ResetRequest(has_seed=True, seed=3030), None)

    encoded = encode_actions(actions)
    direct_obs_stack, direct_rewards, direct_dones, direct_trace = env_direct.step_many(encoded, include_trace=True)
    grpc_reply = grpc_service.StepMany(_build_step_many_request(encoded, include_trace=True), None)
    grpc_market, grpc_portfolio, grpc_orders, grpc_rewards, grpc_dones, grpc_trace = _grpc_reply_to_arrays(grpc_reply)

    np_testing.assert_allclose(direct_obs_stack["market_window_handle"], grpc_market, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(direct_obs_stack["portfolio_vector"], grpc_portfolio, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(direct_obs_stack["order_summary_vector"], grpc_orders, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(direct_rewards, grpc_rewards, rtol=1e-7, atol=1e-9)
    np_testing.assert_array_equal(direct_dones, grpc_dones)
    np_testing.assert_allclose(_trace_rows_to_array(direct_trace), grpc_trace, rtol=1e-7, atol=1e-9)


def test_error_parity_for_invalid_order_type_code(
    market_data_factory: Callable[..., MarketData],
) -> None:
    data = market_data_factory(n=64, slope=0.1, spread=0.2, volume=10_000.0)
    cfg = GameConfig(seed=77, use_numba=False)
    env_direct = MarketEnv(data=data, cfg=cfg)
    env_grpc = MarketEnv(data=data, cfg=cfg)
    grpc_service = cast(_EngineGrpcServiceLike, EngineGrpcService(env_grpc))

    env_direct.reset(seed=77)
    grpc_service.Reset(engine_pb2.ResetRequest(has_seed=True, seed=77), None)

    invalid_actions = np_asarray([[1.0, 1.0, 2.0, np_nan]], dtype=np_float64)
    request = _build_step_many_request(invalid_actions)

    try:
        env_direct.step_many(invalid_actions)
        raise AssertionError("direct env did not raise")
    except ValueError as exc_direct:
        direct_msg = str(exc_direct)

    try:
        grpc_service.StepMany(request, None)
        raise AssertionError("grpc service did not raise")
    except ValueError as exc_grpc:
        grpc_msg = str(exc_grpc)

    assert "invalid order_type code" in direct_msg
    assert "invalid order_type code" in grpc_msg


def test_grpc_reset_start_t_matches_env_behavior(
    market_data_factory: Callable[..., MarketData],
) -> None:
    """gRPC ResetRequest.has_start_t/start_t must reset to the same bar as the
    direct engine call — F1's start_t fix is not HTTP-only.
    """
    data = market_data_factory(n=256, slope=0.15, spread=0.2, volume=15_000.0)
    cfg = GameConfig(seed=17, use_numba=False)

    env_direct = MarketEnv(data=data, cfg=cfg)
    env_grpc = MarketEnv(data=data, cfg=cfg)
    grpc_service = cast(_EngineGrpcServiceLike, EngineGrpcService(env_grpc))

    direct_state = env_direct.reset(seed=17, start_t=150)
    grpc_state = grpc_service.Reset(
        engine_pb2.ResetRequest(has_seed=True, seed=17, has_start_t=True, start_t=150),
        None,
    ).state
    assert direct_state.t == grpc_state.t == 150
    assert direct_state.cash == grpc_state.cash
    assert direct_state.equity == grpc_state.equity
    assert direct_state.done == grpc_state.done


def test_grpc_reset_omitted_start_t_defaults_to_zero(
    market_data_factory: Callable[..., MarketData],
) -> None:
    data = market_data_factory(n=64, slope=0.1, spread=0.2, volume=10_000.0)
    cfg = GameConfig(seed=3, use_numba=False)
    env_grpc = MarketEnv(data=data, cfg=cfg)
    grpc_service = cast(_EngineGrpcServiceLike, EngineGrpcService(env_grpc))

    grpc_state = grpc_service.Reset(engine_pb2.ResetRequest(has_seed=True, seed=3), None).state
    assert grpc_state.t == 0


def test_grpc_reset_error_parity_for_invalid_start_t(
    market_data_factory: Callable[..., MarketData],
) -> None:
    data = market_data_factory(n=64, slope=0.1, spread=0.2, volume=10_000.0)
    cfg = GameConfig(seed=9, use_numba=False)
    env_direct = MarketEnv(data=data, cfg=cfg)
    env_grpc = MarketEnv(data=data, cfg=cfg)
    grpc_service = cast(_EngineGrpcServiceLike, EngineGrpcService(env_grpc))

    request = engine_pb2.ResetRequest(has_seed=True, seed=9, has_start_t=True, start_t=999)

    try:
        env_direct.reset(seed=9, start_t=999)
        raise AssertionError("direct env did not raise")
    except ValueError as exc_direct:
        direct_msg = str(exc_direct)

    try:
        grpc_service.Reset(request, None)
        raise AssertionError("grpc service did not raise")
    except ValueError as exc_grpc:
        grpc_msg = str(exc_grpc)

    assert "start_t" in direct_msg
    assert "start_t" in grpc_msg
