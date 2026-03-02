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
from stock_simulator.types import Action, EnvStateModel, MarketWindowViewHandleModel, ObservationModel

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


class _ResetRequestLike(_MessageTypeLike, Protocol):
    has_seed: bool
    seed: int


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
    def __call__(self, *, actions: list[_EncodedActionLike]) -> _StepManyRequestLike: ...


class _FactoryResetRequest(Protocol):
    def __call__(self, *, has_seed: bool, seed: int) -> _ResetRequestLike: ...


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
_UNMAPPED_HTTP_PATHS: set[str] = {"/readyz"}
_UNMAPPED_GRPC_METHODS: set[str] = set()
_HYPOTHESIS_MAX_EXAMPLES = int(os_getenv("HYPOTHESIS_MAX_EXAMPLES", "25"))


def _build_step_many_request(actions: NDArray[np_float64]) -> _StepManyRequestLike:
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
    return engine_pb2.StepManyRequest(actions=encoded)


def _grpc_reply_to_arrays(
    reply: _StepManyReplyLike,
) -> tuple[
    NDArray[np_float64],
    NDArray[np_float64],
    NDArray[np_float64],
    NDArray[np_float64],
    NDArray[np_bool_],
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
    return market, portfolio, orders, rewards, dones


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
    action_fields = _contract_fields(engine_pb2.EncodedAction)

    assert request_fields == {"actions"}
    assert action_fields == {
        "side_code",
        "units",
        "order_type_code",
        "has_limit_price",
        "limit_price",
    }


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

    direct_obs_stack, direct_rewards, direct_dones = env_direct.step_many(actions)
    grpc_reply = grpc_service.StepMany(_build_step_many_request(actions), None)

    grpc_market, grpc_portfolio, grpc_orders, grpc_rewards, grpc_dones = _grpc_reply_to_arrays(grpc_reply)

    np_testing.assert_allclose(direct_obs_stack["market_window_handle"], grpc_market)
    np_testing.assert_allclose(direct_obs_stack["portfolio_vector"], grpc_portfolio)
    np_testing.assert_allclose(direct_obs_stack["order_summary_vector"], grpc_orders)
    np_testing.assert_allclose(direct_rewards, grpc_rewards)
    np_testing.assert_array_equal(direct_dones, grpc_dones)


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
    direct_obs_stack, direct_rewards, direct_dones = env_direct.step_many(encoded)
    grpc_reply = grpc_service.StepMany(_build_step_many_request(encoded), None)
    grpc_market, grpc_portfolio, grpc_orders, grpc_rewards, grpc_dones = _grpc_reply_to_arrays(grpc_reply)

    np_testing.assert_allclose(direct_obs_stack["market_window_handle"], grpc_market, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(direct_obs_stack["portfolio_vector"], grpc_portfolio, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(direct_obs_stack["order_summary_vector"], grpc_orders, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(direct_rewards, grpc_rewards, rtol=1e-7, atol=1e-9)
    np_testing.assert_array_equal(direct_dones, grpc_dones)


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
