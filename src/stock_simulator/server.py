from __future__ import annotations

from asyncio import Lock as AsyncioLock
from collections.abc import Callable
from importlib import import_module as importlib_import_module
from os import getenv as os_getenv
from time import monotonic
from types import ModuleType
from typing import TYPE_CHECKING, Literal, Protocol, cast

from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from numpy import nan as np_nan
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .config import GameConfig
from .data import MarketData
from .env import MarketEnv
from .types import EnvStateModel, MarketWindowViewHandleModel, ObservationModel, StepTraceRowModel

#: Callers that never send `session_id` (legacy scripts, curl, other-language
#: clients) all land here and share state exactly as before this change --
#: no regression for them, but no isolation either. Every `SimulatorClient`
#: in this fleet should be sending a real, unique session_id instead.
DEFAULT_SESSION_ID = "default"

#: A session with no request in this long is evicted on the next request to
#: ANY session -- bounds memory growth across many short eval/train runs
#: without needing a background sweep task.
SESSION_IDLE_TIMEOUT_S = 60.0 * 30

if TYPE_CHECKING:
    from fastapi import FastAPI


class _UvicornRun(Protocol):
    def __call__(self, app: str, *, host: str, port: int, reload: bool) -> None: ...


class _MakeAsgiApp(Protocol):
    def __call__(self) -> object: ...


class _HTTPExceptionFactory(Protocol):
    def __call__(self, *, status_code: int, detail: str) -> Exception: ...


class _FastApiModule(Protocol):
    FastAPI: type
    HTTPException: _HTTPExceptionFactory
    status: ModuleType


def _load_fastapi_module() -> ModuleType | None:
    try:
        return importlib_import_module("fastapi")
    except ModuleNotFoundError:
        return None


def _load_uvicorn_run() -> _UvicornRun | None:
    try:
        return importlib_import_module("uvicorn").run
    except ModuleNotFoundError:
        return None


def _load_make_asgi_app() -> _MakeAsgiApp | None:
    try:
        return importlib_import_module("prometheus_client").make_asgi_app
    except ModuleNotFoundError:
        return None


def _require_api_dependencies() -> tuple[_FastApiModule, _MakeAsgiApp, _UvicornRun]:
    if fastapi_module is None or make_asgi_app is None or uvicorn_run is None:
        raise RuntimeError("HTTP API dependencies are unavailable. Install with `uv sync --extra api`.")
    return cast(_FastApiModule, fastapi_module), make_asgi_app, uvicorn_run


fastapi_module = _load_fastapi_module()
uvicorn_run = _load_uvicorn_run()
make_asgi_app = _load_make_asgi_app()


class HealthzResponse(BaseModel):
    """Liveness response payload."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]


class ReadyzResponse(BaseModel):
    """Readiness response payload."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ready", "not_ready"]
    engine_enabled: bool
    engine_ready: bool


class ResetRequest(BaseModel):
    """Reset request payload."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    session_id: str = DEFAULT_SESSION_ID


class ResetResponse(BaseModel):
    """Reset response payload."""

    model_config = ConfigDict(extra="forbid")

    state: EnvStateModel


class ObserveResponse(BaseModel):
    """Observe response payload."""

    model_config = ConfigDict(extra="forbid")

    observation: ObservationModel


class EncodedActionModel(BaseModel):
    """Transport model for encoded ``step_many`` action rows."""

    model_config = ConfigDict(extra="forbid")

    side_code: int
    units: float
    order_type_code: int
    has_limit_price: bool
    limit_price: float | None = None


class StepManyRequest(BaseModel):
    """Step-many request payload."""

    model_config = ConfigDict(extra="forbid")

    actions: list[EncodedActionModel]
    include_trace: bool = False
    session_id: str = DEFAULT_SESSION_ID


class StepManyResponse(BaseModel):
    """Step-many response payload."""

    model_config = ConfigDict(extra="forbid")

    observations: list[ObservationModel]
    rewards: list[float]
    dones: list[bool]
    trace: list[StepTraceRowModel]


class RuntimeState:
    """Mutable process-local runtime state for the FastAPI service.

    One `MarketEnv` per `session_id`, built from the same shared, read-only
    `MarketData`/`GameConfig` -- the market arrays are never mutated, only
    read, so sharing them across sessions is safe; each session keeps its
    own independent core state, RNG, and portfolio. Without this split, two
    concurrent callers (two training/eval runs against the same process)
    silently interleave resets and steps: one client's `/v1/reset` zeroes
    another's in-flight portfolio, and the two share one `t`. The failure
    mode is not an error -- both clients keep getting plausible-looking
    responses, just wrong ones. See
    `stock-rl-agent-repo/docs/incident-simulator-session-isolation.md`.
    """

    def __init__(self) -> None:
        self.engine_enabled = _as_bool(os_getenv("ENGINE_ENABLED"), default=True)
        self.engine_ready = False
        self.engine_error: str | None = None
        self._env_factory: Callable[[], MarketEnv] | None = None
        self._sessions: dict[str, MarketEnv] = {}
        self._session_locks: dict[str, AsyncioLock] = {}
        self._session_last_access: dict[str, float] = {}

    def set_env_factory(self, factory: Callable[[], MarketEnv]) -> None:
        """Set the factory every session's fresh `MarketEnv` is built from."""
        self._env_factory = factory

    def require_session(self, session_id: str) -> MarketEnv:
        """Return `session_id`'s `MarketEnv`, creating it fresh on first use."""
        if self._env_factory is None:
            raise RuntimeError("engine is not available")
        self._evict_idle_sessions()
        env = self._sessions.get(session_id)
        if env is None:
            env = self._env_factory()
            self._sessions[session_id] = env
            self._session_locks[session_id] = AsyncioLock()
        self._session_last_access[session_id] = monotonic()
        return env

    def session_lock(self, session_id: str) -> AsyncioLock:
        """Lock serializing requests for one session (not across sessions)."""
        return self._session_locks[session_id]

    def _evict_idle_sessions(self) -> None:
        now = monotonic()
        stale = [sid for sid, last in self._session_last_access.items() if now - last > SESSION_IDLE_TIMEOUT_S]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._session_locks.pop(sid, None)
            self._session_last_access.pop(sid, None)


def _as_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean-like environment variable value."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _load_engine_factory() -> Callable[[], MarketEnv]:
    """Build the per-session env factory from environment configuration.

    Loads `MarketData`/`GameConfig` once (shared, read-only, across every
    session) and returns a callable that builds one fresh, independent
    `MarketEnv` per call -- `RuntimeState.require_session` calls it exactly
    once per new `session_id`.
    """
    market_data_csv = os_getenv("MARKET_DATA_CSV")
    if not market_data_csv:
        raise ValueError("MARKET_DATA_CSV must be set when ENGINE_ENABLED=true")
    market_data = MarketData.from_csv(market_data_csv)
    config_yaml = os_getenv("STOCK_SIM_CONFIG_YAML")
    config = GameConfig.load(yaml_path=config_yaml)
    return lambda: MarketEnv(data=market_data, cfg=config)


def _build_actions_matrix(
    actions: list[EncodedActionModel],
    http_exception_cls: _HTTPExceptionFactory,
    bad_request_status: int,
) -> NDArray[np_float64]:
    """Convert encoded action models to a numpy matrix for the engine."""
    rows: list[list[float]] = []
    for action in actions:
        limit_price = np_nan
        if action.has_limit_price:
            if action.limit_price is None:
                raise http_exception_cls(  # NOSONAR — factory callable wraps FastAPI HTTPException, not a generic base
                    status_code=bad_request_status,
                    detail="limit_price must be provided when has_limit_price=true",
                )
            limit_price = float(action.limit_price)
        rows.append(
            [
                float(action.side_code),
                float(action.units),
                float(action.order_type_code),
                float(limit_price),
            ]
        )
    return np_asarray(rows, dtype=np_float64)


def _build_step_many_response(
    observations: dict[str, NDArray[np_float64]],
    rewards: NDArray[np_float64],
    dones: NDArray[np_bool_],
    trace: tuple[object, ...],
) -> StepManyResponse:
    """Build a StepManyResponse from raw engine outputs."""
    num_rows = int(rewards.shape[0])
    observation_rows: list[ObservationModel] = []
    for i in range(num_rows):
        observation_rows.append(
            ObservationModel(
                market_window_handle=MarketWindowViewHandleModel(
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
    return StepManyResponse(
        observations=observation_rows,
        rewards=rewards.tolist(),
        dones=dones.tolist(),
        trace=[StepTraceRowModel.from_dataclass(row) for row in trace],  # type: ignore[arg-type]
    )


def create_app() -> FastAPI:
    """Create the HTTP service app with health/readiness and metrics endpoints."""
    fastapi_mod, make_asgi_app, _ = _require_api_dependencies()
    json_response_cls = importlib_import_module("fastapi.responses").JSONResponse
    fastapi_cls = fastapi_mod.FastAPI
    http_exception_cls = fastapi_mod.HTTPException
    status = fastapi_mod.status

    app = fastapi_cls(
        title="Stock Simulator Service",
        version="0.1.0",
        description="Health/ready endpoints and Prometheus metrics endpoint.",
    )
    runtime = RuntimeState()
    app.state.runtime = runtime

    app.mount("/metrics", make_asgi_app())

    @app.on_event("startup")
    async def startup() -> None:
        if not runtime.engine_enabled:
            runtime.engine_ready = True
            return
        try:
            runtime.set_env_factory(_load_engine_factory())
            runtime.engine_ready = True
        except Exception as exc:  # pragma: no cover
            runtime.engine_ready = False
            runtime.engine_error = str(exc)

    def require_session(session_id: str) -> MarketEnv:
        try:
            return runtime.require_session(session_id)
        except RuntimeError as exc:
            raise http_exception_cls(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @app.get("/healthz", response_model=HealthzResponse)
    async def healthz() -> HealthzResponse:
        return HealthzResponse(status="ok")

    @app.get("/readyz", response_model=ReadyzResponse)
    async def readyz() -> object:
        if not runtime.engine_ready:
            payload = ReadyzResponse(
                status="not_ready",
                engine_enabled=runtime.engine_enabled,
                engine_ready=False,
            )
            return json_response_cls(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(),
            )
        return ReadyzResponse(
            status="ready",
            engine_enabled=runtime.engine_enabled,
            engine_ready=True,
        )

    @app.post("/v1/reset", response_model=ResetResponse)
    async def reset(payload: ResetRequest) -> ResetResponse:
        env = require_session(payload.session_id)
        async with runtime.session_lock(payload.session_id):
            state = env.reset(seed=payload.seed)
        return ResetResponse(state=EnvStateModel.from_dataclass(state))

    @app.get("/v1/observe", response_model=ObserveResponse)
    async def observe(session_id: str = DEFAULT_SESSION_ID) -> ObserveResponse:
        env = require_session(session_id)
        async with runtime.session_lock(session_id):
            observation = env.observe()
        return ObserveResponse(observation=ObservationModel.from_dataclass(observation))

    @app.post("/v1/step_many", response_model=StepManyResponse)
    async def step_many(payload: StepManyRequest) -> StepManyResponse:
        env = require_session(payload.session_id)
        if not payload.actions:
            return StepManyResponse(observations=[], rewards=[], dones=[], trace=[])
        actions_matrix = _build_actions_matrix(payload.actions, http_exception_cls, status.HTTP_400_BAD_REQUEST)
        try:
            async with runtime.session_lock(payload.session_id):
                observations, rewards, dones, trace = env.step_many(
                    actions_matrix,
                    include_trace=payload.include_trace,
                )
        except ValueError as exc:
            raise http_exception_cls(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return _build_step_many_response(observations, rewards, dones, trace)

    return app


if fastapi_module is not None and make_asgi_app is not None:
    app: FastAPI | None = create_app()
else:
    app = None


def main() -> None:
    """Run the ASGI server entrypoint."""
    _, _, uvicorn_run_required = _require_api_dependencies()
    if app is None:
        raise RuntimeError("HTTP API app is unavailable. Install with `uv sync --extra api`.")
    host = os_getenv("HOST", "0.0.0.0")
    port = int(os_getenv("PORT", "8000"))
    uvicorn_run_required("stock_simulator.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
