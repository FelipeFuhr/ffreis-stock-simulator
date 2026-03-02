from __future__ import annotations

from importlib import import_module as importlib_import_module
from os import getenv as os_getenv
from types import ModuleType
from typing import TYPE_CHECKING, Literal, Protocol, cast

from numpy import asarray as np_asarray
from numpy import float64 as np_float64
from numpy import nan as np_nan
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .config import GameConfig
from .data import MarketData
from .env import MarketEnv
from .types import EnvStateModel, MarketWindowViewHandleModel, ObservationModel

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


class StepManyResponse(BaseModel):
    """Step-many response payload."""

    model_config = ConfigDict(extra="forbid")

    observations: list[ObservationModel]
    rewards: list[float]
    dones: list[bool]


class RuntimeState:
    """Mutable process-local runtime state for the FastAPI service."""

    def __init__(self) -> None:
        self.engine_enabled = _as_bool(os_getenv("ENGINE_ENABLED"), default=True)
        self.engine_ready = False
        self.engine_error: str | None = None
        self.env: MarketEnv | None = None


def _as_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean-like environment variable value."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _load_engine() -> MarketEnv:
    """Construct a simulator instance from environment configuration."""
    market_data_csv = os_getenv("MARKET_DATA_CSV")
    if not market_data_csv:
        raise ValueError("MARKET_DATA_CSV must be set when ENGINE_ENABLED=true")
    market_data = MarketData.from_csv(market_data_csv)
    config_yaml = os_getenv("STOCK_SIM_CONFIG_YAML")
    config = GameConfig.load(yaml_path=config_yaml)
    return MarketEnv(data=market_data, cfg=config)


def create_app() -> FastAPI:
    """Create the HTTP service app with health/readiness and metrics endpoints."""
    fastapi_mod, make_asgi_app, _ = _require_api_dependencies()
    JSONResponse = importlib_import_module("fastapi.responses").JSONResponse
    FastAPI = fastapi_mod.FastAPI
    HTTPException = fastapi_mod.HTTPException
    status = fastapi_mod.status

    app = FastAPI(
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
            runtime.env = _load_engine()
            runtime.engine_ready = True
        except Exception as exc:  # pragma: no cover
            runtime.engine_ready = False
            runtime.engine_error = str(exc)

    def require_env() -> MarketEnv:
        if runtime.env is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="engine is not available",
            )
        return runtime.env

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
            return JSONResponse(
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
        env = require_env()
        state = env.reset(seed=payload.seed)
        return ResetResponse(state=EnvStateModel.from_dataclass(state))

    @app.get("/v1/observe", response_model=ObserveResponse)
    async def observe() -> ObserveResponse:
        env = require_env()
        observation = env.observe()
        return ObserveResponse(observation=ObservationModel.from_dataclass(observation))

    @app.post("/v1/step_many", response_model=StepManyResponse)
    async def step_many(payload: StepManyRequest) -> StepManyResponse:
        env = require_env()
        if not payload.actions:
            return StepManyResponse(observations=[], rewards=[], dones=[])

        rows: list[list[float]] = []
        for action in payload.actions:
            limit_price = np_nan
            if action.has_limit_price:
                if action.limit_price is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
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
        actions_matrix: NDArray[np_float64] = np_asarray(rows, dtype=np_float64)
        try:
            observations, rewards, dones = env.step_many(actions_matrix)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

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
        )

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
