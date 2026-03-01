from __future__ import annotations

from contextlib import closing as contextlib_closing
from os import environ as os_environ
from socket import AF_INET as socket_AF_INET, SOCK_STREAM as socket_SOCK_STREAM, socket as socket_socket
from subprocess import DEVNULL as subprocess_DEVNULL, Popen as subprocess_Popen, TimeoutExpired as subprocess_TimeoutExpired
from sys import executable as sys_executable
from time import sleep as time_sleep, time as time_time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

from numpy import asarray as np_asarray, float64 as np_float64, full as np_full, linspace as np_linspace, nan as np_nan
from pandas import DataFrame as pd_DataFrame, date_range as pd_date_range
from pytest import fixture as pytest_fixture, skip as pytest_skip
from numpy.typing import NDArray

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.types import Action

try:
    from stocksim_grpc import engine_pb2 as _engine_pb2
except ImportError:  # pragma: no cover
    _engine_pb2 = None

try:
    import grpc as _grpc
except ModuleNotFoundError:  # pragma: no cover
    _grpc = None

grpc = cast(ModuleType | None, _grpc)
engine_pb2 = cast(ModuleType | None, _engine_pb2)


@pytest_fixture
def market_data_factory() -> Callable[..., MarketData]:
    def _factory(
        *,
        n: int = 256,
        close: Sequence[float] | None = None,
        base_price: float = 100.0,
        slope: float = 0.1,
        spread: float = 0.5,
        volume: float = 10_000.0,
    ) -> MarketData:
        if close is None:
            close_values = [base_price + slope * i for i in range(n)]
        else:
            close_values = [float(v) for v in close]
            n = len(close_values)
        idx = pd_date_range("2024-01-01", periods=n, freq="h")
        frame = pd_DataFrame(
            {
                "timestamp": idx,
                "open": close_values,
                "high": [v + spread for v in close_values],
                "low": [v - spread for v in close_values],
                "close": close_values,
                "volume": [volume] * n,
            }
        )
        return MarketData(frame)

    return _factory


@pytest_fixture
def cfg_factory() -> Callable[..., GameConfig]:
    def _factory(**overrides: int | float | bool) -> GameConfig:
        defaults: dict[str, int | float | bool] = {
            "seed": 123,
            "use_numba": False,
            "market_latency_bars": 0,
            "partial_fill_min": 1.0,
            "partial_fill_max": 1.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
        }
        defaults.update(overrides)
        return GameConfig(**defaults)

    return _factory


@pytest_fixture
def encode_actions() -> Callable[[Sequence[Action]], NDArray[np_float64]]:
    def _encode(actions: Sequence[Action]) -> NDArray[np_float64]:
        rows: list[list[float]] = []
        for action in actions:
            if action.side == "hold":
                side_code = 0.0
            elif action.side == "buy":
                side_code = 1.0
            else:
                side_code = -1.0
            order_code = 1.0 if action.order_type == "limit" else 0.0
            limit = np_nan if action.limit_price is None else float(action.limit_price)
            rows.append([side_code, float(action.units), order_code, limit])
        return np_asarray(rows, dtype=np_float64)

    return _encode


@dataclass(frozen=True)
class RunningService:
    """Metadata for a started local test service process."""

    process: subprocess_Popen[str]
    host: str
    port: int

    @property
    def http_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def grpc_target(self) -> str:
        return f"{self.host}:{self.port}"


def _find_free_port() -> int:
    with contextlib_closing(socket_socket(socket_AF_INET, socket_SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_http_ready(url: str, timeout_seconds: float = 20.0) -> None:
    from httpx import get as httpx_get

    deadline = time_time() + timeout_seconds
    last_error: Exception | None = None
    while time_time() < deadline:
        try:
            response = httpx_get(url, timeout=1.0)
            if response.status_code < 600:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time_sleep(0.2)
    raise TimeoutError(f"timed out waiting for HTTP service at {url}: {last_error}")


def _wait_grpc_ready(target: str, timeout_seconds: float = 20.0) -> None:
    if grpc is None or engine_pb2 is None:
        pytest_skip("grpc stubs/runtime are required for gRPC fixtures/tests")
    deadline = time_time() + timeout_seconds
    last_error: Exception | None = None
    while time_time() < deadline:
        try:
            with grpc.insecure_channel(target) as channel:
                ping_rpc = channel.unary_unary(
                    "/stocksim.grpc.EngineService/Ping",
                    request_serializer=engine_pb2.PingRequest.SerializeToString,
                    response_deserializer=engine_pb2.PingResponse.FromString,
                )
                reply = ping_rpc(engine_pb2.PingRequest(), timeout=1.0)
                if reply.status == "pong":
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time_sleep(0.2)
    raise TimeoutError(f"timed out waiting for gRPC service at {target}: {last_error}")


def _write_market_csv(path: Path, n: int = 512) -> None:
    idx = pd_date_range("2024-01-01", periods=n, freq="h")
    close = np_linspace(100.0, 120.0, n)
    frame = pd_DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np_full(n, 20_000.0),
        }
    )
    frame.to_csv(path, index=False)


@pytest_fixture
def launch_http_server(tmp_path: Path) -> Generator[Callable[..., RunningService]]:
    """Launch the FastAPI server in a subprocess and wait for readiness."""

    processes: list[subprocess_Popen[str]] = []

    def _launch(*, engine_enabled: bool, with_market_data: bool = False) -> RunningService:
        port = _find_free_port()
        env = dict(os_environ)
        env["PYTHONPATH"] = "src"
        env["ENGINE_ENABLED"] = "true" if engine_enabled else "false"
        env["HOST"] = "127.0.0.1"
        env["PORT"] = str(port)
        if with_market_data:
            market_path = tmp_path / f"market-{port}.csv"
            _write_market_csv(market_path)
            env["MARKET_DATA_CSV"] = str(market_path)
        else:
            env.pop("MARKET_DATA_CSV", None)
        proc = subprocess_Popen(  # noqa: S603
            [sys_executable, "-m", "stock_simulator.server"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=subprocess_DEVNULL,
            stderr=subprocess_DEVNULL,
            text=True,
        )
        processes.append(proc)
        _wait_http_ready(f"http://127.0.0.1:{port}/readyz")
        return RunningService(process=proc, host="127.0.0.1", port=port)

    yield _launch

    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess_TimeoutExpired:
                proc.kill()


@pytest_fixture
def launch_grpc_server(tmp_path: Path) -> Generator[Callable[..., RunningService]]:
    """Launch the gRPC server in a subprocess and wait for readiness."""

    processes: list[subprocess_Popen[str]] = []

    def _launch(*, seed: int = 1234, with_market_data: bool = False) -> RunningService:
        port = _find_free_port()
        env = dict(os_environ)
        env["PYTHONPATH"] = "src"
        if with_market_data:
            market_path = tmp_path / f"market-grpc-{port}.csv"
            _write_market_csv(market_path)
            env["MARKET_DATA_CSV"] = str(market_path)
        else:
            env.pop("MARKET_DATA_CSV", None)
        proc = subprocess_Popen(  # noqa: S603
            [
                sys_executable,
                "-m",
                "stock_simulator.grpc.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--seed",
                str(seed),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=subprocess_DEVNULL,
            stderr=subprocess_DEVNULL,
            text=True,
        )
        processes.append(proc)
        _wait_grpc_ready(f"127.0.0.1:{port}")
        return RunningService(process=proc, host="127.0.0.1", port=port)

    yield _launch

    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess_TimeoutExpired:
                proc.kill()
