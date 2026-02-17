from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import grpc
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.types import Action
from stocksim_grpc import engine_pb2 as _engine_pb2

engine_pb2: Any = cast(Any, _engine_pb2)


@pytest.fixture
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
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        frame = pd.DataFrame(
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


@pytest.fixture
def cfg_factory() -> Callable[..., GameConfig]:
    def _factory(**overrides: Any) -> GameConfig:
        defaults: dict[str, Any] = {
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


@pytest.fixture
def encode_actions() -> Callable[[Sequence[Action]], NDArray[np.float64]]:
    def _encode(actions: Sequence[Action]) -> NDArray[np.float64]:
        rows: list[list[float]] = []
        for action in actions:
            if action.side == "hold":
                side_code = 0.0
            elif action.side == "buy":
                side_code = 1.0
            else:
                side_code = -1.0
            order_code = 1.0 if action.order_type == "limit" else 0.0
            limit = np.nan if action.limit_price is None else float(action.limit_price)
            rows.append([side_code, float(action.units), order_code, limit])
        return np.asarray(rows, dtype=np.float64)

    return _encode


@dataclass(frozen=True)
class RunningService:
    """Metadata for a started local test service process."""

    process: subprocess.Popen[str]
    host: str
    port: int

    @property
    def http_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def grpc_target(self) -> str:
        return f"{self.host}:{self.port}"


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_http_ready(url: str, timeout_seconds: float = 20.0) -> None:
    import httpx

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code < 600:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for HTTP service at {url}: {last_error}")


def _wait_grpc_ready(target: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
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
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for gRPC service at {target}: {last_error}")


def _write_market_csv(path: Path, n: int = 512) -> None:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.linspace(100.0, 120.0, n)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.full(n, 20_000.0),
        }
    )
    frame.to_csv(path, index=False)


@pytest.fixture
def launch_http_server(tmp_path: Path) -> Generator[Callable[..., RunningService]]:
    """Launch the FastAPI server in a subprocess and wait for readiness."""

    processes: list[subprocess.Popen[str]] = []

    def _launch(*, engine_enabled: bool, with_market_data: bool = False) -> RunningService:
        port = _find_free_port()
        env = dict(os.environ)
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
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "stock_simulator.server"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture
def launch_grpc_server(tmp_path: Path) -> Generator[Callable[..., RunningService]]:
    """Launch the gRPC server in a subprocess and wait for readiness."""

    processes: list[subprocess.Popen[str]] = []

    def _launch(*, seed: int = 1234, with_market_data: bool = False) -> RunningService:
        port = _find_free_port()
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        if with_market_data:
            market_path = tmp_path / f"market-grpc-{port}.csv"
            _write_market_csv(market_path)
            env["MARKET_DATA_CSV"] = str(market_path)
        else:
            env.pop("MARKET_DATA_CSV", None)
        proc = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
            except subprocess.TimeoutExpired:
                proc.kill()
