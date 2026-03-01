from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict, cast

import grpc
import numpy as np
import pytest

from tests.conftest import RunningService

try:
    from stock_simulator.grpc.client import EngineGrpcClient
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(f"grpc parity dependencies unavailable: {exc}", allow_module_level=True)


class _MarketWindowHandle(TypedDict):
    start: int
    end: int
    t: int
    current_price: float


def test_grpc_live_service_roundtrip(
    launch_grpc_server: Callable[..., RunningService],
) -> None:
    service = launch_grpc_server(seed=1234, with_market_data=True)
    client = EngineGrpcClient(target=service.grpc_target)
    try:
        assert client.ping() == "pong"

        reset = client.reset(seed=1234)
        assert reset["t"] == 0
        assert reset["done"] is False

        observe = client.observe()
        market_handle = cast(_MarketWindowHandle, observe["market_window_handle"])
        assert int(market_handle["t"]) >= 0

        actions = np.asarray(
            [
                [0.0, 0.0, 0.0, np.nan],
                [1.0, 2.0, 0.0, np.nan],
                [-1.0, 1.0, 1.0, 101.0],
            ],
            dtype=np.float64,
        )
        observations, rewards, dones = client.step_many(actions)
        assert len(observations) == 3
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
    finally:
        client.close()


def test_grpc_live_invalid_action_returns_invalid_argument(
    launch_grpc_server: Callable[..., RunningService],
) -> None:
    service = launch_grpc_server(seed=1, with_market_data=True)
    client = EngineGrpcClient(target=service.grpc_target)
    try:
        _ = client.reset(seed=1)
        invalid = np.asarray([[2.0, 1.0, 0.0, np.nan]], dtype=np.float64)
        try:
            _ = client.step_many(invalid)
            raise AssertionError("step_many should fail for invalid side code")
        except grpc.RpcError as exc:
            assert exc.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "invalid side code" in (exc.details() or "")
    finally:
        client.close()
