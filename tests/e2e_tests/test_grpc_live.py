from __future__ import annotations

from collections.abc import Callable
from typing import cast

import grpc
import numpy as np

from stock_simulator.grpc.client import EngineGrpcClient
from tests.conftest import RunningService


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
        market_handle = cast(dict[str, object], observe["market_window_handle"])
        assert int(cast(int, market_handle["t"])) >= 0

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
