from __future__ import annotations

from collections.abc import Callable
from typing import cast

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray

from tests.conftest import RunningService

try:
    from stock_simulator.grpc.client import EngineGrpcClient
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(f"grpc parity dependencies unavailable: {exc}", allow_module_level=True)


def _http_step_many(
    *,
    base_url: str,
    actions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    encoded = []
    for row in actions:
        encoded.append(
            {
                "side_code": int(row[0]),
                "units": float(row[1]),
                "order_type_code": int(row[2]),
                "has_limit_price": not np.isnan(row[3]),
                "limit_price": None if np.isnan(row[3]) else float(row[3]),
            }
        )
    reply = httpx.post(f"{base_url}/v1/step_many", json={"actions": encoded}, timeout=10.0)
    reply.raise_for_status()
    payload = reply.json()

    market = np.asarray(
        [
            [
                float(obs["market_window_handle"]["start"]),
                float(obs["market_window_handle"]["end"]),
                float(obs["market_window_handle"]["t"]),
                float(obs["market_window_handle"]["current_price"]),
            ]
            for obs in payload["observations"]
        ],
        dtype=np.float64,
    )
    portfolio = np.asarray([obs["portfolio_vector"] for obs in payload["observations"]], dtype=np.float64)
    orders = np.asarray([obs["order_summary_vector"] for obs in payload["observations"]], dtype=np.float64)
    rewards = np.asarray(payload["rewards"], dtype=np.float64)
    dones = np.asarray(payload["dones"], dtype=np.bool_)
    return market, portfolio, orders, rewards, dones


def test_live_http_grpc_deterministic_parity(
    launch_http_server: Callable[..., RunningService],
    launch_grpc_server: Callable[..., RunningService],
) -> None:
    http_service = launch_http_server(engine_enabled=True, with_market_data=True)
    grpc_service = launch_grpc_server(seed=4242, with_market_data=True)

    seed = 4242
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0, np.nan],
            [1.0, 1.5, 0.0, np.nan],
            [1.0, 2.0, 1.0, 99.5],
            [-1.0, 0.7, 0.0, np.nan],
            [-1.0, 1.2, 1.0, 105.0],
        ],
        dtype=np.float64,
    )

    reset_http = httpx.post(f"{http_service.http_base_url}/v1/reset", json={"seed": seed}, timeout=5.0)
    reset_http.raise_for_status()

    grpc_client = EngineGrpcClient(target=grpc_service.grpc_target)
    try:
        _ = grpc_client.reset(seed=seed)

        http_market, http_portfolio, http_orders, http_rewards, http_dones = _http_step_many(
            base_url=http_service.http_base_url,
            actions=actions,
        )
        grpc_observations, grpc_rewards, grpc_dones = grpc_client.step_many(actions)
    finally:
        grpc_client.close()

    grpc_market = np.asarray(
        [
            [
                float(cast(dict[str, float | int], obs["market_window_handle"])["start"]),
                float(cast(dict[str, float | int], obs["market_window_handle"])["end"]),
                float(cast(dict[str, float | int], obs["market_window_handle"])["t"]),
                float(cast(dict[str, float | int], obs["market_window_handle"])["current_price"]),
            ]
            for obs in grpc_observations
        ],
        dtype=np.float64,
    )
    grpc_portfolio = np.asarray(
        [cast(list[float], obs["portfolio_vector"]) for obs in grpc_observations],
        dtype=np.float64,
    )
    grpc_orders = np.asarray(
        [cast(list[float], obs["order_summary_vector"]) for obs in grpc_observations],
        dtype=np.float64,
    )

    np.testing.assert_allclose(http_market, grpc_market, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(http_portfolio, grpc_portfolio, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(http_orders, grpc_orders, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(http_rewards, grpc_rewards, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(http_dones, grpc_dones)
