from __future__ import annotations

from collections.abc import Callable
from typing import cast

from httpx import post as httpx_post
from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from numpy import isnan as np_isnan
from numpy import nan as np_nan
from numpy import testing as np_testing
from numpy.typing import NDArray
from pytest import skip as pytest_skip

from tests.conftest import RunningService

try:
    from stock_simulator.grpc.client import EngineGrpcClient
except ImportError as exc:
    pytest_skip(f"grpc parity dependencies unavailable: {exc}", allow_module_level=True)


def _http_step_many(
    *,
    base_url: str,
    actions: NDArray[np_float64],
) -> tuple[NDArray[np_float64], NDArray[np_float64], NDArray[np_float64], NDArray[np_float64], NDArray[np_bool_]]:
    encoded = []
    for row in actions:
        encoded.append(
            {
                "side_code": int(row[0]),
                "units": float(row[1]),
                "order_type_code": int(row[2]),
                "has_limit_price": not np_isnan(row[3]),
                "limit_price": None if np_isnan(row[3]) else float(row[3]),
            }
        )
    reply = httpx_post(f"{base_url}/v1/step_many", json={"actions": encoded}, timeout=10.0)
    reply.raise_for_status()
    payload = reply.json()

    market = np_asarray(
        [
            [
                float(obs["market_window_handle"]["start"]),
                float(obs["market_window_handle"]["end"]),
                float(obs["market_window_handle"]["t"]),
                float(obs["market_window_handle"]["current_price"]),
            ]
            for obs in payload["observations"]
        ],
        dtype=np_float64,
    )
    portfolio = np_asarray([obs["portfolio_vector"] for obs in payload["observations"]], dtype=np_float64)
    orders = np_asarray([obs["order_summary_vector"] for obs in payload["observations"]], dtype=np_float64)
    rewards = np_asarray(payload["rewards"], dtype=np_float64)
    dones = np_asarray(payload["dones"], dtype=np_bool_)
    return market, portfolio, orders, rewards, dones


def test_live_http_grpc_deterministic_parity(
    launch_http_server: Callable[..., RunningService],
    launch_grpc_server: Callable[..., RunningService],
) -> None:
    http_service = launch_http_server(engine_enabled=True, with_market_data=True)
    grpc_service = launch_grpc_server(seed=4242, with_market_data=True)

    seed = 4242
    actions = np_asarray(
        [
            [0.0, 0.0, 0.0, np_nan],
            [1.0, 1.5, 0.0, np_nan],
            [1.0, 2.0, 1.0, 99.5],
            [-1.0, 0.7, 0.0, np_nan],
            [-1.0, 1.2, 1.0, 105.0],
        ],
        dtype=np_float64,
    )

    reset_http = httpx_post(f"{http_service.http_base_url}/v1/reset", json={"seed": seed}, timeout=5.0)
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

    grpc_market = np_asarray(
        [
            [
                float(cast(dict[str, float | int], obs["market_window_handle"])["start"]),
                float(cast(dict[str, float | int], obs["market_window_handle"])["end"]),
                float(cast(dict[str, float | int], obs["market_window_handle"])["t"]),
                float(cast(dict[str, float | int], obs["market_window_handle"])["current_price"]),
            ]
            for obs in grpc_observations
        ],
        dtype=np_float64,
    )
    grpc_portfolio = np_asarray(
        [cast(list[float], obs["portfolio_vector"]) for obs in grpc_observations],
        dtype=np_float64,
    )
    grpc_orders = np_asarray(
        [cast(list[float], obs["order_summary_vector"]) for obs in grpc_observations],
        dtype=np_float64,
    )

    np_testing.assert_allclose(http_market, grpc_market, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(http_portfolio, grpc_portfolio, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(http_orders, grpc_orders, rtol=1e-7, atol=1e-9)
    np_testing.assert_allclose(http_rewards, grpc_rewards, rtol=1e-7, atol=1e-9)
    np_testing.assert_array_equal(http_dones, grpc_dones)
