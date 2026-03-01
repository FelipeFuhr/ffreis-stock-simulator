from __future__ import annotations

from collections.abc import Callable

from httpx import get as httpx_get, post as httpx_post

from tests.conftest import RunningService


def test_http_service_health_ready_metrics(
    launch_http_server: Callable[..., RunningService],
) -> None:
    service = launch_http_server(engine_enabled=False)
    base = service.http_base_url

    health = httpx_get(f"{base}/healthz", timeout=5.0)
    ready = httpx_get(f"{base}/readyz", timeout=5.0)
    metrics = httpx_get(f"{base}/metrics", timeout=5.0, follow_redirects=True)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert metrics.status_code == 200
    assert "python_info" in metrics.text


def test_http_engine_endpoints_live(
    launch_http_server: Callable[..., RunningService],
) -> None:
    service = launch_http_server(engine_enabled=True, with_market_data=True)
    base = service.http_base_url

    reset = httpx_post(f"{base}/v1/reset", json={"seed": 2024}, timeout=5.0)
    assert reset.status_code == 200
    assert reset.json()["state"]["t"] == 0

    observe = httpx_get(f"{base}/v1/observe", timeout=5.0)
    assert observe.status_code == 200
    assert observe.json()["observation"]["market_window_handle"]["t"] >= 0

    step_many = httpx_post(
        f"{base}/v1/step_many",
        json={
            "actions": [
                {
                    "side_code": 0,
                    "units": 0.0,
                    "order_type_code": 0,
                    "has_limit_price": False,
                },
                {
                    "side_code": 1,
                    "units": 2.0,
                    "order_type_code": 0,
                    "has_limit_price": False,
                },
            ]
        },
        timeout=5.0,
    )
    payload = step_many.json()
    assert step_many.status_code == 200
    assert len(payload["observations"]) == 2
    assert len(payload["rewards"]) == 2
    assert len(payload["dones"]) == 2


def test_http_failure_paths(
    launch_http_server: Callable[..., RunningService],
) -> None:
    service = launch_http_server(engine_enabled=True, with_market_data=True)
    base = service.http_base_url
    _ = httpx_post(f"{base}/v1/reset", json={"seed": 11}, timeout=5.0)

    invalid_payload = httpx_post(
        f"{base}/v1/step_many",
        json={"actions": "not-a-list"},
        timeout=5.0,
    )
    assert invalid_payload.status_code == 422

    invalid_action = httpx_post(
        f"{base}/v1/step_many",
        json={
            "actions": [
                {
                    "side_code": 2,
                    "units": 1.0,
                    "order_type_code": 0,
                    "has_limit_price": False,
                }
            ]
        },
        timeout=5.0,
    )
    assert invalid_action.status_code == 400
    assert "invalid side code" in invalid_action.json()["detail"]


def test_http_misconfiguration_sets_not_ready(
    launch_http_server: Callable[..., RunningService],
) -> None:
    service = launch_http_server(engine_enabled=True, with_market_data=False)
    ready = httpx_get(f"{service.http_base_url}/readyz", timeout=5.0)

    assert ready.status_code == 503
    payload = ready.json()
    assert payload["status"] == "not_ready"
    assert payload["engine_enabled"] is True
    assert payload["engine_ready"] is False
