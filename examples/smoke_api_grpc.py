from __future__ import annotations

import json
import os
import time
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit

import grpc

from stocksim_grpc import engine_pb2


def _validate_http_base(api_base: str) -> tuple[str, str]:
    parsed = urlsplit(api_base)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"SIM_API_BASE must use http or https scheme; got {parsed.scheme or '<empty>'}"
        )
    if not parsed.netloc:
        raise ValueError("SIM_API_BASE must include network location (host[:port])")
    return parsed.scheme, parsed.netloc


def _http_request(
    scheme: str,
    netloc: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 5.0,
) -> tuple[int, bytes]:
    conn_cls = HTTPSConnection if scheme == "https" else HTTPConnection
    conn = conn_cls(netloc, timeout=timeout_seconds)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    status = response.status
    payload = response.read()
    conn.close()
    return status, payload


def _wait_http_ok(scheme: str, netloc: str, path: str, timeout_seconds: float = 30.0) -> bytes:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            status, payload = _http_request(
                scheme, netloc, path, timeout_seconds=3.0
            )
            if status == 200:
                return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for HTTP 200 at {path}: {last_error}")


def _assert_http_endpoints(api_base: str) -> None:
    scheme, netloc = _validate_http_base(api_base)
    health_body = _wait_http_ok(scheme, netloc, "/healthz")
    ready_body = _wait_http_ok(scheme, netloc, "/readyz")

    health_payload = json.loads(health_body.decode("utf-8"))
    ready_payload = json.loads(ready_body.decode("utf-8"))

    assert health_payload.get("status") == "ok", health_payload
    assert ready_payload.get("status") == "ready", ready_payload

    if ready_payload.get("engine_enabled") is True:
        status, reset_body = _http_request(
            scheme,
            netloc,
            "/v1/reset",
            method="POST",
            body=json.dumps({"seed": 1234}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout_seconds=5.0,
        )
        assert status == 200
        reset_payload = json.loads(reset_body.decode("utf-8"))
        assert reset_payload["state"]["t"] == 0

        status, observe_body = _http_request(
            scheme, netloc, "/v1/observe", timeout_seconds=5.0
        )
        assert status == 200
        observe_payload = json.loads(observe_body.decode("utf-8"))
        assert observe_payload["observation"]["market_window_handle"]["t"] >= 0


def _assert_grpc_endpoints(grpc_target: str) -> None:
    with grpc.insecure_channel(grpc_target) as channel:
        ping_rpc = channel.unary_unary(
            "/stocksim.grpc.EngineService/Ping",
            request_serializer=engine_pb2.PingRequest.SerializeToString,
            response_deserializer=engine_pb2.PingResponse.FromString,
        )
        reset_rpc = channel.unary_unary(
            "/stocksim.grpc.EngineService/Reset",
            request_serializer=engine_pb2.ResetRequest.SerializeToString,
            response_deserializer=engine_pb2.ResetResponse.FromString,
        )
        observe_rpc = channel.unary_unary(
            "/stocksim.grpc.EngineService/Observe",
            request_serializer=engine_pb2.ObserveRequest.SerializeToString,
            response_deserializer=engine_pb2.ObserveResponse.FromString,
        )

        ping = ping_rpc(engine_pb2.PingRequest(), timeout=5.0)
        assert ping.status == "pong", ping

        reset = reset_rpc(
            engine_pb2.ResetRequest(has_seed=True, seed=1234),
            timeout=5.0,
        )
        assert reset.state.t == 0, reset
        assert reset.state.done is False, reset

        observed = observe_rpc(engine_pb2.ObserveRequest(), timeout=5.0)
        assert observed.observation.market_window_handle.t >= 0, observed


def main() -> None:
    api_base = os.getenv("SIM_API_BASE", "http://simulator-api:8000")
    grpc_target = os.getenv("SIM_GRPC_TARGET", "simulator-grpc:50051")

    _assert_http_endpoints(api_base)
    _assert_grpc_endpoints(grpc_target)

    print("stock simulator API and gRPC smoke checks passed")


if __name__ == "__main__":
    main()
