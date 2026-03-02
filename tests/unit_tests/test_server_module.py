from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fastapi.testclient import TestClient
from numpy import asarray as np_asarray
from numpy import float64 as np_float64
from pytest import MonkeyPatch

from stock_simulator import server as server_mod
from stock_simulator.types import EnvState, MarketWindowViewHandle, Observation


@dataclass
class _FakeEnv:
    raise_on_step: bool = False

    def reset(self, seed: int | None = None) -> EnvState:
        _ = seed
        return EnvState(
            t=0,
            cash=1000.0,
            units=0.0,
            equity=1000.0,
            leverage=0.0,
            open_orders=0,
            done=False,
        )

    def observe(self) -> Observation:
        return Observation(
            market=MarketWindowViewHandle(start=0, end=64, t=3, current_price=101.5),
            portfolio_vector=np_asarray([1000.0, 0.0, 1000.0, 0.0], dtype=np_float64),
            order_summary_vector=np_asarray([0.0, 0.0, 0.0], dtype=np_float64),
            done=False,
        )

    def step_many(self, _actions: object) -> tuple[dict[str, object], object, object]:
        if self.raise_on_step:
            raise ValueError("invalid order_type code")
        observations: dict[str, object] = {
            "market_window_handle": np_asarray([[0.0, 64.0, 4.0, 102.0]], dtype=np_float64),
            "portfolio_vector": np_asarray([[999.0, 1.0, 1101.0, 0.2]], dtype=np_float64),
            "order_summary_vector": np_asarray([[1.0, 1.0, 0.0]], dtype=np_float64),
        }
        rewards = np_asarray([101.0], dtype=np_float64)
        dones = np_asarray([False], dtype=np_float64)
        return observations, cast(object, rewards), cast(object, dones)


def test_as_bool_parses_expected_values() -> None:
    assert server_mod._as_bool(None, default=True) is True
    assert server_mod._as_bool("true", default=False) is True
    assert server_mod._as_bool(" YES ", default=False) is True
    assert server_mod._as_bool("0", default=True) is False


def test_load_engine_requires_market_data_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("MARKET_DATA_CSV", raising=False)
    try:
        server_mod._load_engine()
        raise AssertionError("expected ValueError for missing MARKET_DATA_CSV")
    except ValueError as exc:
        assert "MARKET_DATA_CSV must be set" in str(exc)


def test_create_app_readyz_when_engine_disabled(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_ENABLED", "false")
    app = server_mod.create_app()
    with TestClient(app) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"

        # Engine disabled means no runtime env for action routes.
        reset = client.post("/v1/reset", json={"seed": 7})
        assert reset.status_code == 503


def test_create_app_action_routes_with_loaded_engine(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_ENABLED", "true")
    monkeypatch.setattr(server_mod, "_load_engine", lambda: _FakeEnv())
    app = server_mod.create_app()

    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["engine_ready"] is True

        reset = client.post("/v1/reset", json={"seed": 7})
        assert reset.status_code == 200
        assert reset.json()["state"]["t"] == 0

        observe = client.get("/v1/observe")
        assert observe.status_code == 200
        assert observe.json()["observation"]["market_window_handle"]["t"] == 3

        empty_step = client.post("/v1/step_many", json={"actions": []})
        assert empty_step.status_code == 200
        assert empty_step.json() == {"observations": [], "rewards": [], "dones": []}

        invalid_limit = client.post(
            "/v1/step_many",
            json={
                "actions": [
                    {
                        "side_code": 1,
                        "units": 1.0,
                        "order_type_code": 1,
                        "has_limit_price": True,
                    }
                ]
            },
        )
        assert invalid_limit.status_code == 400
        assert "limit_price must be provided" in invalid_limit.json()["detail"]

        valid_step = client.post(
            "/v1/step_many",
            json={
                "actions": [
                    {
                        "side_code": 1,
                        "units": 1.0,
                        "order_type_code": 0,
                        "has_limit_price": False,
                    }
                ]
            },
        )
        assert valid_step.status_code == 200
        payload = valid_step.json()
        assert payload["rewards"] == [101.0]
        assert payload["dones"] == [False]


def test_create_app_step_many_maps_value_error_to_http_400(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_ENABLED", "true")
    monkeypatch.setattr(server_mod, "_load_engine", lambda: _FakeEnv(raise_on_step=True))
    app = server_mod.create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/step_many",
            json={
                "actions": [
                    {
                        "side_code": 1,
                        "units": 1.0,
                        "order_type_code": 2,
                        "has_limit_price": False,
                    }
                ]
            },
        )
        assert response.status_code == 400
        assert "invalid order_type code" in response.json()["detail"]


def test_main_uses_host_and_port_from_env(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, str, int, bool]] = []

    def _fake_run(app: str, host: str, port: int, reload: bool) -> None:
        calls.append((app, host, port, reload))

    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setattr(server_mod, "uvicorn_run", _fake_run)

    server_mod.main()

    assert calls == [("stock_simulator.server:app", "127.0.0.2", 8123, False)]
