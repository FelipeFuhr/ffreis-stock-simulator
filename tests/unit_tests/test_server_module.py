from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient
from numpy import asarray as np_asarray
from numpy import bool_ as np_bool_
from numpy import float64 as np_float64
from pytest import MonkeyPatch

from stock_simulator import server as server_mod
from stock_simulator.types import EnvState, MarketWindowViewHandle, Observation, StepTraceRow

_TINY_MARKET_CSV = (
    "timestamp,open,high,low,close,volume\n"
    "2024-01-01T00:00:00,100.0,100.5,99.5,100.0,10000\n"
    "2024-01-01T01:00:00,100.0,100.5,99.5,100.0,10000\n"
    "2024-01-01T02:00:00,100.0,100.5,99.5,100.0,10000\n"
    "2024-01-01T03:00:00,100.0,100.5,99.5,100.0,10000\n"
)


@dataclass
class _FakeEnv:
    raise_on_step: bool = False
    raise_on_reset: bool = False
    last_reset_kwargs: dict[str, object] = field(default_factory=dict)
    closed: bool = False

    def close(self) -> None:
        self.closed = True

    def reset(self, seed: int | None = None, start_t: int = 0) -> EnvState:
        self.last_reset_kwargs = {"seed": seed, "start_t": start_t}
        if self.raise_on_reset:
            raise ValueError(f"start_t must satisfy 0 <= start_t < 10 (market has 10 bars); got {start_t}")
        return EnvState(
            t=start_t,
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

    def step_many(
        self, _actions: object, *, include_trace: bool = False
    ) -> tuple[dict[str, object], object, object, tuple[StepTraceRow, ...]]:
        if self.raise_on_step:
            raise ValueError("invalid order_type code")
        observations: dict[str, object] = {
            "market_window_handle": np_asarray([[0.0, 64.0, 4.0, 102.0]], dtype=np_float64),
            "portfolio_vector": np_asarray([[999.0, 1.0, 1101.0, 0.2]], dtype=np_float64),
            "order_summary_vector": np_asarray([[1.0, 1.0, 0.0]], dtype=np_float64),
        }
        rewards = np_asarray([101.0], dtype=np_float64)
        dones = np_asarray([False], dtype=np_bool_)
        trace_rows: tuple[StepTraceRow, ...] = (
            (
                StepTraceRow(
                    index=0,
                    side_code=1,
                    requested_units=1.0,
                    order_type_code=0,
                    has_limit_price=False,
                    limit_price=None,
                    fills=1,
                    reward=101.0,
                    done=False,
                    t=4,
                    cash=999.0,
                    position_units=1.0,
                    equity=1101.0,
                    leverage=0.2,
                    open_orders=1,
                    market_price=102.0,
                    has_filled_order=True,
                    filled_order_slot=0,
                    exec_price=102.0,
                ),
            )
            if include_trace
            else ()
        )
        return observations, cast(object, rewards), cast(object, dones), trace_rows


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


def test_load_engine_wires_trace_sink_when_env_var_set(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """N10: STOCK_SIM_TRACE_JSONL makes _load_engine's env write every step's
    trace to the given path, even though this test never sets include_trace.
    """
    market_csv = tmp_path / "market.csv"
    market_csv.write_text(_TINY_MARKET_CSV, encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("MARKET_DATA_CSV", str(market_csv))
    monkeypatch.setenv("STOCK_SIM_TRACE_JSONL", str(trace_path))

    env = server_mod._load_engine()
    try:
        env.reset(seed=1)
        actions = np_asarray([[0.0, 0.0, 0.0, float("nan")]], dtype=np_float64)  # a single "hold"
        env.step_many(actions)  # include_trace defaults to False on this call
    finally:
        env.close()

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_load_engine_leaves_trace_sink_unset_by_default(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Unsetting STOCK_SIM_TRACE_JSONL leaves default behavior unchanged: no file."""
    market_csv = tmp_path / "market.csv"
    market_csv.write_text(_TINY_MARKET_CSV, encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("MARKET_DATA_CSV", str(market_csv))
    monkeypatch.delenv("STOCK_SIM_TRACE_JSONL", raising=False)

    env = server_mod._load_engine()
    env.reset(seed=1)
    actions = np_asarray([[0.0, 0.0, 0.0, float("nan")]], dtype=np_float64)
    env.step_many(actions)

    assert not trace_path.exists()


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
        assert empty_step.json() == {"observations": [], "rewards": [], "dones": [], "trace": []}

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
        assert payload["trace"] == []

        with_trace = client.post(
            "/v1/step_many",
            json={
                "include_trace": True,
                "actions": [
                    {
                        "side_code": 1,
                        "units": 1.0,
                        "order_type_code": 0,
                        "has_limit_price": False,
                    }
                ],
            },
        )
        assert with_trace.status_code == 200
        trace_payload = with_trace.json()["trace"]
        assert len(trace_payload) == 1
        # N9: the actually-filled order's slot/price ride alongside the
        # submitted action's own (here-empty) limit_price field.
        assert trace_payload[0]["filled_order_slot"] == 0
        assert trace_payload[0]["exec_price"] == 102.0


def test_create_app_reset_default_start_t_is_zero(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_ENABLED", "true")
    fake_env = _FakeEnv()
    monkeypatch.setattr(server_mod, "_load_engine", lambda: fake_env)
    app = server_mod.create_app()

    with TestClient(app) as client:
        reset = client.post("/v1/reset", json={"seed": 7})
        assert reset.status_code == 200
        assert reset.json()["state"]["t"] == 0
        assert fake_env.last_reset_kwargs == {"seed": 7, "start_t": 0}


def test_create_app_reset_passes_start_t_to_engine(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_ENABLED", "true")
    fake_env = _FakeEnv()
    monkeypatch.setattr(server_mod, "_load_engine", lambda: fake_env)
    app = server_mod.create_app()

    with TestClient(app) as client:
        reset = client.post("/v1/reset", json={"seed": 7, "start_t": 2000})
        assert reset.status_code == 200
        assert reset.json()["state"]["t"] == 2000
        assert fake_env.last_reset_kwargs == {"seed": 7, "start_t": 2000}


def test_create_app_reset_maps_invalid_start_t_to_http_400(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_ENABLED", "true")
    monkeypatch.setattr(server_mod, "_load_engine", lambda: _FakeEnv(raise_on_reset=True))
    app = server_mod.create_app()

    with TestClient(app) as client:
        response = client.post("/v1/reset", json={"start_t": 999})
        assert response.status_code == 400
        assert "start_t" in response.json()["detail"]


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
