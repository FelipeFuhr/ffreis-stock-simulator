"""Volume threading and the /v1/market_window content endpoint.

Covers the additive market-window surface: the engine accessor
(:meth:`MarketEnv.market_window`) and the HTTP endpoint that exposes raw
OHLCV+volume rows without touching the numba step core or the 11-feature
observation contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pandas import DataFrame
from pytest import MonkeyPatch

from stock_simulator import server as server_mod
from stock_simulator.config import GameConfig
from stock_simulator.data import MarketData
from stock_simulator.env import MarketEnv

_HOLD_ACTION = {"side_code": 0, "units": 0.0, "order_type_code": 0, "has_limit_price": False}


def _market_data(n: int = 24) -> MarketData:
    """Distinct per-column ramps + ramped volume so slice values are verifiable."""
    base = np.arange(n, dtype=np.float64)
    return MarketData(
        DataFrame(
            {
                "timestamp": [f"2024-01-01 {i:02d}:00:00" for i in range(n)],
                "open": base + 100.0,
                "high": base + 200.0,
                "low": base + 50.0,
                "close": base + 150.0,
                "volume": base + 1_000.0,
            }
        )
    )


def test_env_market_window_threads_volume() -> None:
    data = _market_data(n=24)
    env = MarketEnv(data=data, cfg=GameConfig(observation_window=4, use_numba=False))
    env.reset(seed=1)

    content = env.market_window()
    assert content.t == 0  # fresh episode
    assert content.start == 0
    assert content.end == 1
    # Volume threaded through unchanged from the source array.
    assert content.volume == pytest.approx(data.volume[content.start : content.end].tolist())
    assert content.close == pytest.approx(data.close[content.start : content.end].tolist())


def test_env_market_window_history_slice_matches_source() -> None:
    data = _market_data(n=24)
    env = MarketEnv(data=data, cfg=GameConfig(observation_window=4, use_numba=False))
    env.reset(seed=1)
    content = env.market_window(start=2, end=6)
    # t is 0 right after reset, so end clamps to t + 1 = 1 (no future leakage).
    assert content.end == 1
    assert content.start == 1  # start clamped up to end
    assert content.close == ()


def _client_with_engine(monkeypatch: MonkeyPatch, env: MarketEnv) -> TestClient:
    monkeypatch.setenv("ENGINE_ENABLED", "true")
    monkeypatch.setattr(server_mod, "_load_engine", lambda: env)
    return TestClient(server_mod.create_app())


def test_market_window_endpoint_aligns_with_observe(monkeypatch: MonkeyPatch) -> None:
    data = _market_data(n=24)
    env = MarketEnv(data=data, cfg=GameConfig(observation_window=4, use_numba=False))
    with _client_with_engine(monkeypatch, env) as client:
        client.post("/v1/reset", json={"seed": 5})
        # Advance a few bars so t > 0 and a full window is available.
        client.post("/v1/step_many", json={"actions": [_HOLD_ACTION] * 6})

        handle = client.get("/v1/observe").json()["observation"]["market_window_handle"]
        payload = client.get("/v1/market_window").json()

        assert payload["start"] == handle["start"]
        assert payload["end"] == handle["end"]
        assert payload["t"] == handle["t"]

        rows = payload["rows"]
        start, end = payload["start"], payload["end"]
        assert rows["open"] == pytest.approx(data.open[start:end].tolist())
        assert rows["high"] == pytest.approx(data.high[start:end].tolist())
        assert rows["low"] == pytest.approx(data.low[start:end].tolist())
        assert rows["close"] == pytest.approx(data.close[start:end].tolist())
        assert rows["volume"] == pytest.approx(data.volume[start:end].tolist())


def test_market_window_endpoint_explicit_bounds_no_future_leakage(monkeypatch: MonkeyPatch) -> None:
    data = _market_data(n=24)
    env = MarketEnv(data=data, cfg=GameConfig(observation_window=4, use_numba=False))
    with _client_with_engine(monkeypatch, env) as client:
        client.post("/v1/reset", json={"seed": 5})
        client.post("/v1/step_many", json={"actions": [_HOLD_ACTION] * 6})
        current_t = client.get("/v1/observe").json()["observation"]["market_window_handle"]["t"]

        # Warm-up request from index 0 far past the current bar.
        payload = client.get("/v1/market_window", params={"start": 0, "end": 10_000}).json()
        assert payload["start"] == 0
        assert payload["end"] == current_t + 1  # clamped: no index > t served
        rows = payload["rows"]
        assert len(rows["close"]) == current_t + 1
        assert rows["close"] == pytest.approx(data.close[0 : current_t + 1].tolist())
        assert rows["volume"] == pytest.approx(data.volume[0 : current_t + 1].tolist())


def test_market_window_endpoint_negative_start_clamps(monkeypatch: MonkeyPatch) -> None:
    data = _market_data(n=24)
    env = MarketEnv(data=data, cfg=GameConfig(observation_window=4, use_numba=False))
    with _client_with_engine(monkeypatch, env) as client:
        client.post("/v1/reset", json={"seed": 5})
        client.post("/v1/step_many", json={"actions": [_HOLD_ACTION] * 6})
        payload = client.get("/v1/market_window", params={"start": -20, "end": 3}).json()
        assert payload["start"] == 0
        assert payload["end"] == 3
        assert payload["rows"]["close"] == pytest.approx(data.close[0:3].tolist())


def test_market_window_endpoint_requires_engine(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_ENABLED", "false")
    with TestClient(server_mod.create_app()) as client:
        assert client.get("/v1/market_window").status_code == 503


def test_load_engine_sniffs_parquet_extension(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def _fake_from_parquet(_cls: type[MarketData], path: str) -> MarketData:
        captured["path"] = path
        return _market_data(n=8)

    monkeypatch.setenv("MARKET_DATA_CSV", "/data/btc.parquet")
    monkeypatch.delenv("STOCK_SIM_CONFIG_YAML", raising=False)
    monkeypatch.setattr(MarketData, "from_parquet", classmethod(_fake_from_parquet))

    engine = server_mod._load_engine()
    assert isinstance(engine, MarketEnv)
    assert captured["path"] == "/data/btc.parquet"
