from __future__ import annotations

from pathlib import Path

from stock_simulator.config import GameConfig


def test_config_load_from_yaml_and_env_override(monkeypatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "use_numba: false",
                "initial_cash: 250000.0",
                "max_open_orders: 32",
                "fee_bps: 2.0",
                "seed: 777",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("STOCK_SIM_USE_NUMBA", "true")
    monkeypatch.setenv("STOCK_SIM_MAX_OPEN_ORDERS", "96")

    cfg = GameConfig.load(yaml_path=yaml_path)
    assert cfg.use_numba is True
    assert cfg.max_open_orders == 96
    assert cfg.initial_cash == 250000.0
    assert cfg.seed == 777


def test_config_hash_is_stable() -> None:
    a = GameConfig(seed=1, fee_bps=4.0)
    b = GameConfig(seed=1, fee_bps=4.0)
    c = GameConfig(seed=2, fee_bps=4.0)
    assert a.stable_hash() == b.stable_hash()
    assert a.stable_hash() != c.stable_hash()
