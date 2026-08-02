from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from stock_simulator.config import GameConfig


def test_config_load_from_yaml_and_env_override(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
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


def test_maintenance_margin_fields_load_from_yaml_and_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # The maintenance-margin tier has to travel the same three-layer path as every
    # other field: dataclass default → YAML → STOCK_SIM_* env override (last wins).
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(["maintenance_margin_rate: 0.02", "maintenance_amount: 25.0"]),
        encoding="utf-8",
    )
    assert GameConfig.from_yaml(yaml_path).maintenance_margin_rate == 0.02

    monkeypatch.setenv("STOCK_SIM_MAINTENANCE_MARGIN_RATE", "0.04")
    cfg = GameConfig.load(yaml_path=yaml_path)
    assert cfg.maintenance_margin_rate == 0.04
    assert cfg.maintenance_amount == 25.0


def test_maintenance_margin_fields_round_trip_through_to_dict() -> None:
    cfg = GameConfig(maintenance_margin_rate=0.03, maintenance_amount=7.5)
    payload = cfg.to_dict()
    assert payload["maintenance_margin_rate"] == 0.03
    assert payload["maintenance_amount"] == 7.5
    assert GameConfig.from_mapping(payload) == cfg


def test_maintenance_margin_fields_change_the_config_hash() -> None:
    # `stable_hash` tags telemetry with the active config; a margin change must be
    # visible there, which it only is if the fields reach `to_dict`.
    baseline = GameConfig()
    assert baseline.stable_hash() != GameConfig(maintenance_margin_rate=0.01).stable_hash()
    assert baseline.stable_hash() != GameConfig(maintenance_amount=1.0).stable_hash()


def test_config_hash_is_stable() -> None:
    a = GameConfig(seed=1, fee_bps=4.0)
    b = GameConfig(seed=1, fee_bps=4.0)
    c = GameConfig(seed=2, fee_bps=4.0)
    assert a.stable_hash() == b.stable_hash()
    assert a.stable_hash() != c.stable_hash()
