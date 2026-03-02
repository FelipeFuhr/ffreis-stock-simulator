from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from stock_simulator.config import GameConfig, _coerce_env_value


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    try:
        GameConfig.from_yaml(path)
        raise AssertionError("expected ValueError for non-mapping YAML")
    except ValueError as exc:
        assert "key/value mapping" in str(exc)


def test_from_yaml_treats_empty_as_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    cfg = GameConfig.from_yaml(path)
    assert cfg == GameConfig()


def test_from_mapping_rejects_unknown_keys() -> None:
    try:
        GameConfig.from_mapping({"not_a_field": 1})
        raise AssertionError("expected ValueError for unknown keys")
    except ValueError as exc:
        assert "Unknown config keys" in str(exc)


def test_coerce_env_value_variants() -> None:
    assert _coerce_env_value("yes", bool) is True
    assert _coerce_env_value("0", bool) is False
    assert _coerce_env_value(" 7 ", int) == 7
    assert _coerce_env_value(" 7.5 ", float) == 7.5
    assert _coerce_env_value("abc", str) == "abc"
    try:
        _coerce_env_value("not-bool", bool)
        raise AssertionError("expected invalid bool to raise")
    except ValueError as exc:
        assert "Invalid boolean value" in str(exc)


def test_load_without_yaml_uses_env_values(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("STOCK_SIM_SEED", "8080")
    monkeypatch.setenv("STOCK_SIM_USE_NUMBA", "true")
    cfg = GameConfig.load(yaml_path=None)
    assert cfg.seed == 8080
    assert cfg.use_numba is True
