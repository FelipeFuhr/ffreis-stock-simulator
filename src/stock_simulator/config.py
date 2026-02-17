from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints


@dataclass
class GameConfig:
    use_numba: bool = False
    observation_window: int = 64
    max_open_orders: int = 64

    initial_cash: float = 100_000.0
    max_leverage: float = 3.0
    delta_exposure: float = 0.25

    fee_bps: float = 4.0
    slippage_bps: float = 1.0

    market_latency_bars: int = 1
    limit_ttl_bars: int = 10

    partial_fill_min: float = 0.3
    partial_fill_max: float = 1.0

    shock_prob: float = 0.0
    shock_size_bps: float = 500.0

    seed: int = 1234

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: str | Path) -> GameConfig:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyYAML is required for YAML config loading. Install pyyaml."
            ) from exc

        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("YAML config must decode to a key/value mapping")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> GameConfig:
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        return cls(**raw)

    @classmethod
    def from_env(cls, prefix: str = "STOCK_SIM_") -> dict[str, Any]:
        values: dict[str, Any] = {}
        hints = get_type_hints(cls)
        for item in fields(cls):
            env_key = f"{prefix}{item.name}".upper()
            raw = os.getenv(env_key)
            if raw is None:
                continue
            expected_type = hints.get(item.name, str)
            values[item.name] = _coerce_env_value(raw, expected_type)
        return values

    @classmethod
    def load(
        cls,
        *,
        yaml_path: str | Path | None = None,
        env_prefix: str = "STOCK_SIM_",
    ) -> GameConfig:
        config = cls()
        if yaml_path is not None:
            config = cls.from_yaml(yaml_path)
        env_values = cls.from_env(prefix=env_prefix)
        if env_values:
            base = config.to_dict()
            base.update(env_values)
            config = cls.from_mapping(base)
        return config


def _coerce_env_value(raw: str, expected_type: Any) -> Any:
    value = raw.strip()
    if expected_type is bool:
        lowered = value.lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw}")
    if expected_type is int:
        return int(value)
    if expected_type is float:
        return float(value)
    return value
