# Repository Audit Report (2026-02-16)

## Scope
- `ffreis-stock-simulator` (Python)
- `ffreis-python-onnx-model-converter` (Python)
- `ffreis-python-onnx-model-serving` (Python)
- `ffreis-rust-onnx-model-serving/app` (Rust)
- `app` (Rust)

## Checks performed
- Unused code signals: `ruff` (Python), `cargo clippy -- -D warnings` (Rust), targeted `rg` searches.
- Naming consistency: `Action` / `Observation` / `StepResult` usage and imports.
- Typing: `mypy` for stock simulator + focused manual checks.
- Dataclass defaults: scan for mutable defaults in dataclasses.
- Pandas in hot loop: scan around environment/core execution paths.

## Findings

### 1) `ffreis-stock-simulator`
- Naming consistency is now aligned on `Action`, `Observation`, `StepResult` across public API and engine.
- No mutable dataclass defaults found.
- Pandas is only used for I/O/data assembly (`data.py`, `recorder.py`), not in hot loop (`core.py`, `env.step`, `step_core`, `step_core_jit`).
- Implemented typing fixes:
  - `src/stock_simulator/env.py`: explicit `Side`/`OrderType` typing in action decoding.
  - `tests/unit_tests/test_invariants.py`: explicit `Side`/`OrderType` strategy typing.
  - `src/stock_simulator/recorder.py`: made parquet replay row conversion type-safe for mypy.
  - `pyproject.toml`: added `pandas-stubs`, `types-PyYAML`; kept strict checks and disabled `import-untyped` (needed for untyped libs such as `numba`).
- Validation status:
  - `ruff`: pass
  - `mypy`: pass (`Success: no issues found in 24 source files`)

### 2) `ffreis-python-onnx-model-converter`
- No `Action`/`Observation`/`StepResult` naming concerns (not part of this domain model).
- No pandas usage detected in `src/`.
- No mutable dataclass defaults detected in `src/` scan.
- Validation status:
  - `ruff`: pass (`uv run --no-project --with ruff ruff check src`)
  - `mypy`: pass (`uv run --no-project --with mypy mypy src`)
  - Note: standard `uv run` is currently blocked by a dependency resolution conflict between project extras (`all` vs `autosklearn`).

### 3) `ffreis-python-onnx-model-serving`
- No `Action`/`Observation`/`StepResult` naming concerns (different domain).
- No pandas usage detected in `src/`.
- No mutable dataclass defaults detected in `src/` scan.
- Validation status:
  - `ruff`: pass
  - `mypy`: pass (`Success: no issues found in 11 source files`)

### 4) Rust repos (`app`, `ffreis-rust-onnx-model-serving/app`)
- `cargo clippy --all-targets -- -D warnings`: pass on both crates.
- No unused-code warnings surfaced by clippy under denied warnings.

## Changes applied
- `ffreis-stock-simulator/src/stock_simulator/env.py`
- `ffreis-stock-simulator/src/stock_simulator/recorder.py`
- `ffreis-stock-simulator/tests/unit_tests/test_invariants.py`
- `ffreis-stock-simulator/pyproject.toml`
