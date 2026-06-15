# ffreis-stock-simulator

<!-- ffreis-badges:start -->
[![CI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FelipeFuhr/ffreis-badges/main/badges/ffreis-stock-simulator/ci.json)](https://github.com/FelipeFuhr/ffreis-stock-simulator/actions)
<!-- ffreis-badges:end -->

A deterministic stock-market simulation environment for reinforcement-learning
agents. It replays OHLCV market data bar-by-bar, models a single-symbol order book
(market and limit orders with fees, slippage, partial fills, latency, and limit
TTL), and exposes a fixed 11-feature observation per step. The engine has a pure
Python path and a Numba-JIT path, and is served over both HTTP (FastAPI) and gRPC.
Same seed + same config always produces identical outcomes — the RL agent relies
on this reproducibility for training and offline replay.

## What it does

The core abstraction is `MarketEnv` (`src/stock_simulator/env.py`), a stateful
environment with a deliberately small public API: `reset`, `step`, `observe`, and a
batched `step_many`.

- **Market data** (`MarketData`, `data.py`) loads OHLCV time series from CSV with
  columns `timestamp, open, high, low, close, volume`.
- **Configuration** (`GameConfig`, `config.py`) sets engine dynamics and execution
  behavior: `observation_window` (default 64), `max_open_orders`, `initial_cash`,
  `max_leverage`, `delta_exposure`, `fee_bps`, `slippage_bps`, `market_latency_bars`,
  `limit_ttl_bars`, `partial_fill_min`/`partial_fill_max`, `shock_prob`/`shock_size_bps`,
  `use_numba`, and `seed`. Config loads from defaults, then an optional YAML file,
  then `STOCK_SIM_*` environment overrides (last wins). `stable_hash()` tags
  telemetry with the active config.
- **Actions** (`Action`, `types.py`): a `side` (`buy`/`sell`/`hold`), `units`, an
  `order_type` (`market`/`limit`), and a `limit_price` (required for non-hold limit
  orders). For batched/transport use, actions are encoded as
  `[side_code, units, order_type_code, limit_price_or_nan]` where `side_code` is
  `-1` (sell) / `0` (hold) / `1` (buy) and `order_type_code` is `0` (market) /
  `1` (limit).
- **Observation** (`Observation`, `types.py`): the **11-feature vector** consumed by
  the RL agent (enforced by `ml/ffreis-integration-hub` smoke tests) is the
  concatenation of three dense sub-vectors:
  1. market window handle (4): `start, end, t, current_price`
  2. portfolio vector (4): `cash, units, equity, leverage`
  3. order summary vector (3): `open_orders, buy_open_orders, sell_open_orders`

  Changing this shape silently breaks the agent — there is no compile-time check.
- **Recording / replay** (`recorder.py`): pluggable `Recorder` port with
  `NullRecorder` (default, no-op), `InMemoryRecorder`, and `ParquetRecorder`.
  Each `RecordedStep` captures `episode_id, step, seed, side, order_type, units,
  limit_price, fills, equity, leverage, price`. The RL agent reads this for offline
  training, so the schema is a cross-repo contract — changing it is a breaking change.
- **Telemetry** (`telemetry.py`): OpenTelemetry spans plus Prometheus metrics around
  steps, orders, fills, and episode ends.

Both HTTP and gRPC are first-class production transports (not alternatives): the RL
agent uses HTTP; the integration hub exercises gRPC.

**HTTP API** (`server.py`, requires the `api` extra):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness (engine enabled/ready) |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/v1/reset` | Reset the episode (optional `seed`) |
| `GET` | `/v1/observe` | Current observation |
| `POST` | `/v1/step_many` | Batched encoded actions; returns observations, rewards, dones, optional trace |

The contract is documented in `docs/openapi.yaml` (validated by `make openapi-check`).

**gRPC** (`grpc/server.py`, requires the `grpc` extra): listens on `:50051` by
default (`GRPC_HOST`/`GRPC_PORT`). The canonical schema is
`proto/stocksim_grpc/engine.proto`; the generated stubs in `src/stocksim_grpc/` are
checked in and **must not be edited by hand** — regenerate them with
`make grpc-generate` and verify with `make grpc-check`.

## Usage

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). The package ships console
scripts `stock-simulator` (HTTP) and `stock-simulator-grpc` (gRPC).

```bash
# Install with dev + API + gRPC extras (uses the frozen lock file)
make install            # uv sync --frozen --extra dev --extra api --extra grpc
```

Run the HTTP service (the engine loads from environment configuration):

```bash
export MARKET_DATA_CSV=/path/to/ohlcv.csv      # required when ENGINE_ENABLED=true
export STOCK_SIM_CONFIG_YAML=/path/to/cfg.yaml # optional GameConfig overrides
export HOST=0.0.0.0 PORT=8000                  # optional
stock-simulator
```

Run the gRPC service:

```bash
stock-simulator-grpc --host 0.0.0.0 --port 50051 [--use-numba] [--seed 1234]
```

Library use:

```python
from stock_simulator import MarketData, GameConfig, MarketEnv

env = MarketEnv(data=MarketData.from_csv("ohlcv.csv"), cfg=GameConfig(seed=1234))
state = env.reset(seed=1234)
obs = env.observe()
```

Container builds (`container/Containerfile`) and a combined HTTP+gRPC smoke setup are
also provided:

```bash
docker build -t ffreis-stock-simulator -f container/Containerfile .
make smoke-api-grpc     # docker-compose HTTP + gRPC smoke test
```

## Development

```bash
make help          # list all targets
make fmt           # ruff format
make lint          # ruff check
make typecheck     # mypy (strict) — alias: make validate
make test          # full pytest suite
make ci            # grpc-check + openapi-check + lint + typecheck + test
```

Test markers (`pyproject.toml`): `unit`, `integration`, `e2e`, `property`
(Hypothesis); per-suite targets `test-unit`, `test-integration`, `test-e2e`.
Coverage is branch-based with a configured floor (see `pyproject.toml` /
`scripts/check_coverage.py`). The `test-grpc-parity` target is required by
`ml/ffreis-integration-hub` and asserts HTTP/gRPC behavioral parity.

Notable invariants when contributing:

- **Determinism** — never introduce unseeded randomness; the RL agent depends on it.
- **11-feature observation** — do not change the observation shape without updating
  the agent and the integration-hub compatibility check.
- **`RecordedStep` replay schema** and **encoded action layout** are cross-repo
  contracts; treat changes as breaking.
- **gRPC stubs are generated** — edit `proto/stocksim_grpc/engine.proto` and
  regenerate, never the files in `src/stocksim_grpc/`.
- `uv.lock` is required (`uv sync --frozen` is used in CI and the container build).

Linting runs Ruff (line length 120, NumPy docstring convention) and mypy in strict
mode; lefthook installs the standard pre-commit/commit-msg/pre-push hooks
(`make lefthook`).

## License

All Rights Reserved (proprietary). Copyright (c) Felipe Fuhr.
