# Agent Context

**This repo:** `ffreis-stock-simulator` — Python stock market simulation environment
with a Numba-JIT order book, exposed via HTTP (FastAPI) and gRPC.

## Non-obvious facts

- **Simulation is deterministic.** Same seed produces identical outcomes. Never
  introduce randomness without seeding. The RL agent relies on reproducibility for
  training.

- **Episodes can start at an arbitrary bar via `reset(start_t=...)`.** Defaults to
  `0` (unchanged prior behavior — every existing caller that omits `start_t` sees
  identical episodes). This exists because a hardcoded `t=0` on every reset meant
  a downstream RL agent's post-reset warm-up fetch (`min(1500, t+1)` bars via
  `/v1/market_window`) only ever returned 1 bar, leaving the slowest EMA feature
  inert for the first ~250 steps of every episode. `start_t` must satisfy
  `0 <= start_t < n` (`n` = loaded market bar count) or `reset` raises
  `ValueError` — mapped to HTTP 400 on `/v1/reset`, and to gRPC
  `INVALID_ARGUMENT` on the `Reset` RPC. Determinism holds per `(seed, start_t)`
  pair — `reset(seed=S, start_t=T)` called twice produces byte-identical
  episodes. Present on both transports: HTTP's `ResetRequest.start_t` (optional
  int) and gRPC's `ResetRequest.has_start_t`/`start_t` (explicit-presence pair,
  mirroring the existing `has_seed`/`seed` fields) — see
  `proto/stocksim_grpc/engine.proto`.

- **Feature vector shape is 11 features.** This is enforced by `ml/ffreis-integration-hub`
  smoke tests. Changing the observation space breaks the RL agent without a compile error.

- **Generated gRPC stubs in `src/stocksim_grpc/` — DO NOT EDIT.** Regenerate from
  `proto/stocksim_grpc/simulator.proto`. Stubs are checked in for CI compatibility.

- **Replay format is JSONL with `RecordedStep` schema.** The RL agent reads this for
  offline training. Changing the schema is a breaking change for the agent.

- **Both HTTP and gRPC are production transports** — not alternatives. The RL agent
  uses HTTP; the integration hub tests gRPC.

- **Required make targets** (called by integration-hub): `test-grpc-parity`.

- **Coverage minimum 80%.** Test markers: `unit`, `integration`, `e2e`, `property`
  (Hypothesis).

- **`uv.lock` is required** — `uv sync --frozen` used in CI and Docker.

## Structure

```
src/stock_simulator/    ← simulation engine + FastAPI server + recorder
src/stocksim_grpc/      ← generated stubs (DO NOT EDIT)
proto/                  ← canonical proto
```

## Build/test

```bash
uv sync && make test && make run
make test-grpc-parity       # called by integration-hub
docker build -t ffreis-stock-simulator .
```

## Keeping this file current

- **If you discover a fact not reflected here:** add it before finishing your task.
- **If something here is wrong or outdated:** correct it in the same commit as the code change.
- **If you rename a file, command, or concept referenced here:** update the reference.
