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

- **`max_leverage` is ENFORCED at fill time — this changed pre-existing behavior.**
  `GameConfig.max_leverage` (default `3.0`) used to be parsed and never read: `core.py`
  had no fill-time check, so an agent could accumulate unbounded leverage. It is now a
  hard cap. Before a fill is applied, `clip_units_for_leverage` (`core.py`) computes the
  leverage the book would carry after the fill — `leverage_ratio(units_after, exec_price,
  equity_after)`, the same expression `portfolio.py` reports — and **clips the fill size
  down** to the largest size that stays at or under the cap. It is one more upper bound
  on the same fill-sizing computation as the `partial_fill_min`/`partial_fill_max` draw,
  not a separate rejection path: the order still fills, just smaller, and an order that
  cannot add any exposure fills for **zero units** (counted as a fill, order cleared, no
  error). Orders that shrink gross exposure are **never** clipped, even from above the
  cap — de-risking always goes through. **Callers that relied on the old unbounded
  behavior will now see smaller fills than they requested**; compare `EnvState.units`
  against the requested size rather than assuming full fills.

- **Episodes terminate on insolvency, not only at end of data.** `done` used to be
  purely `next_t >= n - 1`, so an episode kept executing steps against negative equity
  (an RL walk-forward run saw equity from −$86k to +$20.4M on a $100k account, with 91
  of 99 episodes ending insolvent while `done` stayed `False`). `settle_insolvency`
  (`core.py`) now runs twice per step — once after the bar's fills, once after the
  mark-to-market move into the next bar — and when equity is non-positive it force-closes
  the whole position at that price (units zeroed, loss realized into cash, so equity
  equals cash exactly and cannot drift further from a stale unit count) and sets
  `done = True`. Both causes are covered: a fill on the current bar and a losing
  position's mark-to-market drop.

- **Both engines implement margin identically — keep them in sync.** The clip and the
  liquidation exist twice: `clip_units_for_leverage`/`settle_insolvency` for the
  pure-Python `step_core`, and `_clip_units_for_leverage_jit`/`_settle_insolvency_jit`
  for `step_core_jit`. The bodies are statement-for-statement identical, and the
  comparisons are written multiplied-through (`abs(u) * price <= cap * equity`) instead
  of as divisions so both produce bit-identical floats. `step_core_jit` takes
  `max_leverage` as a parameter — **both** call sites (`step_core_numba` and
  `MarketEnv._advance_one_step`) must pass it. Parity is pinned by
  `tests/unit_tests/test_margin_enforcement.py::TestNumbaParity` (exact equality, not
  approx) and `tests/unit_tests/test_use_numba_parity.py`.

- **Reported `leverage` is always finite.** `abs(units * price) / equity` is unbounded
  when an open position is carried against non-positive equity, and the old `inf` there
  serialized to JSON `null`, crashing clients with a `TypeError`. A flat book now reports
  `0.0` (no exposure, nothing to lever, whatever the sign of equity), and the unbounded
  case is capped at the finite `portfolio.INSOLVENT_LEVERAGE` (`1e9`) sentinel. With
  insolvency liquidation in place that ceiling is unreachable through `step`/`observe` —
  it only guards direct `snapshot_from_state` calls on hand-built states — but the value
  stays finite so no transport can emit `null` for `leverage`. Response schemas are
  unchanged (`leverage` is still `number`), so `docs/openapi.yaml` needs no update.

- **Feature vector shape is 11 features.** This is enforced by `ml/ffreis-integration-hub`
  smoke tests. Changing the observation space breaks the RL agent without a compile error.

- **Generated gRPC stubs in `src/stocksim_grpc/` — DO NOT EDIT.** Regenerate from
  `proto/stocksim_grpc/simulator.proto`. Stubs are checked in for CI compatibility.

- **Replay format is JSONL with `RecordedStep` schema.** The RL agent reads this for
  offline training. Changing the schema is a breaking change for the agent.

- **Both HTTP and gRPC are production transports** — not alternatives. The RL agent
  uses HTTP; the integration hub tests gRPC.

- **Required make targets** (called by integration-hub): `test-grpc-parity`.

- **Coverage minimum 90%** (`fail_under` in `pyproject.toml`; `make coverage` measures
  `src` over `tests/unit_tests`). Test markers: `unit`, `integration`, `e2e`, `property`
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
