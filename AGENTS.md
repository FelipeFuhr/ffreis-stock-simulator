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

- **Margin is TWO-TIER, exactly as on a real exchange — don't conflate the tiers.**
  `max_leverage` is the **initial margin**: it governs whether an order may be
  OPENED/INCREASED and is evaluated *once, at order time* (`clip_units_for_leverage`,
  next bullet). `maintenance_margin_rate` / `maintenance_amount` are the **maintenance
  margin**: a separate, far more permissive threshold governing whether an
  ALREADY-OPEN position is force-closed, re-evaluated at *every* mark
  (`settle_maintenance_margin`). The formula is one bracket of an exchange's
  maintenance table — `maintenance_margin = abs(notional) * rate - amount`, floored at
  zero — and liquidation triggers when margin balance (this engine's `equity`: cash
  plus unrealized PnL at the mark) drops below it.
  **The consequence to internalize: leverage drifting well above `max_leverage`
  between fills is NORMAL, REAL exchange behavior, not a bug and not something to
  suppress by continuously re-clipping.** A book opened at the 2.0x cap and marked
  down 40% carries 6.0x and is left alone — it is still ~33x above its 0.5%
  maintenance requirement. What a real exchange does instead is intervene *earlier
  than total wipeout* through this second, more sensitive check, closing the account
  out with **positive** residual equity (worked example: 20 units against −1000 cash,
  marked at 50.125 → equity +2.50 vs a 5.0125 requirement → liquidated holding $2.50).
  Before this existed, `equity <= 0` was the only trigger, which is why liquidation
  was near-unreachable at 3x on 1h bars.
  Defaults (`rate = 0.005`, `amount = 0.0`) are a **documented approximation** of
  Binance USDⓈ-M BTCUSDT-perpetual's lowest bracket, **not live-synced** with any
  exchange — Binance renders its bracket table dynamically and revises it over time.
  Both are ordinary `GameConfig` fields and overridable per run.
  Deliberate, honest simplifications (do not read these as oversights):
  **single flat tier** — `config.MAINTENANCE_MARGIN_TIERS` is a one-row table typed as
  `MaintenanceMarginTier(notional_floor, rate, amount)` so a real multi-bracket table
  can be added by extending the tuple and selecting a row by notional, without
  reshaping the formula; multi-tier brackets exist to make very large positions harder
  to hold, which is meaningless with one participant. **Full-close on trigger, never
  partial/graduated liquidation** — real exchanges liquidate in tranches specifically
  to limit market impact on a shared order book; this simulator has no other
  participants and no market-impact model, so a full close is the correct
  simplification. **No separate liquidation fee** — Binance's published liquidation
  formula does not define one as its own line item, so none is invented here.

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

- **Episodes terminate on liquidation, not only at end of data.** `done` used to be
  purely `next_t >= n - 1`, so an episode kept executing steps against negative equity
  (an RL walk-forward run saw equity from −$86k to +$20.4M on a $100k account, with 91
  of 99 episodes ending insolvent while `done` stayed `False`). Margin is now settled at
  **two points per step** — after the bar's fills, and after the mark-to-market move
  into the next bar — and at each point **two checks** run in order:
  `settle_maintenance_margin` (the primary trigger; normally fires with equity still
  positive) then `settle_insolvency` (`equity <= 0`). `settle_insolvency` is kept
  deliberately as the deep-tail backstop: a single volatile bar can gap clean past the
  maintenance threshold and land below zero in one move — real, and not catchable by
  the maintenance check. Either trigger force-closes the whole position at that price
  via the shared `liquidate_position` (units zeroed, PnL realized into cash, so equity
  equals cash exactly and cannot drift from a stale unit count) and sets `done = True`.
  The two triggers differ only in *when* they fire, never in what they do.

- **Both engines implement margin identically — keep them in sync.** Every margin
  helper exists twice: `clip_units_for_leverage`, `maintenance_margin`,
  `is_below_maintenance_margin`, `liquidate_position`, `settle_maintenance_margin`,
  `settle_insolvency` for the pure-Python `step_core`, each with a
  `_<name>_jit` mirror for `step_core_jit`. The bodies are statement-for-statement
  identical, and the comparisons are written multiplied-through
  (`abs(u) * price <= cap * equity`) instead of as divisions so both produce
  bit-identical floats. `step_core_jit` takes `max_leverage`,
  `maintenance_margin_rate` and `maintenance_amount` as parameters — **both** call
  sites (`step_core_numba` and `MarketEnv._advance_one_step`) must pass all three.
  Parity is pinned by an AST comparison of each mirror pair
  (`TestNumbaMirrorsAreIdenticalSource`, which normalizes *only* the `_jit` helper
  names a delegating mirror must call — see its `_JIT_ALIASES`) plus exact-equality
  scenarios in `tests/unit_tests/test_margin_enforcement.py::TestNumbaParity` and
  `tests/unit_tests/test_use_numba_parity.py`.

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
  `proto/stocksim_grpc/engine.proto` (`make grpc-generate`). The generated
  `*_pb2.py`/`*_pb2_grpc.py` files are gitignored, **not** checked in — every
  fresh checkout/CI run regenerates them from the proto; `make ci`'s `grpc-check`
  step does this automatically before lint/test.

- **Replay format is `RecordedStep` via the `Recorder` port (`recorder.py`).** The
  RL agent reads this for offline training. Changing the schema is a breaking
  change for the agent. The only concrete writer today is `ParquetRecorder`
  (Parquet, not JSONL) — `Recorder` is wired into the singular `MarketEnv.step`
  call only; `step_many` (the RL agent's bulk HTTP path) deliberately bypasses it
  for headless-execution throughput. `STOCK_SIM_TRACE_JSONL` (below) is a
  separate, additive mechanism for exactly that gap — do not conflate the two.

- **`StepTraceRow.filled_order_slot`/`.exec_price` (N9).** The submitted action's
  own `limit_price` is *not* the actually-filled order's price — an older queued
  limit order can fill on a step whose submitted action was itself a `hold` (or a
  different order). `filled_order_slot`/`exec_price` recover the real fill: which
  order slot filled and at what execution price, populated only when `fills > 0`
  this step (`None`/absent otherwise). Both engines (`step_core`/`_match_orders`
  and `step_core_jit`) compute and thread these through identically — parity is
  pinned by `test_margin_enforcement.py::TestNumbaParity` alongside the existing
  clip/liquidation fields. `StepTraceRow` is only ever constructed in
  `env.py::MarketEnv._build_trace_row` (the `step_many` wrapper layer) — neither
  `step_core` nor `step_core_jit` builds trace rows themselves, so this was a
  pure-Python-layer plumbing fix threading two already-computed values one hop
  further, not a new computation.

- **`STOCK_SIM_TRACE_JSONL` (N10, `trace_writer.py`).** Server-startup-time env var
  (read directly in `server.py::_load_engine`, like `STOCK_SIM_CONFIG_YAML` —
  deliberately *not* a `GameConfig` field, so it never perturbs
  `GameConfig.stable_hash()`). When set to a file path, every `step_many` step's
  full trace row is appended to that file as JSONL, regardless of any individual
  request's `include_trace` flag — a debugging/verification aid for the fact that
  `include_trace` is per-request and client-opted, so traces are otherwise
  invisible unless you own the client code. Distinct from the `Recorder`/JSONL
  point above — see that bullet. The sink is written to **only** from
  `MarketEnv.step_many` — the singular `MarketEnv.step` never references
  `self._trace_sink` at all (it already has the `Recorder` port for that job).
  A script or test that wants trace output from a configured sink must drive
  the episode through `step_many` (even with single-action batches), not `step`.

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

- **`examples/margin_scenario.py`** — narrated, human-runnable walkthrough of the
  whole margin system over synthetic markets, in two scenarios: (1) fill clipping
  (initial margin) + the `equity <= 0` backstop catching a violent gap +
  `STOCK_SIM_TRACE_JSONL` trace recording; (2) leverage drifting to 6.0x on a 3.0x
  cap **without** anything firing, then a maintenance-margin liquidation closing the
  book at **+$2.50** equity. Read scenario 2 first if the two-tier model is the thing
  you are trying to understand. No `make examples`
  target exists (nothing else under `examples/` is wired to `make` either —
  `smoke_api_grpc.py` runs via the separate `make smoke-api-grpc` docker-compose
  target). Run it directly: `uv run python examples/margin_scenario.py` — no
  extras needed, it only touches `stock_simulator.env.MarketEnv`, not the
  HTTP/gRPC transports. See `tests/integration_tests/test_margin_lifecycle.py`
  for the same scenario shape pinned with exact assertions through the real
  `/v1/step_many` HTTP surface.

## Keeping this file current

- **If you discover a fact not reflected here:** add it before finishing your task.
- **If something here is wrong or outdated:** correct it in the same commit as the code change.
- **If you rename a file, command, or concept referenced here:** update the reference.
