---
paths:
  - "src/stock_simulator/recorder.py"
  - "src/stock_simulator/trace_writer.py"
  - "src/stock_simulator/env.py"
  - "src/stock_simulator/server.py"
---

## Trace rows and replay recording

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
