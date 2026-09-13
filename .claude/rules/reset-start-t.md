---
paths:
  - "src/stock_simulator/env.py"
  - "src/stock_simulator/server.py"
  - "src/stock_simulator/grpc/server.py"
  - "proto/stocksim_grpc/engine.proto"
---

## Episode reset with `start_t`

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
