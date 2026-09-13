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
