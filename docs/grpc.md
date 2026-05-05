# gRPC Interface

The simulator exposes gRPC in addition to HTTP (FastAPI).

Server reflection is intentionally not enabled in runtime paths.

## Install

```bash
uv sync --extra grpc
```

## Run server

```bash
uv run --extra grpc stock-simulator-grpc --host 0.0.0.0 --port 50051 --use-numba
```

## API surface

- `Ping` -> returns `"pong"`
- `Reset` -> resets environment (optional seed)
- `Observe` -> current observation
- `StepMany` -> batched encoded actions, returns observations/rewards/dones (+ optional per-step `trace`)

`StepManyRequest` supports `include_trace=true` to return one `StepTraceRow` per processed action
with action, fill count, reward, done flag, and post-step portfolio snapshot fields.

Proto contract:

- `proto/stocksim_grpc/engine.proto`

Generated modules:

- `src/stocksim_grpc/engine_pb2.py`
- `src/stocksim_grpc/engine_pb2_grpc.py`

These files are generated on demand and are not committed.

## Sync checks

Regenerate:

```bash
make grpc-generate
```

Verify generated files are in sync with proto:

```bash
make grpc-check
```

Remove generated stubs:

```bash
make grpc-clean
```
