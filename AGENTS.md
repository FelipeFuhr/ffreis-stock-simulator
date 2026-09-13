# Agent Context

**This repo:** `ffreis-stock-simulator` — Python stock market simulation environment
with a Numba-JIT order book, exposed via HTTP (FastAPI) and gRPC.

## Where the rest lives

This file used to carry every section below inline; it now only routes to
them. Nothing was dropped — each heading/bullet from the original file lives
verbatim in exactly one file below. `.claude/rules/*.md` sections auto-load
only when a matching file path is touched; `.claude/reference/*.md` sections
load on demand, by name. `.claude/reference/_manifest.json` tracks which
section landed in which file.

| Section | Loads when | File |
| --- | --- | --- |
| Non-obvious facts (heading — split into the rows below) | — | this file (routing table) |
| Simulation is deterministic | on demand | `.claude/reference/determinism.md` |
| Episode reset with `start_t` | touching `env.py`, `server.py`, `grpc/server.py`, or `proto/stocksim_grpc/engine.proto` | `.claude/rules/reset-start-t.md` |
| Margin and leverage enforcement (two-tier margin, fill-time leverage clip, liquidation-on-termination, engine parity, finite leverage reporting) | touching `core.py`, `config.py`, `portfolio.py`, or `env.py` | `.claude/rules/margin-and-leverage.md` |
| Feature vector shape | touching `types.py` or `orders.py` | `.claude/rules/feature-vector-shape.md` |
| Generated gRPC stubs — DO NOT EDIT | touching `src/stocksim_grpc/**` or `proto/stocksim_grpc/engine.proto` | `.claude/rules/grpc-generated-stubs.md` |
| Trace rows and replay recording (`Recorder`, `StepTraceRow` N9, `STOCK_SIM_TRACE_JSONL` N10) | touching `recorder.py`, `trace_writer.py`, `env.py`, or `server.py` | `.claude/rules/trace-and-replay.md` |
| Transports and required make targets | on demand | `.claude/reference/transports-and-make-targets.md` |
| Coverage requirements | on demand | `.claude/reference/coverage-requirements.md` |
| `mutmut` version pin | touching `pyproject.toml`, `.github/workflows/mutation.yml`, or `Makefile` | `.claude/rules/mutmut-pin.md` |
| `uv.lock` | on demand | `.claude/reference/uv-lock.md` |
| Structure | on demand | `.claude/reference/structure.md` |
| Build/test | on demand | `.claude/reference/build-and-test.md` |
| Keeping this file current | on demand | `.claude/reference/keeping-this-file-current.md` |

`bash scripts/check-instructions.sh` (also `make lint-instructions`) verifies
every section in `_manifest.json` resolves to a non-empty file and every rule
file carries valid `paths:` frontmatter.
