## Coverage requirements

- **Coverage minimum 90%** (`fail_under` in `pyproject.toml`; `make coverage` measures
  `src` over `tests/unit_tests`; also enforced in CI via `coverage-min: 90` on
  `.github/workflows/tests.yml`'s `python-test.yml` call). Test markers: `unit`,
  `integration`, `e2e`, `property` (Hypothesis).

- **Integration coverage floor is 70%, just under the fleet's 75% aspiration.**
  `.github/workflows/integration.yml` (hand-rolled, not the shared
  `python-test.yml`'s `integration-coverage-min` input — it predates that input
  and already had its own runner/gRPC-stub-check steps) gates `tests/integration_tests`
  at `--cov-fail-under=70`; measured actual was 74.37% on 2026-08-08 (a few
  points of margin kept below that for branch-coverage noise). `make
  coverage-integration` mirrors it locally. Closing the remaining ~5 points to
  75% needs a bit more coverage of `telemetry.py`/`recorder.py`/`grpc/server.py`
  (54%/44%/63% respectively in the integration-only run) — not done here,
  flagged as follow-up.
