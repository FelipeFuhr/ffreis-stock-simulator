.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash

IMAGE_PROVIDER ?=
IMAGE_PREFIX ?= ffreis
IMAGE_TAG ?= api-grpc-smoke
SMOKE_TIMEOUT ?= 20m
IMAGE_ROOT := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(IMAGE_PREFIX)

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Install project with dev and grpc extras
	uv sync --frozen --extra dev --extra grpc

.PHONY: grpc-generate
grpc-generate: ## Regenerate protobuf/gRPC stubs
	./scripts/generate_grpc_stubs.sh

.PHONY: grpc-check
grpc-check: ## Verify protobuf/gRPC stubs are up to date
	./scripts/check_grpc_stubs.sh

.PHONY: grpc-clean
grpc-clean: ## Remove generated protobuf/gRPC stubs
	rm -f src/stocksim_grpc/engine_pb2.py src/stocksim_grpc/engine_pb2_grpc.py

.PHONY: lint
lint: ## Run Ruff checks
	uv run --frozen --extra dev --extra grpc ruff check src tests benchmarks examples

.PHONY: typecheck
typecheck: ## Run mypy checks
	uv run --frozen --extra dev --extra grpc mypy --config-file pyproject.toml src tests benchmarks

.PHONY: test
test: ## Run test suite
	uv run --frozen --extra dev --extra grpc pytest -q

.PHONY: test-unit
test-unit: ## Run unit tests
	uv run --frozen --extra dev --extra grpc pytest -q tests/unit_tests

.PHONY: test-integration
test-integration: ## Run integration tests
	uv run --frozen --extra dev --extra grpc pytest -q tests/integration_tests

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	uv run --frozen --extra dev --extra grpc pytest -q tests/e2e_tests

.PHONY: test-grpc-parity
test-grpc-parity: ## Run gRPC/API parity tests
	uv run --frozen --extra dev --extra grpc pytest -q tests/integration_tests/test_grpc_parity.py

.PHONY: test-grpc-parity-property
test-grpc-parity-property: ## Run gRPC/API property parity tests (Hypothesis)
	uv run --frozen --extra dev --extra grpc pytest -q tests/integration_tests/test_grpc_parity.py -m property

.PHONY: openapi-check
openapi-check: ## Validate OpenAPI contract and verify runtime drift
	env -u VIRTUAL_ENV uv run --frozen --project . --extra dev --extra grpc --with openapi-spec-validator --with pyyaml python scripts/check_openapi.py

.PHONY: test-throughput-smoke
test-throughput-smoke: ## Run step_many throughput regression smoke test
	uv run --frozen --extra dev --extra grpc pytest -q tests/integration_tests/test_step_many_throughput.py

.PHONY: smoke-api-grpc
smoke-api-grpc: ## Run docker-compose HTTP + gRPC smoke test
	@set -euo pipefail; \
	cleanup() { \
		IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" docker compose -f examples/docker-compose.api-grpc.yml down --remove-orphans || true; \
	}; \
	trap cleanup EXIT; \
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" timeout --foreground "$(SMOKE_TIMEOUT)" docker compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke

.PHONY: ci
ci: grpc-check openapi-check lint typecheck test ## Full CI checks
