.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash

IMAGE_PROVIDER ?=
IMAGE_PREFIX ?= ffreis
IMAGE_TAG ?= api-grpc-smoke
SMOKE_TIMEOUT ?= 20m
IMAGE_ROOT := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(IMAGE_PREFIX)

GITLEAKS         ?= gitleaks
LEFTHOOK_VERSION ?= 1.7.10
LEFTHOOK_DIR     ?= $(CURDIR)/.bin
LEFTHOOK_BIN     ?= $(LEFTHOOK_DIR)/lefthook

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Install project with dev + API + gRPC extras
	uv sync --frozen --extra dev --extra api --extra grpc

.PHONY: grpc-generate
grpc-generate: ## Regenerate protobuf/gRPC stubs
	./scripts/generate_grpc_stubs.sh

.PHONY: grpc-check
grpc-check: ## Verify protobuf/gRPC stubs are up to date
	./scripts/check_grpc_stubs.sh

.PHONY: grpc-clean
grpc-clean: ## Remove generated protobuf/gRPC stubs
	rm -f src/stocksim_grpc/engine_pb2.py src/stocksim_grpc/engine_pb2_grpc.py

.PHONY: fmt
fmt: ## Format code in place (ruff format)
	uv run --frozen --extra dev --extra api --extra grpc ruff format src tests benchmarks examples

.PHONY: lint
lint: ## Run Ruff checks
	uv run --frozen --extra dev --extra api --extra grpc ruff check src tests benchmarks examples

.PHONY: validate
validate: ## Static type checking (mypy); alias for typecheck
	uv run --frozen --extra dev --extra api --extra grpc mypy --config-file pyproject.toml src tests benchmarks

.PHONY: typecheck
typecheck: ## Run mypy checks
	uv run --frozen --extra dev --extra api --extra grpc mypy --config-file pyproject.toml src tests benchmarks

.PHONY: plan
plan: ## Not applicable — use 'make validate' or 'make test' for Python repos
	@echo "INFO: 'plan' is Terraform-specific and does not apply to Python repos."
	@echo "      To type-check: make validate"
	@echo "      To run tests: make test"

.PHONY: test
test: ## Run test suite
	uv run --frozen --extra dev --extra api --extra grpc pytest -q

.PHONY: test-unit
test-unit: ## Run unit tests
	uv run --frozen --extra dev --extra api --extra grpc pytest -q tests/unit_tests

.PHONY: test-integration
test-integration: ## Run integration tests
	uv run --frozen --extra dev --extra api --extra grpc pytest -q tests/integration_tests

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	uv run --frozen --extra dev --extra api --extra grpc pytest -q tests/e2e_tests

.PHONY: test-grpc-parity
test-grpc-parity: ## Run gRPC/API parity tests
	uv run --frozen --extra dev --extra api --extra grpc pytest -q tests/integration_tests/test_grpc_parity.py

.PHONY: test-grpc-parity-property
test-grpc-parity-property: ## Run gRPC/API property parity tests (Hypothesis)
	uv run --frozen --extra dev --extra api --extra grpc pytest -q tests/integration_tests/test_grpc_parity.py -m property

.PHONY: openapi-check
openapi-check: ## Validate OpenAPI contract and verify runtime drift
	env -u VIRTUAL_ENV uv run --frozen --project . --extra dev --extra api --extra grpc --with openapi-spec-validator --with pyyaml python scripts/check_openapi.py

.PHONY: test-throughput-smoke
test-throughput-smoke: ## Run step_many throughput regression smoke test
	uv run --frozen --extra dev --extra api --extra grpc pytest -q tests/integration_tests/test_step_many_throughput.py

.PHONY: smoke-api-grpc
smoke-api-grpc: ## Run docker-compose HTTP + gRPC smoke test
	@set -euo pipefail; \
	cleanup() { \
		IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" docker compose -f examples/docker-compose.api-grpc.yml down --remove-orphans || true; \
	}; \
	trap cleanup EXIT; \
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" timeout --foreground "$(SMOKE_TIMEOUT)" docker compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke

.PHONY: secrets-scan-staged lefthook-bootstrap lefthook-install lefthook-run lefthook

secrets-scan-staged: ## Scan staged diff for secrets
	@command -v $(GITLEAKS) >/dev/null 2>&1 || (echo "Missing tool: $(GITLEAKS). Install: https://github.com/gitleaks/gitleaks#installing" && exit 1)
	$(GITLEAKS) protect --staged --redact

lefthook-bootstrap: ## Download lefthook binary into ./.bin
	LEFTHOOK_VERSION="$(LEFTHOOK_VERSION)" BIN_DIR="$(LEFTHOOK_DIR)" bash ./scripts/bootstrap_lefthook.sh

lefthook-install: lefthook-bootstrap ## Install git hooks (runs bootstrap first)
	@if [ -x "$(LEFTHOOK_BIN)" ] && [ -x ".git/hooks/pre-commit" ] && [ -x ".git/hooks/pre-push" ] && [ -x ".git/hooks/commit-msg" ]; then \
		echo "lefthook hooks already installed"; \
		exit 0; \
	fi
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" install

lefthook-run: lefthook-bootstrap ## Run all hooks locally (pre-commit + commit-msg + pre-push)
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-commit
	@tmp_msg="$$(mktemp)"; \
	echo "chore(hooks): validate commit-msg hook" > "$$tmp_msg"; \
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run commit-msg -- "$$tmp_msg"; \
	rm -f "$$tmp_msg"
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-push

lefthook: lefthook-bootstrap lefthook-install lefthook-run ## Install hooks and run them

.PHONY: ci
ci: grpc-check openapi-check lint typecheck test ## Full CI checks

# ── Standard quality-system targets ──────────────────────────────────────────
SRC_DIR  ?= src
TEST_DIR ?= tests/unit_tests

.PHONY: fmt-check
fmt-check: ## Check formatting (no changes)
	uv run --frozen --extra dev --extra api --extra grpc ruff format --check .
	uv run --frozen --extra dev --extra api --extra grpc ruff check .

.PHONY: test-all
test-all: ## Run full test suite
	uv run --frozen --extra dev --extra api --extra grpc pytest tests/

.PHONY: test-property
test-property: ## Run Hypothesis property-based tests
	uv run --frozen --extra dev --extra api --extra grpc pytest -q tests/hypothesis_tests/ 2>/dev/null || \
	  uv run --frozen --extra dev --extra api --extra grpc pytest -q -k "hypothesis or property" tests/ 2>/dev/null || true

.PHONY: coverage
coverage: ## Run tests with coverage report
	uv run --frozen --extra dev --extra api --extra grpc pytest \
	  --cov=$(SRC_DIR) --cov-report=term-missing \
	  --cov-report=xml:coverage.xml $(TEST_DIR)

.PHONY: mutation-test
mutation-test: ## Run mutation testing with mutmut (slow — run in CI)
	uv run mutmut run --paths-to-mutate=$(SRC_DIR) --tests-dir=$(TEST_DIR) || true
	uv run mutmut results

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -type f -name '*.py[cod]' -delete
