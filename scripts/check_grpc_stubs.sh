#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

"${ROOT_DIR}/scripts/generate_grpc_stubs.sh"

if ! git diff --quiet -- \
  proto/stocksim_grpc/engine.proto \
  src/stocksim_grpc/engine_pb2.py \
  src/stocksim_grpc/engine_pb2_grpc.py; then
  echo "gRPC stubs are out of date. Run: scripts/generate_grpc_stubs.sh"
  git --no-pager diff -- \
    proto/stocksim_grpc/engine.proto \
    src/stocksim_grpc/engine_pb2.py \
    src/stocksim_grpc/engine_pb2_grpc.py
  exit 1
fi

echo "gRPC stubs are up to date."
