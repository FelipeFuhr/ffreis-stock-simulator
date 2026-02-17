# Docker Compose Setup

This project can run API and gRPC simulator services with Docker Compose.

## Services

- `simulator-api`
  - Runs `stock_simulator.server` (FastAPI)
  - Exposes HTTP endpoint on `http://localhost:8000`
- `simulator-grpc`
  - Runs gRPC engine endpoint on `localhost:50051`

## Run

```bash
docker compose up --build
```

Run API + gRPC smoke stack:

```bash
docker compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke
```

Run only API or gRPC server:

```bash
docker compose up --build simulator-api
docker compose up --build simulator-grpc
```
