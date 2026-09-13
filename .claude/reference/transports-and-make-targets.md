## Transports and required make targets

- **Both HTTP and gRPC are production transports** — not alternatives. The RL agent
  uses HTTP; the integration hub tests gRPC.

- **Required make targets** (called by integration-hub): `test-grpc-parity`.
