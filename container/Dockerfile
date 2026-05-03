FROM ghcr.io/astral-sh/uv:0.8.22@sha256:9874eb7afe5ca16c363fe80b294fe700e460df29a55532bbfea234a0f12eddb1 AS uv

FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY proto ./proto
COPY scripts ./scripts
COPY src ./src
COPY examples ./examples

RUN uv venv "${VIRTUAL_ENV}" --python 3.13 \
    && ./scripts/generate_grpc_stubs.sh \
    && uv sync --active --frozen --extra api --extra grpc

CMD ["python", "-m", "stock_simulator.server"]
