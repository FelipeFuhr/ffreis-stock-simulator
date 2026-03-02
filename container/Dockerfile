FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.13-slim

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
