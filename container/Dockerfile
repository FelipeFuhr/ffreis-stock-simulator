FROM ghcr.io/astral-sh/uv:0.11.9@sha256:6b6fa841d71a48fbc9e2c55651c5ad570e01104d7a7d701f57b2b22c0f58e9b1 AS uv

FROM python:3.14-slim@sha256:5b3879b6f3cb77e712644d50262d05a7c146b7312d784a18eff7ff5462e77033

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
