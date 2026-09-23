# syntax=docker/dockerfile:1

# ---------- Stage 1: builder - resolve and install dependencies into a venv ----------
FROM python:3.14 AS builder

# Install uv by copying the static binary from the official image (same version as local)
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency manifests only: this layer (and the slow sync below) is rebuilt
# only when pyproject.toml or uv.lock change, not when the source code changes
COPY pyproject.toml uv.lock ./

# --no-install-project: install the locked dependencies only; the app is run
# from src/ in the runtime stage, so the project package itself isn't needed.
# The cache mount keeps uv's downloaded wheels between builds (not in the image)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# ---------- Stage 2: runtime - slim image with just the venv and the code ----------
FROM python:3.14-slim AS runtime

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

USER app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
