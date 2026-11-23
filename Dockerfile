# Python environment for the project (Linux, PyTorch CPU build).
# The code is not copied in: docker-compose mounts the project folder at /app,
# so edits are picked up without rebuilding. Rebuild only when uv.lock changes.

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# both stages run as an unprivileged user; the model cache (a volume) belongs to it
RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown app /models
USER app
