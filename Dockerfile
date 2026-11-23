# Python image for the project (Linux, PyTorch CPU build), in two stages:
#
#   base     the locked dependencies only. Used for development and for setup and
#            evaluation commands: docker-compose mounts the project folder at /app.
#   runtime  base plus the application code, running the API. This is the image that
#            is deployed. It holds no data: the database, the documents' chunks, the
#            vector index and the models are mounted at run time (the documents may
#            not be redistributed, so they never go into an image).

FROM python:3.13-slim AS base

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


FROM base AS runtime

COPY src/ /app/src/
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "dc_ai_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
