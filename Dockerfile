# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY api.py auth.py list_chats.py main.py ./
COPY config/ config/
COPY evals/ evals/
COPY pipeline/ pipeline/
COPY storage/ storage/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV HOME=/home/eidolon \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid "${APP_GID}" eidolon \
    && useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" eidolon

WORKDIR /app

COPY --from=builder --chown=eidolon:eidolon /app /app
RUN mkdir -p /app/data \
    && chown eidolon:eidolon /app/data \
    && chmod 0700 /app/data

USER eidolon

EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).close()"]

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
