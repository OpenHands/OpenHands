ARG OPENHANDS_BUILD_VERSION=dev

# --- Professional, robust Dockerfile ---
FROM python:3.13.7-slim AS base

# Install all dependencies in one step
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl make git build-essential jq gettext tmux ssh sudo \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user
WORKDIR /app
ARG OPENHANDS_BUILD_VERSION=dev
ENV PYTHONPATH='/app'
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    RUN_AS_OPENHANDS=true \
    OPENHANDS_USER_ID=42420 \
    SANDBOX_LOCAL_RUNTIME_URL=http://host.docker.internal \
    USE_HOST_NETWORK=false \
    WORKSPACE_BASE=/opt/workspace_base \
    OPENHANDS_BUILD_VERSION=$OPENHANDS_BUILD_VERSION \
    SANDBOX_USER_ID=0 \
    FILE_STORE=local \
    FILE_STORE_PATH=/app/.openhands-state \
    INIT_GIT_IN_EMPTY_WORKSPACE=1

# Create required directories
RUN mkdir -p /app/.openhands-state /opt/workspace_base

# Create user and set permissions
RUN groupadd --gid 42420 openhands && \
    useradd -l -m -u 42420 --gid 42420 -s /bin/bash openhands && \
    usermod -aG sudo openhands && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R openhands:openhands /app && chmod -R 770 /app && \
    chown -R openhands:openhands /opt/workspace_base && chmod -R 770 /opt/workspace_base

# --- Build frontend ---
FROM node:24.8-trixie-slim AS frontend-builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# --- Build backend ---
FROM base AS backend-builder
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry
RUN poetry config virtualenvs.in-project true
RUN touch README.md
RUN poetry install --no-root && rm -rf /tmp/poetry_cache

# --- Final image ---
FROM base AS openhands-app
WORKDIR /app

# Install tmux as root in final image to guarantee availability (after all COPY/USER)

ENV PYTHONPATH='/app' \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

COPY --from=backend-builder /app/.venv /app/.venv
COPY --from=frontend-builder /app/build ./frontend/build
COPY --chown=openhands:openhands --chmod=770 ./skills ./skills
COPY --chown=openhands:openhands --chmod=770 ./openhands ./openhands
COPY --chown=openhands:openhands pyproject.toml poetry.lock README.md MANIFEST.in LICENSE ./
COPY --chown=openhands:openhands ./containers/app/entrypoint.sh /app/entrypoint.sh

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/* && which tmux && tmux -V

RUN python openhands/core/download.py || true

ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
RUN /app/.venv/bin/playwright install --with-deps


ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "openhands.server.listen:app", "--host", "0.0.0.0", "--port", "3000"]
