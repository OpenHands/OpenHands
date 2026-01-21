# OpenHands with Claude Code CLI Transport
# This Dockerfile creates an OpenHands image with Claude Code CLI installed
# for subscription-based authentication instead of direct API keys.
#
# Build:
#   docker build -t openhands-claude-cli -f containers/claude-cli/Dockerfile .
#
# Run:
#   docker run -e CLAUDE_CODE_OAUTH_TOKEN="sk-ant-..." -p 3000:3000 openhands-claude-cli

ARG OPENHANDS_BUILD_VERSION=dev
FROM node:22-trixie-slim AS frontend-builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
RUN npm run build

FROM python:3.13.7-slim-trixie AS base
FROM base AS backend-builder

WORKDIR /app
ENV PYTHONPATH='/app'

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt-get update -y \
    && apt-get install -y curl make git build-essential jq gettext \
    && python3 -m pip install poetry --break-system-packages

COPY pyproject.toml poetry.lock ./
RUN touch README.md
RUN export POETRY_CACHE_DIR && poetry install --no-root && rm -rf $POETRY_CACHE_DIR

FROM base AS openhands-app

WORKDIR /app

# Re-declare for this section
ARG OPENHANDS_BUILD_VERSION

ENV RUN_AS_OPENHANDS=true
ENV OPENHANDS_USER_ID=42420
ENV SANDBOX_LOCAL_RUNTIME_URL=http://host.docker.internal
ENV USE_HOST_NETWORK=false
ENV WORKSPACE_BASE=/opt/workspace_base
ENV OPENHANDS_BUILD_VERSION=$OPENHANDS_BUILD_VERSION
ENV SANDBOX_USER_ID=0
ENV FILE_STORE=local
ENV FILE_STORE_PATH=/.openhands
ENV INIT_GIT_IN_EMPTY_WORKSPACE=1
RUN mkdir -p $FILE_STORE_PATH
RUN mkdir -p $WORKSPACE_BASE

# Install system dependencies including Node.js for Claude Code CLI
RUN apt-get update -y \
    && apt-get install -y curl ssh sudo gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22.x LTS for Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Verify Claude CLI installation
RUN claude --version

# Default is 1000, but OSX is often 501
RUN sed -i 's/^UID_MIN.*/UID_MIN 499/' /etc/login.defs
# Default is 60000, but we've seen up to 200000
RUN sed -i 's/^UID_MAX.*/UID_MAX 1000000/' /etc/login.defs

RUN groupadd --gid $OPENHANDS_USER_ID openhands
RUN useradd -l -m -u $OPENHANDS_USER_ID --gid $OPENHANDS_USER_ID -s /bin/bash openhands && \
    usermod -aG openhands openhands && \
    usermod -aG sudo openhands && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R openhands:openhands /app && chmod -R 770 /app
RUN sudo chown -R openhands:openhands $WORKSPACE_BASE && sudo chmod -R 770 $WORKSPACE_BASE
USER openhands

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH='/app'

COPY --chown=openhands:openhands --chmod=770 --from=backend-builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY --chown=openhands:openhands --chmod=770 ./skills ./skills
COPY --chown=openhands:openhands --chmod=770 ./openhands ./openhands
COPY --chown=openhands:openhands --chmod=777 ./openhands/runtime/plugins ./openhands/runtime/plugins
COPY --chown=openhands:openhands pyproject.toml poetry.lock README.md MANIFEST.in LICENSE ./

# Download assets
RUN python openhands/core/download.py
RUN find /app \! -group openhands -exec chgrp openhands {} +

COPY --chown=openhands:openhands --chmod=770 --from=frontend-builder /app/build ./frontend/build
COPY --chown=openhands:openhands --chmod=770 ./containers/claude-cli/entrypoint.sh /app/entrypoint.sh

USER root

# Claude CLI configuration
# The CLAUDE_CODE_OAUTH_TOKEN should be provided at runtime via environment variable
# or Kubernetes secret
ENV CLAUDE_CODE_OAUTH_TOKEN=""

# Enable Claude CLI transport by default for this image
ENV USE_CLAUDE_CLI=true

WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "openhands.server.listen:app", "--host", "0.0.0.0", "--port", "3000"]
