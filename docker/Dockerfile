# syntax=docker/dockerfile:1.7

# ═══════════════════════════════════════════════════════════════════════════════
# agent-canvas all-in-one Docker image (HEADLESS)
#
# Combines three services into a single image:
#   1. Agent Server  — binary + venv copied from the agent-server image
#   2. Automation    — installed via pip from openhands-automation
#   3. Frontend      — agent-canvas static build served by Node.js
#
# The entrypoint starts all three services and an ingress proxy that unifies
# them behind a single port (default 8000):
#   /api/automation/*  → automation backend (:18001)
#   /api/*, /sockets   → agent server (:18000)
#   /* (default)       → static frontend + SPA fallback
#
# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN (headless)
# Instead of inheriting the full agent-server image (which includes Docker,
# VNC, XFCE Desktop, Chromium, and VSCode Server — ~625 MB of GUI tooling
# that agent-canvas never uses), this build starts from the same slim
# python-nodejs base and copies only the agent-server artefacts it needs:
#
#   • /agent-server/                           (Python venv + dependencies)
#   • /usr/local/bin/openhands-agent-server    (PyInstaller binary)
#   • /opt/acp-node/                           (ACP CLI agents)
#   • /bin/uv, /bin/uvx                        (Python package manager)
#
# The slim base (nikolaik/python-nodejs:python3.13-nodejs22-slim) is the
# same image used by agent-server's own base-image-minimal target, so the
# copied artefacts are binary-compatible.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Build args ────────────────────────────────────────────────────────────────
# No hardcoded defaults — values are derived from config/defaults.json.
# CI passes these via --build-arg; for local builds use:
#   node scripts/docker-build.mjs          (recommended, reads JSON for you)
#   docker build --build-arg AGENT_SERVER_IMAGE=... --build-arg AUTOMATION_VERSION=... -f docker/Dockerfile .
ARG AGENT_SERVER_IMAGE
ARG AUTOMATION_VERSION

# ── Stage 1: Build frontend ──────────────────────────────────────────────────
FROM node:24-slim AS frontend-build

WORKDIR /build

# Cache-friendly: package files first
COPY package.json package-lock.json ./
RUN npm ci

# Copy everything needed for the build
COPY . .

# Build the static frontend.
# VITE_POSTHOG_API_KEY is a public client key baked into the bundle. CI passes
# production for tagged releases and staging for PR/main images.
# VITE_BASE_PATH is baked into React Router/Vite so the Docker image serves
# Canvas under the Cloud subpath instead of a dedicated hostname.
ARG VITE_POSTHOG_API_KEY=""
ARG VITE_BASE_PATH="/canvas"
ENV VITE_POSTHOG_API_KEY=${VITE_POSTHOG_API_KEY}
ENV VITE_BASE_PATH=${VITE_BASE_PATH}
RUN npm run build

# ── Stage 1b: Generate shell-sourceable defaults from config/defaults.json ──
# This avoids needing jq/python at container runtime to parse the JSON.
FROM node:24-slim AS config-gen
COPY config/defaults.json /tmp/
RUN node -e " \
  const c = JSON.parse(require('fs').readFileSync('/tmp/defaults.json','utf-8')); \
  const lines = [ \
    'CONFIG_AGENT_SERVER_PORT=' + c.ports.agentServer, \
    'CONFIG_AUTOMATION_PORT=' + c.ports.automation, \
    'CONFIG_PROXY_PORT=' + c.ports.proxy, \
    'CONFIG_STATE_SUBDIR=' + c.paths.stateSubdir, \
    'CONFIG_CONVERSATIONS=' + c.paths.conversations, \
    'CONFIG_BASH_EVENTS=' + c.paths.bashEvents, \
    'CONFIG_AUTOMATION_DB=' + c.paths.automationDb, \
    'CONFIG_CANVAS_BASE_PATH=' + c.paths.canvasBasePath, \
    'CONFIG_POSTHOG_API_KEY=' + c.telemetry.posthogApiKey, \
    'CONFIG_POSTHOG_HOST=' + c.telemetry.posthogHost, \
  ]; \
  require('fs').writeFileSync('/tmp/defaults.env', lines.join('\n') + '\n'); \
"

# ── Stage 2: Slim headless runtime ───────────────────────────────────────
# We use the same slim base as agent-server's base-image-minimal target
# rather than inheriting the full image with its Docker/VNC/Chromium/VSCode
# layers. Agent-server artefacts are copied in from the full image.
FROM nikolaik/python-nodejs:python3.13-nodejs22-slim AS final

ARG AUTOMATION_VERSION
ARG AGENT_SERVER_SDK_GIT_REF=""
ARG OPENHANDS_BUILD_GIT_SHA=unknown
ARG OPENHANDS_BUILD_GIT_REF=unknown
ARG VITE_BASE_PATH="/canvas"
ARG VITE_POSTHOG_API_KEY=""

ARG AGENT_CANVAS_VERSION=dev
LABEL org.opencontainers.image.title="agent-canvas"
LABEL org.opencontainers.image.description="agent-canvas (headless): Agent Server + Automation + Frontend"
LABEL org.opencontainers.image.source="https://github.com/OpenHands/OpenHands"
LABEL org.opencontainers.image.version="${AGENT_CANVAS_VERSION}"
LABEL org.opencontainers.image.revision="${OPENHANDS_BUILD_GIT_SHA}"

ENV AGENT_CANVAS_BUILD_GIT_SHA=${OPENHANDS_BUILD_GIT_SHA}
ENV AGENT_CANVAS_BUILD_GIT_REF=${OPENHANDS_BUILD_GIT_REF}
ENV AGENT_CANVAS_BASE_PATH=${VITE_BASE_PATH}
ENV VITE_POSTHOG_API_KEY=${VITE_POSTHOG_API_KEY}
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

USER root

# ── Headless system deps ───────────────────────────────────────────────────
# Mirrors agent-server base-image-minimal: CLI tools only, no GUI packages.
# libpq-dev is needed by automation's asyncpg (PostgreSQL driver).
RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        bash ca-certificates curl wget sudo git jq tmux tar \
        build-essential coreutils util-linux procps findutils grep sed \
        tini xz-utils libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Create openhands user (same UID/GID as agent-server image) ────────────
RUN groupadd -g 10001 openhands \
    && useradd -m -u 10001 -g 10001 -s /bin/bash openhands \
    && echo "openhands ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /workspace/project \
    && chown -R openhands:openhands /workspace

# ── Copy agent-server artefacts from full image ───────────────────────────
# We pull the PyInstaller binary, Python venv, uv, and ACP CLI tools from
# the published agent-server image, then discard the GUI layers.
COPY --from=${AGENT_SERVER_IMAGE} /agent-server /agent-server
COPY --from=${AGENT_SERVER_IMAGE} /usr/local/bin/openhands-agent-server /usr/local/bin/openhands-agent-server
COPY --from=${AGENT_SERVER_IMAGE} /bin/uv /bin/uv
COPY --from=${AGENT_SERVER_IMAGE} /bin/uvx /bin/uvx

# ACP runtime (Claude Code, Codex, Gemini CLI) — copied from full image.
# If the source image is incompatible (e.g. musl libc), these paths may not
# exist; the COPY is best-effort so the build continues.
COPY --from=${AGENT_SERVER_IMAGE} /opt/acp-node /opt/acp-node 2>/dev/null || true
COPY --from=${AGENT_SERVER_IMAGE} /usr/local/bin/claude-agent-acp /usr/local/bin/ 2>/dev/null || true
COPY --from=${AGENT_SERVER_IMAGE} /usr/local/bin/codex-acp /usr/local/bin/ 2>/dev/null || true
COPY --from=${AGENT_SERVER_IMAGE} /usr/local/bin/gemini /usr/local/bin/ 2>/dev/null || true
COPY --from=${AGENT_SERVER_IMAGE} /etc/claude-code /etc/claude-code 2>/dev/null || true

# ── Fix PyInstaller binary library path ───────────────────────────────────
# The agent-server binary was built against the full image's system libs.
# Ensure both amd64 and arm64 lib paths are available at runtime.
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ── Install automation server ─────────────────────────────────────────────
RUN uv pip install --system "openhands-automation==${AUTOMATION_VERSION}" 2>/dev/null \
    || pip install --no-cache-dir "openhands-automation==${AUTOMATION_VERSION}"

RUN if [ -n "${AGENT_SERVER_SDK_GIT_REF}" ]; then \
      BASE_GIT_URL="git+https://github.com/OpenHands/software-agent-sdk@${AGENT_SERVER_SDK_GIT_REF}" && \
      uv pip install --system --force-reinstall \
        "${BASE_GIT_URL}#subdirectory=openhands-sdk" \
        "${BASE_GIT_URL}#subdirectory=openhands-tools" \
        "${BASE_GIT_URL}#subdirectory=openhands-workspace"; \
    fi

# ── Frontend + static server ──────────────────────────────────────────────
COPY --from=frontend-build /build/build /opt/agent-canvas/frontend
COPY scripts/static-server.mjs /opt/agent-canvas/static-server.mjs
COPY scripts/proxy-utils.mjs /opt/agent-canvas/proxy-utils.mjs
COPY --from=frontend-build /build/node_modules/httpxy /opt/agent-canvas/node_modules/httpxy
COPY --from=frontend-build /build/node_modules/sirv /opt/agent-canvas/node_modules/sirv
COPY --from=frontend-build /build/node_modules/@polka /opt/agent-canvas/node_modules/@polka
COPY --from=frontend-build /build/node_modules/mrmime /opt/agent-canvas/node_modules/mrmime
COPY --from=frontend-build /build/node_modules/totalist /opt/agent-canvas/node_modules/totalist

# ── Runtime scripts, config, tools compat, entrypoint ─────────────────────
COPY scripts/runtime-services-info.mjs /opt/agent-canvas/runtime-services-info.mjs
COPY tools/ /opt/agent-canvas/tools/
COPY --from=config-gen /tmp/defaults.env /opt/agent-canvas/defaults.env
COPY docker/entrypoint.sh /opt/agent-canvas/entrypoint.sh
RUN chmod +x /opt/agent-canvas/entrypoint.sh

# ── Persistence directories ───────────────────────────────────────────────
RUN mkdir -p /home/openhands/.openhands/agent-canvas/conversations \
             /home/openhands/.openhands/agent-canvas/bash_events \
             /home/openhands/.openhands/automation \
             /projects && \
    chown -R openhands:openhands /home/openhands/.openhands /projects

USER openhands

VOLUME ["/home/openhands/.openhands", "/projects"]
EXPOSE 8000

ENTRYPOINT ["tini", "--", "/opt/agent-canvas/entrypoint.sh"]
