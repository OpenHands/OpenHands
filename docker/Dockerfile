# syntax=docker/dockerfile:1.7

# ═══════════════════════════════════════════════════════════════════════════════
# agent-canvas all-in-one Docker image
#
# Combines three services into a single image:
#   1. Agent Server  — from ghcr.io/openhands/agent-server (upstream SDK image)
#   2. Automation    — installed via pip from openhands-automation
#   3. Frontend      — agent-canvas static build served by Node.js
#
# The entrypoint starts all three services and an ingress proxy that unifies
# them behind a single port (default 8000):
#   /api/automation/*  → automation backend (:18001)
#   /api/*, /sockets   → agent server (:18000)
#   /* (default)       → static frontend + SPA fallback
# ═══════════════════════════════════════════════════════════════════════════════

# ── Build args ────────────────────────────────────────────────────────────────
# No hardcoded defaults — values are derived from config/defaults.json.
# CI passes these via --build-arg; for local builds use:
#   node scripts/docker-build.mjs          (recommended, reads JSON for you)
#   docker build --build-arg AGENT_SERVER_IMAGE=... --build-arg AUTOMATION_VERSION=... -f docker/Dockerfile .
ARG AGENT_SERVER_IMAGE
ARG AUTOMATION_VERSION

# Optional sibling package for `file:../typescript-client` lockfiles (local forks).
# Override with: docker build --build-context typescript-client=../typescript-client
# or Compose `additional_contexts`. When omitted, this stage is empty and npm
# must resolve `@openhands/typescript-client` from the registry instead.
FROM scratch AS typescript-client

# Optional local software-agent-sdk (workspace clone and other unreleased APIs).
# Override with: --build-context software-agent-sdk=../software-agent-sdk
FROM scratch AS software-agent-sdk

# ── Stage 1: Build frontend ──────────────────────────────────────────────────
FROM node:24-slim AS frontend-build

WORKDIR /build

# Cache-friendly: package files first
COPY package.json package-lock.json ./
# Place the sibling where `file:../typescript-client` resolves from /build.
# Build it first so parent `npm ci` does not re-run `prepare` without its toolchain.
COPY --from=typescript-client . /typescript-client
RUN cd /typescript-client \
  && npm ci \
  && npm run build \
  && npm pkg delete scripts.prepare
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
    'CONFIG_DESKTOP_VNC_PORT=' + c.ports.desktopVnc, \
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

# ── Stage 2: Combined image ──────────────────────────────────────────────────
FROM ${AGENT_SERVER_IMAGE} AS final

ARG AUTOMATION_VERSION
ARG AGENT_SERVER_SDK_GIT_REF=""
ARG OPENHANDS_BUILD_GIT_SHA=unknown
ARG OPENHANDS_BUILD_GIT_REF=unknown
ARG VITE_BASE_PATH="/canvas"
ARG VITE_POSTHOG_API_KEY=""

ARG AGENT_CANVAS_VERSION=dev
LABEL org.opencontainers.image.title="agent-canvas"
LABEL org.opencontainers.image.description="All-in-one agent-canvas: Agent Server + Automation + Frontend"
LABEL org.opencontainers.image.source="https://github.com/OpenHands/OpenHands"
LABEL org.opencontainers.image.version="${AGENT_CANVAS_VERSION}"
LABEL org.opencontainers.image.revision="${OPENHANDS_BUILD_GIT_SHA}"

ENV AGENT_CANVAS_BUILD_GIT_SHA=${OPENHANDS_BUILD_GIT_SHA}
ENV AGENT_CANVAS_BUILD_GIT_REF=${OPENHANDS_BUILD_GIT_REF}
ENV AGENT_CANVAS_BASE_PATH=${VITE_BASE_PATH}
ENV VITE_POSTHOG_API_KEY=${VITE_POSTHOG_API_KEY}

USER root

# Install system deps required by automation's transitive dependencies
# (asyncpg needs libpq, which the agent-server base image may not include),
# plus a light XFCE desktop + KasmVNC for the on-demand Desktop tab.
# KasmVNC listens on loopback only (ports.desktopVnc); ingress proxies it.
ARG KASMVNC_VERSION=1.3.4
RUN if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        libpq-dev \
        ca-certificates \
        curl \
        wget \
        dbus-x11 \
        xfce4 \
        xfce4-terminal \
        thunar \
        xdg-utils \
        chromium \
        fonts-liberation \
        fonts-dejavu-core \
      || apt-get install -y --no-install-recommends \
        libpq-dev \
        ca-certificates \
        curl \
        wget \
        dbus-x11 \
        xfce4 \
        xfce4-terminal \
        thunar \
        xdg-utils \
        fonts-dejavu-core; \
      # Chromium in Docker needs --no-sandbox; wrap and register as default browser.
      if command -v chromium >/dev/null 2>&1 || command -v chromium-browser >/dev/null 2>&1; then \
        CHROME_BIN="$(command -v chromium || command -v chromium-browser)"; \
        printf '%s\n' \
          '#!/bin/sh' \
          "exec \"${CHROME_BIN}\" --no-sandbox --disable-dev-shm-usage --disable-gpu \"\$@\"" \
          > /usr/local/bin/chromium-desktop && chmod +x /usr/local/bin/chromium-desktop; \
        printf '%s\n' \
          '[Desktop Entry]' \
          'Version=1.0' \
          'Name=Chromium' \
          'GenericName=Web Browser' \
          'Comment=Access the Internet' \
          'Exec=/usr/local/bin/chromium-desktop %U' \
          'Terminal=false' \
          'Type=Application' \
          'Icon=chromium' \
          'Categories=Network;WebBrowser;' \
          'MimeType=text/html;text/xml;application/xhtml+xml;x-scheme-handler/http;x-scheme-handler/https;' \
          'StartupNotify=true' \
          > /usr/share/applications/chromium-desktop.desktop; \
        # Point stock Chromium .desktop entries and XFCE "Web Browser" at the wrapper.
        for desk in /usr/share/applications/chromium.desktop \
                    /usr/share/applications/chromium-browser.desktop; do \
          if [ -f "${desk}" ]; then \
            sed -i -E \
              's|^Exec=([^ ]*chromium(-browser)?)|Exec=/usr/local/bin/chromium-desktop|' \
              "${desk}" || true; \
          fi; \
        done; \
        mkdir -p /home/openhands/.config/xfce4 /usr/share/xfce4/helpers; \
        printf '%s\n' \
          '[Desktop Entry]' \
          'Version=1.0' \
          'Type=X-XFCE-Helper' \
          'Name=Chromium' \
          'Icon=chromium' \
          'StartupNotify=true' \
          'X-XFCE-Binaries=chromium-desktop;chromium;chromium-browser;' \
          'X-XFCE-Category=WebBrowser' \
          'X-XFCE-Commands=%B;%B' \
          'X-XFCE-CommandsWithParameter=%B \"%s\";%B \"%s\"' \
          > /usr/share/xfce4/helpers/chromium-desktop.desktop; \
        printf '%s\n' 'WebBrowser=chromium-desktop' \
          > /home/openhands/.config/xfce4/helpers.rc; \
        printf '%s\n' \
          '[Default Applications]' \
          'text/html=chromium-desktop.desktop' \
          'x-scheme-handler/http=chromium-desktop.desktop' \
          'x-scheme-handler/https=chromium-desktop.desktop' \
          'x-scheme-handler/about=chromium-desktop.desktop' \
          'x-scheme-handler/unknown=chromium-desktop.desktop' \
          > /home/openhands/.config/mimeapps.list; \
        chown -R openhands:openhands /home/openhands/.config; \
        update-alternatives --install /usr/bin/x-www-browser x-www-browser /usr/local/bin/chromium-desktop 200 || true; \
        update-alternatives --install /usr/bin/gnome-www-browser gnome-www-browser /usr/local/bin/chromium-desktop 200 || true; \
      else \
        echo "WARN: Chromium not available from apt — Desktop browser shortcut may fail"; \
      fi; \
      . /etc/os-release; \
      case "${VERSION_CODENAME:-}" in \
        jammy|noble|focal) KASM_DISTRO=jammy ;; \
        bookworm|trixie|bullseye) KASM_DISTRO=bookworm ;; \
        *) KASM_DISTRO=jammy ;; \
      esac; \
      ARCH="$(dpkg --print-architecture)"; \
      KASM_DEB="kasmvncserver_${KASM_DISTRO}_${KASMVNC_VERSION}_${ARCH}.deb"; \
      wget -q -O "/tmp/${KASM_DEB}" \
        "https://github.com/kasmtech/KasmVNC/releases/download/v${KASMVNC_VERSION}/${KASM_DEB}" \
      && apt-get install -y --no-install-recommends "/tmp/${KASM_DEB}" \
      && rm -f "/tmp/${KASM_DEB}" \
      && (getent group ssl-cert >/dev/null && usermod -aG ssl-cert openhands || true) \
      || echo "WARN: KasmVNC install failed — Desktop tab will report unavailable"; \
      rm -rf /var/lib/apt/lists/*; \
    fi

# Install automation server via pip.
# Shared deps (openhands-sdk, fastapi, uvicorn, pydantic, httpx, …) are
# already satisfied by the agent-server base image, so only automation-
# specific packages (asyncpg, sqlalchemy, boto3, gcloud, …) are added.
RUN uv pip install --system "openhands-automation==${AUTOMATION_VERSION}" 2>/dev/null \
    || pip install --no-cache-dir "openhands-automation==${AUTOMATION_VERSION}"

# Prefer a local software-agent-sdk checkout (Compose additional_contexts) so
# unreleased agent-server APIs such as POST /api/workspaces/clone ship in the
# image. Falls back to AGENT_SERVER_SDK_GIT_REF from GitHub when the local
# context is empty.
COPY --from=software-agent-sdk . /opt/software-agent-sdk
RUN if [ -f /opt/software-agent-sdk/openhands-agent-server/pyproject.toml ]; then \
      echo "Installing local software-agent-sdk from /opt/software-agent-sdk..." && \
      uv pip install --system --force-reinstall \
        "/opt/software-agent-sdk/openhands-sdk" \
        "/opt/software-agent-sdk/openhands-tools" \
        "/opt/software-agent-sdk/openhands-workspace" \
        "/opt/software-agent-sdk/openhands-agent-server" && \
      # Base image ships a frozen PyInstaller binary that ignores site-packages.
      # Replace it so the reinstalled Python package (with /workspaces/clone) runs.
      printf '%s\n' \
        '#!/bin/sh' \
        'exec python -m openhands.agent_server "$@"' \
        > /usr/local/bin/openhands-agent-server && \
      chmod +x /usr/local/bin/openhands-agent-server; \
    elif [ -n "${AGENT_SERVER_SDK_GIT_REF}" ]; then \
      BASE_GIT_URL="git+https://github.com/OpenHands/software-agent-sdk@${AGENT_SERVER_SDK_GIT_REF}" && \
      uv pip install --system --force-reinstall \
        "${BASE_GIT_URL}#subdirectory=openhands-sdk" \
        "${BASE_GIT_URL}#subdirectory=openhands-tools" \
        "${BASE_GIT_URL}#subdirectory=openhands-workspace" \
        "${BASE_GIT_URL}#subdirectory=openhands-agent-server" && \
      printf '%s\n' \
        '#!/bin/sh' \
        'exec python -m openhands.agent_server "$@"' \
        > /usr/local/bin/openhands-agent-server && \
      chmod +x /usr/local/bin/openhands-agent-server; \
    else \
      rm -rf /opt/software-agent-sdk; \
    fi

# Copy the frontend build output.
# react-router.config.ts unpacks build/client/ into build/ for non-Vercel builds.
COPY --from=frontend-build /build/build /opt/agent-canvas/frontend

# Copy the static-server / ingress scripts and their production runtime deps.
# ingress.mjs is used when STACK_MODE=backend (API proxy without SPA).
COPY scripts/static-server.mjs /opt/agent-canvas/static-server.mjs
COPY scripts/ingress.mjs /opt/agent-canvas/ingress.mjs
COPY scripts/proxy-utils.mjs /opt/agent-canvas/proxy-utils.mjs
COPY scripts/app-login.mjs /opt/agent-canvas/app-login.mjs
COPY scripts/appwrite-proxy.mjs /opt/agent-canvas/appwrite-proxy.mjs
COPY scripts/plane-proxy.mjs /opt/agent-canvas/plane-proxy.mjs
# Imported by static-server.mjs / ingress.mjs for /api/integrations/dependency-track.
COPY scripts/dependency-track-proxy.mjs /opt/agent-canvas/dependency-track-proxy.mjs
# Imported by static-server.mjs / ingress.mjs for /api/integrations/translate.
COPY scripts/translate-proxy.mjs /opt/agent-canvas/translate-proxy.mjs
# Imported by static-server.mjs / ingress.mjs for /api/desktop (KasmVNC).
COPY scripts/desktop-proxy.mjs /opt/agent-canvas/desktop-proxy.mjs
COPY docker/desktop/ /opt/agent-canvas/desktop/
RUN chmod +x /opt/agent-canvas/desktop/start-desktop.sh \
             /opt/agent-canvas/desktop/stop-desktop.sh \
             /opt/agent-canvas/desktop/xstartup
COPY --from=frontend-build /build/node_modules/httpxy /opt/agent-canvas/node_modules/httpxy
COPY --from=frontend-build /build/node_modules/sirv /opt/agent-canvas/node_modules/sirv
COPY --from=frontend-build /build/node_modules/@polka /opt/agent-canvas/node_modules/@polka
COPY --from=frontend-build /build/node_modules/mrmime /opt/agent-canvas/node_modules/mrmime
COPY --from=frontend-build /build/node_modules/totalist /opt/agent-canvas/node_modules/totalist
COPY --from=frontend-build /build/node_modules/bcryptjs /opt/agent-canvas/node_modules/bcryptjs

# Copy the runtime-services-info builder (entrypoint.sh runs it as a CLI to
# emit the agent's <RUNTIME_SERVICES> block; same builder the dev stack uses).
COPY scripts/runtime-services-info.mjs /opt/agent-canvas/runtime-services-info.mjs

# Persisted conversations created before the client_tools migration still
# import canvas_ui_tool by qualname. Keep the compatibility module available;
# new conversations define canvas_ui_control through a JSON client tool.
COPY tools/ /opt/agent-canvas/tools/

# Cursor CLI (`agent`) for the ACP Cursor preset (`agent acp`). Install into
# the openhands home so the agent-server subprocess can find it on PATH.
# Auth in-container uses CURSOR_API_KEY (or CURSOR_AUTH_TOKEN) from Secrets.
RUN if ! command -v curl >/dev/null 2>&1; then \
      if command -v apt-get >/dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends curl ca-certificates && \
        rm -rf /var/lib/apt/lists/*; \
      fi; \
    fi \
 && curl -fsSL https://cursor.com/install | HOME=/home/openhands bash \
 && chown -R openhands:openhands /home/openhands/.local \
 && ln -sf /home/openhands/.local/bin/agent /usr/local/bin/agent \
 && if [ -x /home/openhands/.local/bin/cursor-agent ]; then \
      ln -sf /home/openhands/.local/bin/cursor-agent /usr/local/bin/cursor-agent; \
    fi

# OpenCode CLI for the ACP OpenCode preset (`opencode acp`).
# Auth in-container uses OPENCODE_API_KEY from Secrets (or `opencode auth login`).
RUN curl -fsSL https://opencode.ai/install | HOME=/home/openhands bash \
 && chown -R openhands:openhands /home/openhands/.opencode \
 && if [ -x /home/openhands/.opencode/bin/opencode ]; then \
      ln -sf /home/openhands/.opencode/bin/opencode /usr/local/bin/opencode; \
    elif [ -x /home/openhands/.local/bin/opencode ]; then \
      ln -sf /home/openhands/.local/bin/opencode /usr/local/bin/opencode; \
    fi

# Copy generated defaults.env (from config/defaults.json via config-gen stage)
COPY --from=config-gen /tmp/defaults.env /opt/agent-canvas/defaults.env

# Copy the entrypoint
COPY docker/entrypoint.sh /opt/agent-canvas/entrypoint.sh
RUN chmod +x /opt/agent-canvas/entrypoint.sh

# Pre-create persistence directories with correct ownership so the
# openhands user can write to them even when Docker creates anonymous
# volumes (which default to root).
RUN mkdir -p /home/openhands/.openhands/agent-canvas/conversations \
             /home/openhands/.openhands/agent-canvas/bash_events \
             /home/openhands/.openhands/automation \
             /projects && \
    chown -R openhands:openhands /home/openhands/.openhands /projects

USER openhands

# Persistence volumes:
#   /home/openhands/.openhands — settings, secrets, conversations, automation DB
#   /projects                  — user code the agent can read/edit
# Bind-mount these for data to survive container restarts:
#   docker run -v ~/.openhands:/home/openhands/.openhands -v ~/projects:/projects ...
VOLUME ["/home/openhands/.openhands", "/projects"]

# The entrypoint starts all services and the ingress proxy.
# Port 8000 is the unified entry point.
EXPOSE 8000

ENTRYPOINT ["tini", "--", "/opt/agent-canvas/entrypoint.sh"]
