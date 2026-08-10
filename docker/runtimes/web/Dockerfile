# syntax=docker/dockerfile:1.7
# ghcr.io/heimdall/runtime-web — OWASP ZAP, Nuclei, Wapiti, Nikto, sqlmap
# Plane: PROJETOSIN-186 · ADR-0001

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    ZAP_VERSION=2.17.0 \
    ZAP_SHA256=efe799aaa3627db683b43f00c9c210aea0b75c00cc8f0a0f0434d12bb3ddde5a \
    NUCLEI_VERSION=3.3.9 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps (unzip required for Nuclei release zip). Pin bookworm so
# openjdk-17 remains available (rolling python:3.12-slim moved to trixie).
# nikto is not in Debian bookworm main — install from upstream git below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    openjdk-17-jre-headless \
    libnet-ssleay-perl \
    libwhisker2-perl \
    && rm -rf /var/lib/apt/lists/*

# Nikto (official upstream; not packaged in bookworm main)
RUN git clone --depth 1 https://github.com/sullo/nikto.git /opt/nikto \
    && printf '%s\n' '#!/bin/sh' 'exec perl /opt/nikto/program/nikto.pl "$@"' \
      > /usr/local/bin/nikto \
    && chmod +x /usr/local/bin/nikto /opt/nikto/program/nikto.pl

# OWASP ZAP (official GitHub release tarball + SHA-256 from release notes)
RUN curl -fsSL \
      "https://github.com/zaproxy/zaproxy/releases/download/v${ZAP_VERSION}/ZAP_${ZAP_VERSION}_Linux.tar.gz" \
      -o /tmp/zap.tgz \
    && echo "${ZAP_SHA256}  /tmp/zap.tgz" | sha256sum -c - \
    && tar -xzf /tmp/zap.tgz -C /opt \
    && ln -sf "/opt/ZAP_${ZAP_VERSION}/zap.sh" /usr/local/bin/zap \
    && chmod +x "/opt/ZAP_${ZAP_VERSION}/zap.sh" \
    && rm -f /tmp/zap.tgz

# Nuclei (pinned amd64 release; CI builds linux/amd64)
RUN curl -fsSL \
      "https://github.com/projectdiscovery/nuclei/releases/download/v${NUCLEI_VERSION}/nuclei_${NUCLEI_VERSION}_linux_amd64.zip" \
      -o /tmp/nuclei.zip \
    && unzip -o /tmp/nuclei.zip -d /usr/local/bin nuclei \
    && chmod +x /usr/local/bin/nuclei \
    && rm -f /tmp/nuclei.zip

# Python scanners
RUN pip install --no-cache-dir wapiti3 sqlmap

# MCP SDK (stdio servers from services/mcp-servers mounted or copied at /opt/mcp-servers)
# Plane: PROJETOSIN-187 — see docker/runtimes/README.md § MCP servers
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 1000 --shell /bin/bash runtime \
    && mkdir -p /workspace \
    && chown -R runtime:runtime /workspace /app /opt/ZAP_${ZAP_VERSION}

USER runtime
WORKDIR /workspace

EXPOSE 8080 8090
ENTRYPOINT ["/entrypoint.sh"]
