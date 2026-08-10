# syntax=docker/dockerfile:1.7
# ghcr.io/heimdall/runtime-sast — Semgrep + Trivy
# Plane: PROJETOSIN-186 · ADR-0001

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    TRIVY_VERSION=0.73.0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Semgrep
RUN pip install --no-cache-dir semgrep

# Trivy (official GitHub release tarball)
RUN curl -fsSL \
      "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.tar.gz" \
    | tar -xz -C /usr/local/bin trivy \
    && chmod +x /usr/local/bin/trivy

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 1000 --shell /bin/bash runtime \
    && mkdir -p /workspace \
    && chown -R runtime:runtime /workspace

USER runtime
WORKDIR /workspace

EXPOSE 8094
ENTRYPOINT ["/entrypoint.sh"]
