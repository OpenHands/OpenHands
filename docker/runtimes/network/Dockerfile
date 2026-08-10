# syntax=docker/dockerfile:1.7
# ghcr.io/heimdall/runtime-network — nmap + Metasploit (+ optional OpenVAS/GVM notes)
# Plane: PROJETOSIN-186 · ADR-0001
#
# Runs as root: Metasploit RPC and (when present) GVM/OpenVAS expect elevated privileges.
# Do not bake MSF_PASSWORD into the image — inject at runtime only.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    gnupg \
    nmap \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# OpenVAS/GVM: Debian bookworm ships scanner/tools packages, but a full GVM stack
# (gvmd + PostgreSQL + Redis + NVT feeds) is not viable inside bookworm-slim without
# substantial extra services and first-boot sync time. We install scanner CLI tools
# when apt resolves them; otherwise the image still meets AC via nmap + msfconsole.
# Prefer Greenbone community containers as a sidecar for production GVM — see README.
RUN apt-get update \
    && (apt-get install -y --no-install-recommends openvas-scanner gvm-tools \
        || echo "WARN: openvas-scanner/gvm-tools unavailable from apt; continuing with nmap+msf") \
    && rm -rf /var/lib/apt/lists/*

# Metasploit Framework (official Rapid7 apt repository)
RUN curl -fsSL https://apt.metasploit.com/metasploit-framework.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/metasploit.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/metasploit.gpg] https://apt.metasploit.com/ buster main" \
      > /etc/apt/sources.list.d/metasploit.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends metasploit-framework \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /workspace

# Root required for Metasploit / privileged network tooling (documented in README).
USER root
WORKDIR /workspace

EXPOSE 8091 55553
ENTRYPOINT ["/entrypoint.sh"]
