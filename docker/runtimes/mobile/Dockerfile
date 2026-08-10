# syntax=docker/dockerfile:1.7
# ghcr.io/heimdall/runtime-mobile — adb, Frida, apktool, jadx
# Plane: PROJETOSIN-186 · ADR-0001
#
# MobSF and Android emulator are intentional SIDECARS (not baked into this image).
# See docker/runtimes/README.md for compose fragments.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    APKTOOL_VERSION=2.9.3 \
    JADX_VERSION=1.5.0 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    unzip \
    android-tools-adb \
    python3 \
    python3-pip \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Frida client tools
RUN pip3 install --no-cache-dir frida-tools

# apktool (official GitHub release JAR)
RUN curl -fsSL \
      "https://github.com/iBotPeaches/Apktool/releases/download/v${APKTOOL_VERSION}/apktool_${APKTOOL_VERSION}.jar" \
      -o /usr/local/lib/apktool.jar \
    && printf '%s\n' '#!/bin/sh' 'exec java -jar /usr/local/lib/apktool.jar "$@"' \
      > /usr/local/bin/apktool \
    && chmod +x /usr/local/bin/apktool

# jadx (DEX decompiler) — zip lays out bin/ + lib/ at the extract root
RUN curl -fsSL \
      "https://github.com/skylot/jadx/releases/download/v${JADX_VERSION}/jadx-${JADX_VERSION}.zip" \
      -o /tmp/jadx.zip \
    && mkdir -p /opt/jadx \
    && unzip -o /tmp/jadx.zip -d /opt/jadx \
    && ln -sf /opt/jadx/bin/jadx /usr/local/bin/jadx \
    && chmod +x /opt/jadx/bin/jadx \
    && rm -f /tmp/jadx.zip

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 1000 --shell /bin/bash runtime \
    && mkdir -p /workspace \
    && chown -R runtime:runtime /workspace

USER runtime
WORKDIR /workspace

EXPOSE 8092
ENTRYPOINT ["/entrypoint.sh"]
