FROM ghcr.io/openhands/agent-server:1.19.0-python

USER root

# ---------------------------------------------------------------------------
# Comprehensive system toolbox so the agent rarely hits "command not found".
# ---------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    # networking / diagnostics
    iproute2 net-tools iputils-ping dnsutils traceroute mtr-tiny nmap tcpdump \
    netcat-openbsd socat lsof telnet whois iptables ethtool arping \
    # HTTP clients
    curl wget httpie \
    # editors / text utils
    vim nano less tree jq ripgrep fd-find \
    file bsdmainutils xxd bc moreutils colordiff \
    # VCS / SSH / crypto
    git git-lfs openssh-client ca-certificates gnupg gpg-agent \
    # build toolchain
    build-essential make cmake pkg-config autoconf automake libtool \
    # archives
    unzip zip tar gzip bzip2 xz-utils p7zip-full zstd \
    # process / system inspection
    procps psmisc htop strace ltrace sysstat iotop \
    # python tooling
    python3-pip python3-venv python3-dev \
    # database CLIs
    sqlite3 postgresql-client default-mysql-client redis-tools \
    # misc dev utilities
    gettext-base rsync expect \
    && rm -rf /var/lib/apt/lists/*

# fd-find → fd
RUN ln -sf /usr/bin/fdfind /usr/local/bin/fd || true

# ---------------------------------------------------------------------------
# Node.js 22 (LTS) + pnpm + yarn + typescript
# ---------------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g --no-audit --no-fund --force pnpm yarn typescript ts-node && \
    rm -rf /var/lib/apt/lists/* /root/.npm

# ---------------------------------------------------------------------------
# Go 1.22 (static tarball)
# ---------------------------------------------------------------------------
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    case "$ARCH" in amd64) GOARCH=amd64 ;; arm64) GOARCH=arm64 ;; *) GOARCH=amd64 ;; esac; \
    curl -fsSL "https://go.dev/dl/go1.22.12.linux-${GOARCH}.tar.gz" -o /tmp/go.tgz && \
    tar -C /usr/local -xzf /tmp/go.tgz && \
    rm /tmp/go.tgz && \
    ln -sf /usr/local/go/bin/go /usr/local/bin/go && \
    ln -sf /usr/local/go/bin/gofmt /usr/local/bin/gofmt

# ---------------------------------------------------------------------------
# Python tooling: uv, pipx, poetry + useful libs
# ---------------------------------------------------------------------------
RUN python3 -m ensurepip --upgrade || true && \
    python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip || true && \
    python3 -m pip install --no-cache-dir --break-system-packages \
    uv pipx poetry \
    requests httpx beautifulsoup4 lxml pyyaml rich pydantic \
    pandas numpy \
    || true

# ---------------------------------------------------------------------------
# GitHub CLI + Docker CLI (uses mounted host socket if provided)
# ---------------------------------------------------------------------------
RUN install -d -m 0755 /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    -o /etc/apt/keyrings/githubcli-archive-keyring.asc && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.asc] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
    -o /etc/apt/keyrings/docker.asc && \
    chmod go+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh docker-ce-cli docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# kubectl + helm
# ---------------------------------------------------------------------------
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/${ARCH}/kubectl" -o /usr/local/bin/kubectl && \
    chmod +x /usr/local/bin/kubectl && \
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# ---------------------------------------------------------------------------
# Env / PATH
# ---------------------------------------------------------------------------
ENV PATH="/usr/local/go/bin:/home/openhands/.local/bin:${PATH}" \
    GOPATH=/home/openhands/go \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

RUN mkdir -p /home/openhands/.local/bin /home/openhands/go && \
    chown -R openhands:openhands /home/openhands/.local /home/openhands/go

# Restore the default non-root user from the base image.
USER openhands
