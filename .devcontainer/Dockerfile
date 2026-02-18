FROM mcr.microsoft.com/devcontainers/universal:2

# Remove the Yarn apt repo shipped in the base image — its GPG key is expired,
# which causes `apt-get update` to fail.
RUN rm -f /etc/apt/sources.list.d/yarn.list

# Install nc (netcat) - needed by Makefile to check if backend is ready.
RUN apt-get update && apt-get install -y netcat-openbsd tmux && rm -rf /var/lib/apt/lists/*

# Install uv and uvx
RUN wget -qO- https://astral.sh/uv/install.sh | sh
