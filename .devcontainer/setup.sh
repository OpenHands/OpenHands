#!/bin/bash

# Mark the current repository as safe for Git to prevent "dubious ownership" errors,
# which can occur in containerized environments when directory ownership doesn't match the current user.
git config --global --add safe.directory "$(realpath .)"

# Defensive cleanup for deprecated Yarn APT repository entries that can
# break apt update with NO_PUBKEY during devcontainer provisioning.
sudo rm -f /etc/apt/sources.list.d/yarn.list 2>/dev/null || true

# Install `nc`
sudo apt update && sudo apt install netcat -y

# Install `uv` and `uvx`
wget -qO- https://astral.sh/uv/install.sh | sh

# Do common setup tasks
source .openhands/setup.sh
