#!/usr/bin/env bash

set -euo pipefail

# Avoid Git's "dubious ownership" warning in Codespaces/dev containers.
git config --global --add safe.directory "$(realpath .)"

# OpenHands' source runner uses `nc` while waiting for the backend.
sudo apt-get update
sudo apt-get install -y netcat-openbsd

# Install uv/uvx when the base image does not already provide them.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Run the repository's standard setup, including pre-commit hooks.
bash .openhands/setup.sh
