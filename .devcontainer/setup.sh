#!/bin/bash

# Mark the current repository as safe for Git to prevent "dubious ownership" errors,
# which can occur in containerized environments when directory ownership doesn't match the current user.
git config --global --add safe.directory "$(realpath .)"

# Update repo from remote if running on a stale prebuild (no local changes)
if git diff --quiet && git diff --cached --quiet; then
    git fetch origin
    git pull --ff-only origin "$(git branch --show-current)" || true
    # Unshallow if the prebuild used a shallow clone
    if git rev-parse --is-shallow-repository | grep -q true; then
        git fetch --unshallow || true
    fi
fi

# Do common setup tasks (pre-commit hooks, etc.)
source .openhands/setup.sh
