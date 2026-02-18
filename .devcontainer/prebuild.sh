#!/bin/bash

# This script runs ONLY during the prebuild phase (via onCreateCommand).
# It does NOT run again when a user opens a prebuilt codespace.

# Pre-build the dev Docker image so postAttachCommand only needs to run it
docker compose -f docker-compose.dev.yml build

# Pre-pull the agent-server sandbox image so the first conversation starts instantly
_repo="${AGENT_SERVER_IMAGE_REPOSITORY:-ghcr.io/openhands/agent-server}"
_tag="${AGENT_SERVER_IMAGE_TAG:-31536c8-python}"
echo "Pre-pulling sandbox image: ${_repo}:${_tag}"
docker pull "${_repo}:${_tag}" || echo "Warning: failed to pre-pull sandbox image (non-fatal)"
