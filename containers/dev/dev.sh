#!/usr/bin/env bash
set -o pipefail

function get_docker() {
    echo "Docker is required to build and run OpenHands."
    echo "https://docs.docker.com/get-started/get-docker/"
    exit 1
}

function check_tools() {
	command -v docker &>/dev/null || get_docker
}

function exit_if_indocker() {
    if [ -f /.dockerenv ]; then
        echo "Running inside a Docker container. Exiting..."
        exit 1
    fi
}

#
exit_if_indocker

check_tools

##
OPENHANDS_WORKSPACE=$(git rev-parse --show-toplevel)

cd "$OPENHANDS_WORKSPACE/containers/dev/" || exit 1

##
# Faster image builds (layer cache / parallel); safe to disable with DOCKER_BUILDKIT=0
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-1}"

##
export BACKEND_HOST="0.0.0.0"
#
export SANDBOX_USER_ID=$(id -u)
export WORKSPACE_BASE=${WORKSPACE_BASE:-$OPENHANDS_WORKSPACE/workspace}

if [ "${OPENHANDS_GPU:-0}" = "1" ]; then
	docker compose -f compose.yml -f compose-gpu.yml run --rm --service-ports "$@" dev
else
	docker compose -f compose.yml run --rm --service-ports "$@" dev
fi

##
