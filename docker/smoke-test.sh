#!/usr/bin/env bash
# Start an agent-canvas image with no mounts and fail unless the unified
# entry point (static-server on :8000, proxying /health to the agent-server)
# answers and the container is still up after a stability window.
#
# A unit test cannot catch a module that exists in scripts/ but was never
# copied into the image (#18342): the frontend dies on startup and the
# entrypoint takes the whole container down with it.
#
# Usage: docker/smoke-test.sh <image> [host-port]
set -euo pipefail

IMAGE="${1:?usage: docker/smoke-test.sh <image> [host-port]}"
PORT="${2:-8000}"
READY_TIMEOUT="${SMOKE_READY_TIMEOUT:-300}"
STABLE_SECONDS="${SMOKE_STABLE_SECONDS:-30}"
NAME="agent-canvas-smoke-$$"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- last 60 lines of container logs ---"
    docker logs --tail 60 "$NAME" 2>&1 || true
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "::error::Smoke test failed for ${IMAGE}: $*"
  exit 1
}

running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]
}

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" "$IMAGE" >/dev/null
echo "Started ${IMAGE}; waiting up to ${READY_TIMEOUT}s for ${HEALTH_URL}"

deadline=$((SECONDS + READY_TIMEOUT))
until curl -fs -o /dev/null --max-time 5 "$HEALTH_URL"; do
  running || fail "container exited before serving :8000"
  [ "$SECONDS" -lt "$deadline" ] || fail "no healthy response within ${READY_TIMEOUT}s"
  sleep 2
done
echo "Healthy after ${SECONDS}s; checking it stays up for ${STABLE_SECONDS}s"

stable_until=$((SECONDS + STABLE_SECONDS))
while [ "$SECONDS" -lt "$stable_until" ]; do
  running || fail "container exited after becoming healthy"
  sleep 2
done
curl -fsS -o /dev/null --max-time 5 "$HEALTH_URL" ||
  fail "${HEALTH_URL} stopped answering"

# The served agent-server comes from the base image, while system Python gets
# its own copy as a dependency of openhands-workspace; a resolver conflict once
# downgraded that copy silently (#17531). Check both against the release pin.
if [ -n "${EXPECTED_AGENT_SERVER_VERSION:-}" ]; then
  served=$(docker exec "$NAME" python -c 'import json, urllib.request
print(json.load(urllib.request.urlopen("http://127.0.0.1:8000/server_info"))["version"])') ||
    fail "could not read the agent-server version from /server_info"
  installed=$(docker exec "$NAME" python -c 'from importlib.metadata import version
print(version("openhands-agent-server"))') ||
    fail "openhands-agent-server is not installed in system Python"
  echo "Agent Server: expected ${EXPECTED_AGENT_SERVER_VERSION}, served ${served}, installed ${installed}"
  [ "$served" = "$EXPECTED_AGENT_SERVER_VERSION" ] ||
    fail "serves agent-server ${served}, expected ${EXPECTED_AGENT_SERVER_VERSION}"
  [ "$installed" = "$EXPECTED_AGENT_SERVER_VERSION" ] ||
    fail "system Python has openhands-agent-server ${installed}, expected ${EXPECTED_AGENT_SERVER_VERSION}"
  docker exec "$NAME" python -m pip check || fail "pip check found broken requirements"
fi

echo "Smoke test passed for ${IMAGE}"
