#!/usr/bin/env bash
# Start an agent-canvas image with no mounts and fail unless the unified
# entry point (static-server on :8000, proxying /health to the agent-server)
# answers and the container is still up after a stability window.
#
# A unit test cannot catch a module that exists in scripts/ but was never
# copied into the image (#18342): the frontend dies on startup and the
# entrypoint takes the whole container down with it.
#
# A second container then runs with LOCAL_BACKEND_API_KEY set and asserts the
# property operators rely on: a request without X-Session-API-Key is rejected
# (#17763). The entrypoint decides which key variables the agent-server sees,
# so only a booted image can prove the operator-facing variable protects the
# API.
#
# Usage: docker/smoke-test.sh <image> [host-port]
set -euo pipefail

IMAGE="${1:?usage: docker/smoke-test.sh <image> [host-port]}"
PORT="${2:-8000}"
READY_TIMEOUT="${SMOKE_READY_TIMEOUT:-300}"
STABLE_SECONDS="${SMOKE_STABLE_SECONDS:-30}"
NAME="agent-canvas-smoke-$$"
KEY_NAME="${NAME}-key"
KEY_PORT="${SMOKE_KEY_PORT:-$((PORT + 1))}"
SMOKE_API_KEY="smoke-test-key-$$"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
SETTINGS_URL="http://127.0.0.1:${KEY_PORT}/api/settings"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- last 60 lines of container logs ---"
    docker logs --tail 60 "$NAME" 2>&1 || true
    echo "--- last 60 lines of key-check container logs ---"
    docker logs --tail 60 "$KEY_NAME" 2>&1 || true
  fi
  docker rm -f "$NAME" "$KEY_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "::error::Smoke test failed for ${IMAGE}: $*"
  exit 1
}

running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
}

wait_healthy() {
  local url="$1" container="$2"
  local deadline=$((SECONDS + READY_TIMEOUT))
  until curl -fs -o /dev/null --max-time 5 "$url"; do
    running "$container" || fail "container ${container} exited before serving ${url}"
    [ "$SECONDS" -lt "$deadline" ] || fail "no healthy response from ${container} within ${READY_TIMEOUT}s"
    sleep 2
  done
}

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" "$IMAGE" >/dev/null
echo "Started ${IMAGE}; waiting up to ${READY_TIMEOUT}s for ${HEALTH_URL}"
wait_healthy "$HEALTH_URL" "$NAME"
echo "Healthy after ${SECONDS}s; checking it stays up for ${STABLE_SECONDS}s"

stable_until=$((SECONDS + STABLE_SECONDS))
while [ "$SECONDS" -lt "$stable_until" ]; do
  running "$NAME" || fail "container exited after becoming healthy"
  sleep 2
done
curl -fsS -o /dev/null --max-time 5 "$HEALTH_URL" ||
  fail "${HEALTH_URL} stopped answering"

# Session-key enforcement. Readiness here waits for any HTTP status from
# /api/settings (200 or 401 both prove the API is up), so this does not
# depend on whether /health sits behind the key check.
docker run -d --name "$KEY_NAME" -p "127.0.0.1:${KEY_PORT}:8000" \
  -e LOCAL_BACKEND_API_KEY="$SMOKE_API_KEY" "$IMAGE" >/dev/null
echo "Key-check container on :${KEY_PORT}; waiting for the API to answer"

deadline=$((SECONDS + READY_TIMEOUT))
code="000"
until [ "$code" != "000" ]; do
  running "$KEY_NAME" || fail "key-check container exited before serving :8000"
  [ "$SECONDS" -lt "$deadline" ] ||
    fail "key-check container did not serve /api/settings within ${READY_TIMEOUT}s"
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$SETTINGS_URL" || true)"
  sleep 2
done

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$SETTINGS_URL")"
[ "$code" = "401" ] ||
  fail "keyless ${SETTINGS_URL} returned ${code}, expected 401 with LOCAL_BACKEND_API_KEY set"

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
  -H "X-Session-API-Key: ${SMOKE_API_KEY}" "$SETTINGS_URL")"
[ "$code" = "200" ] ||
  fail "authenticated ${SETTINGS_URL} returned ${code}, expected 200 with the configured key"
echo "Session-key enforcement passed: keyless 401, authenticated 200"

echo "Smoke test passed for ${IMAGE}"
