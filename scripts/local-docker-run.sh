#!/usr/bin/env bash
#
# Build and run Agent Canvas Docker container locally.
#
# Usage:
#   ./scripts/local-docker-run.sh              # Build and run
#   ./scripts/local-docker-run.sh --no-build   # Run existing image (skip build)
#   ./scripts/local-docker-run.sh --rebuild    # Force rebuild (no cache)
#
# The container:
#   - Runs on port 8000 (configurable via PORT env var)
#   - Persists data in the Docker named volume 'openhands-state'
#   - Mounts the feedback-app folder at /projects for agent workspace
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
IMAGE_TAG="${IMAGE_TAG:-agent-canvas:local}"
CONTAINER_NAME="${CONTAINER_NAME:-agent-canvas}"
PORT="${PORT:-8000}"
STATE_VOLUME="${STATE_VOLUME:-openhands-state}"
PROJECTS_DIR="${PROJECTS_DIR:-$HOME/workspace/gitlab-company/feedback-app}"

# Parse arguments
BUILD=true
BUILD_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-build)
      BUILD=false
      ;;
    --rebuild)
      BUILD_ARGS+=("--" "--no-cache")
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --no-build    Skip build, run existing image"
      echo "  --rebuild     Force rebuild without cache"
      echo "  --help        Show this help"
      echo ""
      echo "Environment variables:"
      echo "  IMAGE_TAG       Docker image tag (default: agent-canvas:local)"
      echo "  CONTAINER_NAME  Container name (default: agent-canvas)"
      echo "  PORT            Host port (default: 8000)"
      echo "  STATE_VOLUME    Docker volume for state (default: openhands-state)"
      echo "  PROJECTS_DIR    Projects directory (default: ~/workspace/gitlab-company/feedback-app)"
      echo "  NO_FORWARDER    Set to 1 to skip the MCP OAuth callback forwarder"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container '$CONTAINER_NAME'..."
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Build the image
if [ "$BUILD" = true ]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "Building Docker image: $IMAGE_TAG"
  echo "════════════════════════════════════════════════════════════════"
  echo ""
  node scripts/docker-build.mjs --tag "$IMAGE_TAG" "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
  echo ""
  echo "✓ Build complete"
fi

# Create directories if they don't exist
if [ ! -d "$PROJECTS_DIR" ]; then
  echo "Projects directory not found: $PROJECTS_DIR"
  echo "Set PROJECTS_DIR to an existing folder."
  exit 1
fi

# Run the container
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Starting Agent Canvas container"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Image:        $IMAGE_TAG"
echo "  Container:    $CONTAINER_NAME"
echo "  Port:         $PORT"
echo "  State volume: $STATE_VOLUME"
echo "  Projects:     $PROJECTS_DIR"
echo ""

docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${PORT}:8000" \
  -v "${STATE_VOLUME}:/home/openhands/.openhands" \
  -v "${PROJECTS_DIR}:/projects" \
  -e "WORKSPACE_BASE=/projects" \
  "$IMAGE_TAG"

echo ""
echo "✓ Container started"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Agent Canvas is starting up..."
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  URL:  http://localhost:${PORT}/canvas"
echo ""
echo "  View logs:     docker logs -f $CONTAINER_NAME"
echo "  Stop:          docker stop $CONTAINER_NAME"
echo "  Remove:        docker rm $CONTAINER_NAME"
echo ""

# Forward MCP OAuth callbacks (e.g. installing the GitLab MCP server) from the
# container to this machine. The forwarder exits on its own when the container
# stops. Set NO_FORWARDER=1 to skip it.
if [ "${NO_FORWARDER:-0}" != "1" ] && command -v node >/dev/null 2>&1; then
  FORWARDER_PID_FILE="${TMPDIR:-/tmp}/agent-canvas-mcp-forwarder.pid"
  FORWARDER_LOG="${TMPDIR:-/tmp}/agent-canvas-mcp-forwarder.log"
  if [ -f "$FORWARDER_PID_FILE" ]; then
    kill "$(cat "$FORWARDER_PID_FILE")" 2>/dev/null || true
  fi
  nohup node "$SCRIPT_DIR/mcp-oauth-forwarder.mjs" "$CONTAINER_NAME" \
    >"$FORWARDER_LOG" 2>&1 &
  echo $! >"$FORWARDER_PID_FILE"
  echo "  MCP OAuth forwarder started (log: $FORWARDER_LOG)"
  echo ""
fi

# Wait for the service to be ready
echo "Waiting for service to be ready..."
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/canvas" | grep -q "200\|302"; then
    echo ""
    echo "✓ Agent Canvas is ready at http://localhost:${PORT}/canvas"
    echo ""
    # The container generates an API key on first start and saves it in the
    # state volume. Print it so it can be pasted into the "Add a backend" screen.
    API_KEY="$(docker exec "$CONTAINER_NAME" cat /home/openhands/.openhands/agent-canvas/api-key.txt 2>/dev/null || true)"
    if [ -n "$API_KEY" ]; then
      echo "  API key (paste into the 'Add a backend' screen):"
      echo "  $API_KEY"
    else
      echo "  API key file not found in the container."
      echo "  Check: docker logs $CONTAINER_NAME | grep -i 'api key'"
    fi
    echo ""
    exit 0
  fi
  sleep 2
  WAITED=$((WAITED + 2))
  echo -n "."
done

echo ""
echo "⚠ Service did not become ready within ${MAX_WAIT}s"
echo "  Check logs: docker logs $CONTAINER_NAME"
exit 1
