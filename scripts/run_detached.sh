#!/usr/bin/env bash
# Run OpenHands (backend + frontend) in the background and write logs (Option A)
# - Uses nohup to detach, redirects all output to logs/app.log
# - Writes the background PID to logs/app.pid for later control
# - Defaults to RUNTIME=local and ports matching the provided environment
#
# Usage:
#   scripts/run_detached.sh
#   RUNTIME=local BACKEND_HOST=0.0.0.0 BACKEND_PORT=3000 FRONTEND_HOST=0.0.0.0 FRONTEND_PORT=3001 scripts/run_detached.sh
#
# To view logs:
#   tail -f logs/app.log
#
# To stop:
#   kill "$(cat logs/app.pid)"

set -u

# Defaults aligned with environment hints
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-3000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-3001}"

# Ensure RUNTIME is set (local for ProcessSandboxService)
export RUNTIME="${RUNTIME:-local}"

# Ensure we run from repo root (this script is placed in ./scripts)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

mkdir -p logs

# Start build + run in detached mode; group both under nohup via bash -lc
nohup bash -lc "make build && make run BACKEND_HOST=\"$BACKEND_HOST\" BACKEND_PORT=\"$BACKEND_PORT\" FRONTEND_HOST=\"$FRONTEND_HOST\" FRONTEND_PORT=\"$FRONTEND_PORT\"" > logs/app.log 2>&1 &
PID=$!
echo "$PID" > logs/app.pid

echo "Started OpenHands in background (PID: $PID)"
echo "Backend: http://$BACKEND_HOST:$BACKEND_PORT"
echo "Frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"
echo "Logs: logs/app.log"
