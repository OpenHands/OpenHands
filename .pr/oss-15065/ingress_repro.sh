#!/bin/bash
# Start agent-canvas from this worktree on a spare port, kill the ingress
# process, and check that the launcher restarts it. Usage: ingress_repro.sh <worktree>
set -u
cd "$1"
PORT=18450
LOG=/Users/VascoSchiavo/.claude/jobs/755fefd6/tmp/ingress_repro.log
rm -rf .tmp/ingress-repro
OH_CANVAS_SAFE_STATE_DIR="$PWD/.tmp/ingress-repro/agent-canvas" LOCAL_BACKEND_API_KEY=repro-key \
  node bin/agent-canvas.mjs --port $PORT > "$LOG" 2>&1 &
LAUNCHER=$!
for i in $(seq 1 120); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/" && break
  sleep 2
done
echo "before kill: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/)"
INGRESS=$(lsof -ti tcp:$PORT -sTCP:LISTEN | head -1)
echo "killing ingress pid $INGRESS"
kill -9 "$INGRESS"
sleep 1
echo "right after kill: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/)"
sleep 4
echo "5 s after kill: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/)"
echo "launcher still running: $(kill -0 $LAUNCHER 2>/dev/null && echo yes || echo no)"
echo "--- launcher log (ingress lines) ---"
grep -a "ingress\]" "$LOG" | grep -av "║\|╔\|╚\|╠" | tail -8
kill -TERM $LAUNCHER
sleep 5
