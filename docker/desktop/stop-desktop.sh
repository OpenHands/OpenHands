#!/usr/bin/env bash
# Stop the on-demand KasmVNC desktop session.
set -euo pipefail

DISPLAY_NUM="${DESKTOP_DISPLAY_NUM:-1}"
STATE_DIR="${HOME}/.openhands/agent-canvas/desktop"
PID_FILE="${STATE_DIR}/vnc.pid"

if command -v vncserver >/dev/null 2>&1; then
  vncserver -kill ":${DISPLAY_NUM}" >/dev/null 2>&1 || true
fi

if [ -f "${PID_FILE}" ]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi

echo "Desktop stopped"
