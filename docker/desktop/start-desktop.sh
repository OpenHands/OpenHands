#!/usr/bin/env bash
# Start XFCE + KasmVNC on loopback for the agent-canvas Desktop tab.
# Idempotent: if already listening on DESKTOP_VNC_PORT, exit 0.
set -euo pipefail

DESKTOP_VNC_PORT="${DESKTOP_VNC_PORT:-${CONFIG_DESKTOP_VNC_PORT:-6901}}"
DISPLAY_NUM="${DESKTOP_DISPLAY_NUM:-1}"
STATE_DIR="${HOME}/.openhands/agent-canvas/desktop"
LOG_FILE="${STATE_DIR}/vnc.log"
VNC_DIR="${HOME}/.vnc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VNC_USER="${USER:-$(id -un)}"
VNC_PASS="${DESKTOP_VNC_PASSWORD:-canvas}"

mkdir -p "${STATE_DIR}" "${VNC_DIR}" /tmp/.X11-unix /tmp/.ICE-unix
chmod 1777 /tmp/.X11-unix /tmp/.ICE-unix 2>/dev/null || true

port_ready() {
  if command -v curl >/dev/null 2>&1; then
    # KasmVNC returns 404 for `/`. Any response from /index.html (incl. 401)
    # means the websocket HTTP server is up.
    code="$(
      curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:${DESKTOP_VNC_PORT}/index.html" 2>/dev/null \
        || curl -sk -o /dev/null -w '%{http_code}' --max-time 2 \
          "https://127.0.0.1:${DESKTOP_VNC_PORT}/index.html" 2>/dev/null \
        || echo 000
    )"
    case "${code}" in
      200|301|302|401|403) return 0 ;;
      *) return 1 ;;
    esac
  fi
  if command -v bash >/dev/null 2>&1; then
    bash -c "echo >/dev/tcp/127.0.0.1/${DESKTOP_VNC_PORT}" >/dev/null 2>&1
    return $?
  fi
  return 1
}

if port_ready; then
  echo "Desktop already running on 127.0.0.1:${DESKTOP_VNC_PORT}"
  exit 0
fi

if ! command -v vncserver >/dev/null 2>&1; then
  echo "KasmVNC (vncserver) is not installed — Desktop unavailable" >&2
  exit 2
fi

# KasmVNC stores users in ~/.kasmpasswd (NOT ~/.vnc/passwd).
if [ ! -f "${HOME}/.kasmpasswd" ]; then
  printf '%s\n%s\n' "${VNC_PASS}" "${VNC_PASS}" \
    | vncpasswd -u "${VNC_USER}" -ow >/dev/null
fi

# Refresh config each start so image updates apply.
cp -f "${SCRIPT_DIR}/kasmvnc.yaml" "${VNC_DIR}/kasmvnc.yaml"
cp -f "${SCRIPT_DIR}/xstartup" "${VNC_DIR}/xstartup"
chmod +x "${VNC_DIR}/xstartup"
if command -v sed >/dev/null 2>&1; then
  sed -i "s/websocket_port:.*/websocket_port: ${DESKTOP_VNC_PORT}/" "${VNC_DIR}/kasmvnc.yaml" || true
fi

# Clean stale display if needed.
vncserver -kill ":${DISPLAY_NUM}" >/dev/null 2>&1 || true

EXTRA_ARGS=(-disableBasicAuth)

export DISPLAY=":${DISPLAY_NUM}"
{
  echo "==== $(date -Iseconds) starting desktop :${DISPLAY_NUM} port ${DESKTOP_VNC_PORT} ===="
  nice -n 5 vncserver ":${DISPLAY_NUM}" \
    -select-de xfce \
    -websocketPort "${DESKTOP_VNC_PORT}" \
    -interface 127.0.0.1 \
    "${EXTRA_ARGS[@]}"
} >>"${LOG_FILE}" 2>&1 || {
  echo "Desktop vncserver failed — see ${LOG_FILE}" >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  exit 1
}

# Wait up to ~12s for the web UI.
for _ in $(seq 1 24); do
  if port_ready; then
    echo "Desktop ready on 127.0.0.1:${DESKTOP_VNC_PORT}"
    exit 0
  fi
  sleep 0.5
done

echo "Desktop failed to become ready on port ${DESKTOP_VNC_PORT}" >&2
tail -n 60 "${LOG_FILE}" >&2 || true
ls -la "${VNC_DIR}" >&2 || true
exit 1
