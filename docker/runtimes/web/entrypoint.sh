#!/bin/sh
set -eu

HEALTHZ_PORT="${HEALTHZ_PORT:-8090}"
ZAP_PORT="${ZAP_PORT:-8080}"
ZAP_HOST="${ZAP_HOST:-0.0.0.0}"

start_healthz() {
  python3 - "$HEALTHZ_PORT" <<'PY' &
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/healthz", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        return

HTTPServer(("0.0.0.0", port), Handler).serve_forever()
PY
  echo "healthz listening on :${HEALTHZ_PORT}"
}

start_zap() {
  if ! command -v zap >/dev/null 2>&1; then
    echo "WARN: zap not found on PATH" >&2
    return 0
  fi
  # Daemon API for scanners/MCP; port 8080 per image contract.
  zap -daemon -host "${ZAP_HOST}" -port "${ZAP_PORT}" -config api.disablekey=true &
  echo "ZAP daemon starting on ${ZAP_HOST}:${ZAP_PORT}"
}

start_healthz
start_zap

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

# Keep the container alive for engagement workspaces.
wait
