#!/bin/sh
set -eu

HEALTHZ_PORT="${HEALTHZ_PORT:-8091}"
MSF_RPC_PORT="${MSF_RPC_PORT:-55553}"

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

start_msfrpcd() {
  if [ -z "${MSF_PASSWORD:-}" ]; then
    echo "INFO: MSF_PASSWORD unset — skipping msfrpcd (set env to enable RPC on :${MSF_RPC_PORT})"
    return 0
  fi
  if ! command -v msfrpcd >/dev/null 2>&1; then
    echo "WARN: msfrpcd not found" >&2
    return 0
  fi
  # Password only from runtime env — never bake secrets into the image.
  msfrpcd -P "${MSF_PASSWORD}" -S -a 0.0.0.0 -p "${MSF_RPC_PORT}" &
  echo "msfrpcd listening on 0.0.0.0:${MSF_RPC_PORT}"
}

note_gvm() {
  if command -v openvas >/dev/null 2>&1 || command -v gvm-cli >/dev/null 2>&1; then
    echo "INFO: OpenVAS/GVM CLI tools present — full gvmd+PostgreSQL+feed sync is not auto-started (see README)"
  else
    echo "INFO: OpenVAS/GVM not installed in this image build — use Greenbone sidecar or rebuild with GVM packages (README)"
  fi
}

start_healthz
note_gvm
start_msfrpcd

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

wait
