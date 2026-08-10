#!/bin/sh
set -eu

HEALTHZ_PORT="${HEALTHZ_PORT:-8094}"

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

start_healthz

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

wait
