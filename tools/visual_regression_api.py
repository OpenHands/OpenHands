"""REST handlers for the visual-regression loop.

Until these routes are mounted on the agent-server, run:

    python3 tools/visual_regression_api.py --host 127.0.0.1 --port 18011
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from loop_runner import LoopError, LoopStore, default_db_path as loop_db_path
from visual_regression import VisualRegressionError, VisualRegressionService

JsonBody = dict[str, Any] | None
Handler = Callable[
    [VisualRegressionService, dict[str, str], JsonBody], tuple[int, Any]
]

SETUP_PATH_RE = re.compile(
    r"^/api/visual-regression/projects/(?P<project_id>[^/]+)/setup$"
)
BASELINE_PATH_RE = re.compile(
    r"^/api/visual-regression/projects/(?P<project_id>[^/]+)/capture-baseline$"
)
ARTIFACTS_PATH_RE = re.compile(
    r"^/api/visual-regression/runs/(?P<run_id>[^/]+)/artifacts$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _setup(
    service: VisualRegressionService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    definition = service.setup(params["project_id"], payload.get("config") or payload)
    return 201, definition


def _capture_baseline(
    service: VisualRegressionService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    worktree_dir = str(payload.get("worktree_dir") or "")
    if not worktree_dir:
        raise VisualRegressionError("worktree_dir is required")
    return 200, service.capture_baseline(params["project_id"], worktree_dir)


def _artifacts(
    service: VisualRegressionService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, service.artifacts(params["run_id"])


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("POST", SETUP_PATH_RE, _setup),
    ("POST", BASELINE_PATH_RE, _capture_baseline),
    ("GET", ARTIFACTS_PATH_RE, _artifacts),
)


def handle_request(
    service: VisualRegressionService,
    method: str,
    path: str,
    body: JsonBody = None,
) -> tuple[int, Any]:
    parsed = urlparse(path)
    pathname = parsed.path
    try:
        for route_method, pattern, handler in ROUTES:
            if route_method != method:
                continue
            match = pattern.match(pathname)
            if match is None:
                continue
            return handler(service, match.groupdict(), body)
        return 404, {"error": f"No route for {method} {pathname}"}
    except (VisualRegressionError, LoopError) as exc:
        return getattr(exc, "status", 400), {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class VisualRegressionRequestHandler(BaseHTTPRequestHandler):
    server: "VisualRegressionHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        payload: JsonBody = json.loads(raw) if raw else None
        status, data = handle_request(
            self.server.service, self.command, self.path, payload
        )
        body = b"" if data is None else json.dumps(data).encode()
        self.send_response(status)
        if body:
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
        else:
            self.send_header("Content-Length", "0")
        self.end_headers()
        if body:
            self.wfile.write(body)


class VisualRegressionHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: VisualRegressionService,
    ) -> None:
        super().__init__(server_address, VisualRegressionRequestHandler)
        self.service = service


def serve_visual_regression(
    host: str,
    port: int,
    service: VisualRegressionService | None = None,
) -> VisualRegressionHTTPServer:
    if service is None:
        service = VisualRegressionService(loop_store=LoopStore(loop_db_path()))
    return VisualRegressionHTTPServer((host, port), service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual-regression HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18011)
    args = parser.parse_args()
    server = serve_visual_regression(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.service.close()


if __name__ == "__main__":
    main()
