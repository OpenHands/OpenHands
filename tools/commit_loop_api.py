"""REST handlers for the commit-loop orchestrator.

Until these routes are mounted on the agent-server, run:

    python3 tools/commit_loop_api.py --host 127.0.0.1 --port 18008
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from commit_loop import CommitLoopError, CommitLoopService
from fleet import FleetError
from kanban import KanbanError
from loop_runner import LoopError, LoopStore, default_db_path as loop_db_path

JsonBody = dict[str, Any] | None
Handler = Callable[[CommitLoopService, dict[str, str], JsonBody], tuple[int, Any]]

SETUP_PATH_RE = re.compile(
    r"^/api/commit-loop/projects/(?P<project_id>[^/]+)/setup$"
)
STATUS_PATH_RE = re.compile(
    r"^/api/commit-loop/projects/(?P<project_id>[^/]+)/status$"
)
RUN_PATH_RE = re.compile(
    r"^/api/commit-loop/sessions/(?P<session_id>[^/]+)/run$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _setup(
    service: CommitLoopService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    definition = service.setup(
        params["project_id"],
        max_iterations=payload.get("max_iterations"),
        max_cost_usd=payload.get("max_cost_usd"),
    )
    return 201, definition


def _status(
    service: CommitLoopService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, service.status(params["project_id"])


def _run(
    service: CommitLoopService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    result = service.run_session(
        params["session_id"],
        worktree_dir=payload.get("worktree_dir"),
        push_pr=bool(payload.get("push_pr")),
    )
    return 200, result


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("POST", SETUP_PATH_RE, _setup),
    ("GET", STATUS_PATH_RE, _status),
    ("POST", RUN_PATH_RE, _run),
)


def handle_request(
    service: CommitLoopService,
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
    except (CommitLoopError, LoopError, FleetError, KanbanError) as exc:
        return getattr(exc, "status", 400), {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class CommitLoopRequestHandler(BaseHTTPRequestHandler):
    server: "CommitLoopHTTPServer"

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


class CommitLoopHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: CommitLoopService,
    ) -> None:
        super().__init__(server_address, CommitLoopRequestHandler)
        self.service = service


def serve_commit_loop(
    host: str,
    port: int,
    service: CommitLoopService | None = None,
) -> CommitLoopHTTPServer:
    if service is None:
        service = CommitLoopService(loop_store=LoopStore(loop_db_path()))
    return CommitLoopHTTPServer((host, port), service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local commit-loop HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18008)
    args = parser.parse_args()
    server = serve_commit_loop(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.service.close()


if __name__ == "__main__":
    main()
