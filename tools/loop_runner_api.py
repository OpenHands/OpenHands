"""REST handlers for the loop runner store.

The dispatcher is stdlib-only so unit tests and a tiny HTTP server can
share one implementation. Until these routes are mounted on the
agent-server, run:

    python3 tools/loop_runner_api.py --host 127.0.0.1 --port 18007
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from loop_runner import LoopError, LoopStore, default_db_path

JsonBody = dict[str, Any] | None
Handler = Callable[[LoopStore, dict[str, str], JsonBody], tuple[int, Any]]

LOOPS_PATH = "/api/loops"
LOOP_RUNS_PATH_RE = re.compile(r"^/api/loops/(?P<definition_id>[^/]+)/runs$")
RUN_PATH_RE = re.compile(r"^/api/loops/runs/(?P<run_id>[^/]+)$")
RUN_FIX_PATH_RE = re.compile(r"^/api/loops/runs/(?P<run_id>[^/]+)/request-fix$")
RUN_RETRY_PATH_RE = re.compile(r"^/api/loops/runs/(?P<run_id>[^/]+)/retry-stage$")
RUN_ABORT_PATH_RE = re.compile(r"^/api/loops/runs/(?P<run_id>[^/]+)/abort$")
RUN_CHECKPOINT_PATH_RE = re.compile(
    r"^/api/loops/runs/(?P<run_id>[^/]+)/resume-checkpoint$"
)
RUN_FEEDBACK_PATH_RE = re.compile(
    r"^/api/loops/runs/(?P<run_id>[^/]+)/feedback$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _list_definitions(
    store: LoopStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_definitions()


def _create_definition(
    store: LoopStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    definition = store.create_definition(
        name=str(payload.get("name") or ""),
        project_id=str(payload.get("project_id") or ""),
        stages=list(payload.get("stages") or []),
        max_iterations=payload.get("max_iterations"),
        max_cost_usd=payload.get("max_cost_usd"),
        on_failure=payload.get("on_failure"),
    )
    return 201, definition


def _list_runs(
    store: LoopStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_runs(params["definition_id"])


def _start_run(
    store: LoopStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    worktree_dir = str(payload.get("worktree_dir") or "")
    if not worktree_dir:
        raise LoopError("worktree_dir is required")
    run = store.start_run(
        params["definition_id"],
        worktree_dir=worktree_dir,
        session_id=payload.get("session_id"),
    )
    return 201, run


def _get_run(
    store: LoopStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.get_run(params["run_id"])


def _request_fix(
    store: LoopStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.request_fix(params["run_id"])


def _retry_stage(
    store: LoopStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.retry_stage(params["run_id"])


def _abort_run(
    store: LoopStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.abort_run(params["run_id"])


def _resume_checkpoint(
    store: LoopStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.resume_checkpoint(params["run_id"])


def _feedback(
    store: LoopStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    if "approve" not in payload:
        raise LoopError("approve is required")
    return 200, store.submit_feedback(
        params["run_id"],
        approve=bool(payload.get("approve")),
        note=payload.get("note"),
    )


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{LOOPS_PATH}$"), _list_definitions),
    ("POST", re.compile(rf"^{LOOPS_PATH}$"), _create_definition),
    ("GET", LOOP_RUNS_PATH_RE, _list_runs),
    ("POST", LOOP_RUNS_PATH_RE, _start_run),
    ("GET", RUN_CHECKPOINT_PATH_RE, _resume_checkpoint),
    ("POST", RUN_FIX_PATH_RE, _request_fix),
    ("POST", RUN_RETRY_PATH_RE, _retry_stage),
    ("POST", RUN_ABORT_PATH_RE, _abort_run),
    ("POST", RUN_FEEDBACK_PATH_RE, _feedback),
    ("GET", RUN_PATH_RE, _get_run),
)


def handle_request(
    store: LoopStore,
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
            return handler(store, match.groupdict(), body)
        return 404, {"error": f"No route for {method} {pathname}"}
    except LoopError as exc:
        return exc.status, {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class LoopRequestHandler(BaseHTTPRequestHandler):
    server: "LoopHTTPServer"

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
            self.server.store, self.command, self.path, payload
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


class LoopHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: LoopStore,
    ) -> None:
        super().__init__(server_address, LoopRequestHandler)
        self.store = store


def serve_loops(
    host: str,
    port: int,
    store: LoopStore | None = None,
    db_path: str | None = None,
) -> LoopHTTPServer:
    if store is None:
        store = LoopStore(db_path or default_db_path())
    return LoopHTTPServer((host, port), store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local loop-runner HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18007)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    server = serve_loops(args.host, args.port, db_path=args.db)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
