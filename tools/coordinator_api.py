"""REST handlers for agent coordination.

The dispatcher is stdlib-only so unit tests and a tiny HTTP server can
share one implementation. Until these routes are mounted on the
agent-server, run:

    python3 tools/coordinator_api.py --host 127.0.0.1 --port 18006
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from coordinator import (
    CoordinatorError,
    CoordinatorStore,
    default_db_path,
    list_worker_agents,
)
from fleet import FleetError
from kanban import KanbanError

JsonBody = dict[str, Any] | None
Handler = Callable[[CoordinatorStore, dict[str, str], JsonBody], tuple[int, Any]]

SEQUENCES_PATH = "/api/coordinator/sequences"
SEQUENCE_PATH_RE = re.compile(
    r"^/api/coordinator/sequences/(?P<sequence_id>[^/]+)$"
)
SEQUENCE_ADVANCE_PATH_RE = re.compile(
    r"^/api/coordinator/sequences/(?P<sequence_id>[^/]+)/advance$"
)
AGENTS_PATH = "/api/coordinator/agents"
ASSIGN_PATH = "/api/coordinator/assign"


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _list_sequences(
    store: CoordinatorStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_sequences()


def _create_sequence(
    store: CoordinatorStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    sequence = store.create_sequence(
        project_id=str(payload.get("project_id") or ""),
        card_ids=list(payload.get("card_ids") or []),
        name=payload.get("name"),
    )
    return 201, sequence


def _get_sequence(
    store: CoordinatorStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.get_sequence(params["sequence_id"])


def _advance_sequence(
    store: CoordinatorStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    started = store.advance_sequence(params["sequence_id"])
    return 200, started


def _list_agents(
    _store: CoordinatorStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, list_worker_agents()


def _assign_card(
    store: CoordinatorStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    session = store.assign_card(
        project_id=str(payload.get("project_id") or ""),
        card_id=str(payload.get("card_id") or ""),
        task_type=payload.get("task_type"),
    )
    return 201, session


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{SEQUENCES_PATH}$"), _list_sequences),
    ("POST", re.compile(rf"^{SEQUENCES_PATH}$"), _create_sequence),
    ("POST", SEQUENCE_ADVANCE_PATH_RE, _advance_sequence),
    ("GET", SEQUENCE_PATH_RE, _get_sequence),
    ("GET", re.compile(rf"^{AGENTS_PATH}$"), _list_agents),
    ("POST", re.compile(rf"^{ASSIGN_PATH}$"), _assign_card),
)


def handle_request(
    store: CoordinatorStore,
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
    except (CoordinatorError, FleetError, KanbanError) as exc:
        return getattr(exc, "status", 400), {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class CoordinatorRequestHandler(BaseHTTPRequestHandler):
    server: "CoordinatorHTTPServer"

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


class CoordinatorHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: CoordinatorStore,
    ) -> None:
        super().__init__(server_address, CoordinatorRequestHandler)
        self.store = store


def serve_coordinator(
    host: str,
    port: int,
    store: CoordinatorStore | None = None,
    db_path: str | None = None,
) -> CoordinatorHTTPServer:
    if store is None:
        store = CoordinatorStore(db_path or default_db_path())
    return CoordinatorHTTPServer((host, port), store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local coordinator HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18006)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    server = serve_coordinator(args.host, args.port, db_path=args.db)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
