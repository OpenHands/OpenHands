"""REST handlers for fleet sessions and cost caps.

The dispatcher is stdlib-only so unit tests and a tiny HTTP server can
share one implementation. Until these routes are mounted on the
agent-server, run:

    python3 tools/fleet_api.py --host 127.0.0.1 --port 18005
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from fleet import FleetError, FleetStore, default_db_path
from kanban import KanbanError

JsonBody = dict[str, Any] | None
Handler = Callable[[FleetStore, dict[str, str], JsonBody], tuple[int, Any]]

SESSIONS_PATH = "/api/fleet/sessions"
SESSION_PATH_RE = re.compile(r"^/api/fleet/sessions/(?P<session_id>[^/]+)$")
CAPS_PATH = "/api/fleet/caps"
STATS_PATH = "/api/fleet/stats"


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _list_sessions(
    store: FleetStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_sessions(
        project_id=params.get("project_id"),
        status=params.get("status"),
        model=params.get("model"),
        created_after=params.get("created_after"),
        created_before=params.get("created_before"),
    )


def _spawn_session(
    store: FleetStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    session = store.spawn_session(
        project_id=str(payload.get("project_id") or ""),
        card_id=str(payload.get("card_id") or ""),
        model=payload.get("model"),
        agent_session_id=payload.get("agent_session_id"),
        branch_name=payload.get("branch_name"),
    )
    return 201, session


def _get_session(
    store: FleetStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.get_session(params["session_id"])


def _update_session(
    store: FleetStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    return 200, store.update_session(params["session_id"], **_json_body(body))


def _archive_session(
    store: FleetStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    store.archive_session(params["session_id"])
    return 204, None


def _list_caps(
    store: FleetStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_cost_caps()


def _set_cap(
    store: FleetStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    cap = store.set_cost_cap(
        project_id=str(payload.get("project_id") or ""),
        cap_usd=payload.get("cap_usd"),
        reset_period=payload.get("reset_period"),
    )
    return 201, cap


def _stats(
    store: FleetStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.stats()


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{SESSIONS_PATH}$"), _list_sessions),
    ("POST", re.compile(rf"^{SESSIONS_PATH}$"), _spawn_session),
    ("GET", SESSION_PATH_RE, _get_session),
    ("PATCH", SESSION_PATH_RE, _update_session),
    ("DELETE", SESSION_PATH_RE, _archive_session),
    ("GET", re.compile(rf"^{CAPS_PATH}$"), _list_caps),
    ("POST", re.compile(rf"^{CAPS_PATH}$"), _set_cap),
    ("GET", re.compile(rf"^{STATS_PATH}$"), _stats),
)


def handle_request(
    store: FleetStore,
    method: str,
    path: str,
    body: JsonBody = None,
) -> tuple[int, Any]:
    parsed = urlparse(path)
    pathname = parsed.path
    params = _query(path)
    try:
        for route_method, pattern, handler in ROUTES:
            if route_method != method:
                continue
            match = pattern.match(pathname)
            if match is None:
                continue
            params.update(match.groupdict())
            return handler(store, params, body)
        return 404, {"error": f"No route for {method} {pathname}"}
    except (FleetError, KanbanError) as exc:
        return getattr(exc, "status", 400), {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class FleetRequestHandler(BaseHTTPRequestHandler):
    server: "FleetHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
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


class FleetHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: FleetStore,
    ) -> None:
        super().__init__(server_address, FleetRequestHandler)
        self.store = store


def serve_fleet(
    host: str,
    port: int,
    store: FleetStore | None = None,
    db_path: str | None = None,
) -> FleetHTTPServer:
    if store is None:
        store = FleetStore(db_path or default_db_path())
    return FleetHTTPServer((host, port), store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fleet HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    server = serve_fleet(args.host, args.port, db_path=args.db)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
