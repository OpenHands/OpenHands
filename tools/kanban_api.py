"""REST handlers for the local kanban store.

The dispatcher is intentionally stdlib-only so unit tests and a tiny HTTP
server can share one implementation. Mounting these routes on the agent-server
is a later SDK change; until then run:

    python3 tools/kanban_api.py --host 127.0.0.1 --port 18002
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from kanban import KanbanError, KanbanStore, default_db_path
from kanban_agent import complete_session, link_session, record_progress
from cost_estimator import apply_estimate
from project_bootstrap import (
    DecompositionError,
    init_project,
    preview_project,
)

JsonBody = dict[str, Any] | None
Handler = Callable[[KanbanStore, dict[str, str], JsonBody], tuple[int, Any]]

BOARDS_PATH = "/api/boards"
BOARD_PATH_RE = re.compile(r"^/api/boards/(?P<board_id>[^/]+)$")
BOARD_COLUMNS_PATH_RE = re.compile(
    r"^/api/boards/(?P<board_id>[^/]+)/columns$"
)
BOARD_COSTS_PATH_RE = re.compile(r"^/api/boards/(?P<board_id>[^/]+)/costs$")
COLUMN_PATH_RE = re.compile(r"^/api/columns/(?P<column_id>[^/]+)$")
COLUMN_CARDS_PATH_RE = re.compile(r"^/api/columns/(?P<column_id>[^/]+)/cards$")
CARD_PATH_RE = re.compile(r"^/api/cards/(?P<card_id>[^/]+)$")
CARD_MOVE_PATH_RE = re.compile(r"^/api/cards/(?P<card_id>[^/]+)/move$")
CARD_LINK_SESSION_PATH_RE = re.compile(
    r"^/api/cards/(?P<card_id>[^/]+)/link-session$"
)
CARD_PROGRESS_PATH_RE = re.compile(r"^/api/cards/(?P<card_id>[^/]+)/progress$")
CARD_COMPLETE_PATH_RE = re.compile(r"^/api/cards/(?P<card_id>[^/]+)/complete$")
CARD_ESTIMATE_PATH_RE = re.compile(r"^/api/cards/(?P<card_id>[^/]+)/estimate$")
PROJECT_PREVIEW_PATH = "/api/project/preview"
PROJECT_INIT_PATH = "/api/project/init"


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _list_boards(
    store: KanbanStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_boards()


def _create_board(
    store: KanbanStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    board = store.create_board(
        name=str(payload.get("name") or ""),
        project_id=payload.get("project_id"),
    )
    return 201, board


def _get_board(
    store: KanbanStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.get_board(params["board_id"])


def _add_column(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    column = store.add_column(
        params["board_id"],
        name=str(payload.get("name") or ""),
        color=payload.get("color"),
        position=payload.get("position"),
    )
    return 201, column


def _board_costs(
    store: KanbanStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.board_costs(params["board_id"])


def _update_column(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    column = store.update_column(
        params["column_id"],
        name=payload.get("name"),
        position=payload.get("position"),
        color=payload.get("color"),
    )
    return 200, column


def _delete_column(
    store: KanbanStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    store.delete_column(params["column_id"])
    return 204, None


def _create_card(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    title = str(payload.pop("title", "") or "")
    card = store.create_card(params["column_id"], title, **payload)
    return 201, card


def _update_card(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    return 200, store.update_card(params["card_id"], **_json_body(body))


def _delete_card(
    store: KanbanStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    store.delete_card(params["card_id"])
    return 204, None


def _move_card(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    column_id = payload.get("column_id")
    if not column_id:
        raise KanbanError("column_id is required")
    position = payload.get("position", 0)
    return 200, store.move_card(params["card_id"], str(column_id), int(position))


def _link_session(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    return 200, link_session(
        store, params["card_id"], str(payload.get("session_id") or "")
    )


def _record_progress(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    return 200, record_progress(
        store,
        params["card_id"],
        str(payload.get("message") or ""),
        payload.get("status"),
    )


def _complete_session(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    return 200, complete_session(
        store,
        params["card_id"],
        tests_passed=bool(payload.get("tests_passed")),
        actual_tokens=payload.get("actual_tokens"),
        actual_cost=payload.get("actual_cost"),
        tool_calls=payload.get("tool_calls"),
        agent_time=payload.get("agent_time"),
        model_used=payload.get("model_used"),
    )


def _estimate_card(
    store: KanbanStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    return 200, apply_estimate(
        store,
        params["card_id"],
        model=payload.get("model"),
        cost_cap=payload.get("cost_cap"),
    )


def _project_preview(
    _store: KanbanStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    cards = preview_project(
        payload.get("root") or os.getcwd(),
        payload.get("spec"),
    )
    return 200, {"suggested": [card.__dict__ for card in cards]}


def _project_init(
    store: KanbanStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    result = init_project(
        payload.get("root") or os.getcwd(),
        payload.get("spec"),
        store,
        board_name=str(payload.get("board_name") or "Project board"),
    )
    return 201, result


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{BOARDS_PATH}$"), _list_boards),
    ("POST", re.compile(rf"^{BOARDS_PATH}$"), _create_board),
    ("GET", BOARD_COSTS_PATH_RE, _board_costs),
    ("POST", BOARD_COLUMNS_PATH_RE, _add_column),
    ("GET", BOARD_PATH_RE, _get_board),
    ("PATCH", COLUMN_PATH_RE, _update_column),
    ("DELETE", COLUMN_PATH_RE, _delete_column),
    ("POST", COLUMN_CARDS_PATH_RE, _create_card),
    ("PATCH", CARD_PATH_RE, _update_card),
    ("DELETE", CARD_PATH_RE, _delete_card),
    ("POST", CARD_MOVE_PATH_RE, _move_card),
    ("POST", CARD_LINK_SESSION_PATH_RE, _link_session),
    ("POST", CARD_PROGRESS_PATH_RE, _record_progress),
    ("POST", CARD_COMPLETE_PATH_RE, _complete_session),
    ("POST", CARD_ESTIMATE_PATH_RE, _estimate_card),
    ("POST", re.compile(rf"^{PROJECT_PREVIEW_PATH}$"), _project_preview),
    ("POST", re.compile(rf"^{PROJECT_INIT_PATH}$"), _project_init),
)


def handle_request(
    store: KanbanStore,
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
    except DecompositionError as exc:
        return 400, {"error": str(exc)}
    except KanbanError as exc:
        return exc.status, {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class KanbanRequestHandler(BaseHTTPRequestHandler):
    server: "KanbanHTTPServer"

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


class KanbanHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: KanbanStore,
    ) -> None:
        super().__init__(server_address, KanbanRequestHandler)
        self.store = store


def serve_kanban(
    host: str,
    port: int,
    store: KanbanStore | None = None,
    db_path: str | None = None,
) -> KanbanHTTPServer:
    if store is None:
        store = KanbanStore(db_path or default_db_path())
    return KanbanHTTPServer((host, port), store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local kanban HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18002)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    server = serve_kanban(args.host, args.port, db_path=args.db)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
