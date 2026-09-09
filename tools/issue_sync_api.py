"""REST handlers for optional JIRA/Linear/Plane kanban sync.

This module is self-contained and is not mounted on the agent-server.
Run it standalone when a project opts into external issue tracking:

    python3 tools/issue_sync_api.py --host 127.0.0.1 --port 18007
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from issue_sync import IssueSyncError, IssueSyncStore, default_db_path
from kanban import KanbanError

JsonBody = dict[str, Any] | None
Handler = Callable[[IssueSyncStore, dict[str, str], JsonBody], tuple[int, Any]]

PROVIDERS_PATH = "/api/issue-sync/providers"
MAPPINGS_PATH = "/api/issue-sync/mappings"
SYNC_PATH = "/api/issue-sync/sync"
CARD_PUSH_PATH_RE = re.compile(
    r"^/api/issue-sync/cards/(?P<card_id>[^/]+)/push$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _list_providers(
    store: IssueSyncStore, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_providers()


def _add_provider(
    store: IssueSyncStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    kind = str(payload.pop("kind", "") or "")
    name = str(payload.pop("name", "") or "")
    api_key = str(payload.pop("api_key", "") or "")
    base_url = payload.pop("base_url", None)
    provider = store.add_provider(
        kind, name, api_key, base_url=base_url, **payload
    )
    return 201, provider


def _list_mappings(
    store: IssueSyncStore, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_mappings(params.get("provider_id"))


def _set_mapping(
    store: IssueSyncStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    mapping = store.set_mapping(
        str(payload.get("provider_id") or ""),
        str(payload.get("local_column") or ""),
        str(payload.get("remote_status") or ""),
    )
    return 201, mapping


def _sync(
    store: IssueSyncStore, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    return 200, store.sync(
        provider_id=payload.get("provider_id"),
        direction=str(payload.get("direction") or "both"),
    )


def _push_card(
    store: IssueSyncStore, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    provider_id = payload.get("provider_id")
    if not provider_id:
        raise IssueSyncError("provider_id is required")
    return 200, store.push_card(params["card_id"], str(provider_id))


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{PROVIDERS_PATH}$"), _list_providers),
    ("POST", re.compile(rf"^{PROVIDERS_PATH}$"), _add_provider),
    ("GET", re.compile(rf"^{MAPPINGS_PATH}$"), _list_mappings),
    ("POST", re.compile(rf"^{MAPPINGS_PATH}$"), _set_mapping),
    ("POST", re.compile(rf"^{SYNC_PATH}$"), _sync),
    ("POST", CARD_PUSH_PATH_RE, _push_card),
)


def handle_request(
    store: IssueSyncStore,
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
    except (IssueSyncError, KanbanError) as exc:
        return getattr(exc, "status", 400), {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class IssueSyncRequestHandler(BaseHTTPRequestHandler):
    server: "IssueSyncHTTPServer"

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


class IssueSyncHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: IssueSyncStore,
    ) -> None:
        super().__init__(server_address, IssueSyncRequestHandler)
        self.store = store


def serve_issue_sync(
    host: str,
    port: int,
    store: IssueSyncStore | None = None,
    db_path: str | None = None,
) -> IssueSyncHTTPServer:
    if store is None:
        store = IssueSyncStore(db_path or default_db_path())
    return IssueSyncHTTPServer((host, port), store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local issue-sync HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18007)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    server = serve_issue_sync(args.host, args.port, db_path=args.db)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
