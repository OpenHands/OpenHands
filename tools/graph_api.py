"""REST handlers for the codebase graph.

Until these routes are mounted on the agent-server, run:

    python3 tools/graph_api.py --host 127.0.0.1 --port 18012
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from codebase_graph import (
    DEFAULT_QUERY_BUDGET,
    GraphError,
    GraphStore,
    QUERY_KINDS,
    default_db_path,
    get_active_store,
    set_active_store,
)

JsonBody = dict[str, Any] | None
Handler = Callable[["GraphService", dict[str, str], JsonBody], tuple[int, Any]]

INDEX_PATH = "/api/graph/index"
STATUS_PATH = "/api/graph/index/status"
RETRIGGER_PATH = "/api/graph/index/retrigger"
QUERY_PATH = "/api/graph/query"
DEFINITIONS_PATH = "/api/graph/definitions"
CONFIG_PATH = "/api/graph/config"
IMPORT_PATH = "/api/graph/import-project-config"


class GraphService:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def close(self) -> None:
        self.store.close()


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _root(service: GraphService, params: dict[str, str], body: dict[str, Any]) -> str:
    root = str(body.get("root") or params.get("root") or "").strip()
    if not root:
        root = os.getcwd()
    return os.path.abspath(root)


def _post_index(service: GraphService, params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    payload = _json_body(body)
    root = _root(service, params, payload)
    full = bool(payload.get("full"))
    return 200, service.store.index(root, full=full)


def _get_status(service: GraphService, params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    root = params.get("root") or None
    return 200, service.store.status(root)


def _delete_index(service: GraphService, params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    root = _root(service, params, _json_body(body))
    return 200, service.store.clear(root)


def _retrigger(service: GraphService, params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    root = _root(service, params, _json_body(body))
    return 200, service.store.retrigger(root)


def _get_query(service: GraphService, params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    kind = str(params.get("q") or "").strip()
    if kind not in QUERY_KINDS:
        raise GraphError("q must be callers, deps, or usages")
    budget = int(params.get("budget") or DEFAULT_QUERY_BUDGET)
    return 200, service.store.query(
        kind,
        symbol=params.get("symbol") or None,
        file=params.get("file") or None,
        root=params.get("root") or None,
        budget=budget,
    )


def _get_definitions(service: GraphService, params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    symbol = str(params.get("symbol") or "").strip()
    root = str(params.get("root") or "").strip()
    if not root:
        raise GraphError("root is required")
    return 200, service.store.definitions(symbol, root)


def _get_config(service: GraphService, _params: dict[str, str], _body: JsonBody) -> tuple[int, Any]:
    return 200, service.store.get_config()


def _put_config(service: GraphService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    payload = _json_body(body)
    project_yaml = payload.pop("project_yaml", None)
    updated = service.store.put_config(payload)
    if project_yaml:
        try:
            service.store.import_project_config(str(project_yaml))
            updated = service.store.get_config()
        except Exception:
            pass
    return 200, updated


def _import_project(service: GraphService, _params: dict[str, str], body: JsonBody) -> tuple[int, Any]:
    path = str(_json_body(body).get("path") or "")
    if not path:
        raise GraphError("path is required")
    return 200, service.store.import_project_config(path)


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("POST", re.compile(rf"^{INDEX_PATH}$"), _post_index),
    ("GET", re.compile(rf"^{STATUS_PATH}$"), _get_status),
    ("DELETE", re.compile(rf"^{INDEX_PATH}$"), _delete_index),
    ("POST", re.compile(rf"^{RETRIGGER_PATH}$"), _retrigger),
    ("GET", re.compile(rf"^{QUERY_PATH}$"), _get_query),
    ("GET", re.compile(rf"^{DEFINITIONS_PATH}$"), _get_definitions),
    ("GET", re.compile(rf"^{CONFIG_PATH}$"), _get_config),
    ("PUT", re.compile(rf"^{CONFIG_PATH}$"), _put_config),
    ("POST", re.compile(rf"^{IMPORT_PATH}$"), _import_project),
)


def handle_request(
    service: GraphService,
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
            return handler(service, params, body)
        return 404, {"error": f"No route for {method} {pathname}"}
    except GraphError as exc:
        return exc.status, {"error": str(exc), **exc.payload}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class GraphRequestHandler(BaseHTTPRequestHandler):
    server: "GraphHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
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
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
        else:
            self.send_header("Content-Length", "0")
        self.end_headers()
        if body:
            self.wfile.write(body)


class GraphHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: GraphService,
    ) -> None:
        super().__init__(server_address, GraphRequestHandler)
        self.service = service


def serve_graph(
    host: str,
    port: int,
    service: GraphService | None = None,
) -> GraphHTTPServer:
    if service is None:
        store = get_active_store() or GraphStore(default_db_path())
        set_active_store(store)
        service = GraphService(store)
    return GraphHTTPServer((host, port), service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebase-graph HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18012)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    store = GraphStore(args.db or default_db_path())
    set_active_store(store)
    service = GraphService(store)
    server = serve_graph(args.host, args.port, service=service)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
