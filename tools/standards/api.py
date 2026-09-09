"""REST handlers for standards plugins.

Until these routes are mounted on the agent-server, run:

    python3 tools/standards_api.py --host 127.0.0.1 --port 18004
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from standards.audit_store import DEFAULT_AUDIT_LIMIT, default_db_path
from standards.registry import (
    StandardsRegistry,
    get_active_registry,
    set_active_registry,
    start_default_registry,
)

JsonBody = dict[str, Any] | None
Handler = Callable[["StandardsService", dict[str, str], JsonBody], tuple[int, Any]]

PLUGINS_PATH = "/api/standards/plugins"
CONFIG_PATH = "/api/standards/config"
RUN_PATH = "/api/standards/run"
AUDIT_PATH = "/api/standards/audit"


class StandardsError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class StandardsService:
    def __init__(self, registry: StandardsRegistry) -> None:
        self.registry = registry

    def close(self) -> None:
        self.registry.close()


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _get_plugins(
    service: StandardsService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    root = params.get("root") or None
    if root:
        service.registry.discover(root)
    else:
        service.registry.discover()
    return 200, {
        "plugins": service.registry.plugin_summaries(root),
        "load_errors": [item.as_dict() for item in service.registry.load_errors()],
    }


def _get_config(
    service: StandardsService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    root = params.get("root") or None
    if root:
        service.registry.discover(root)
    return 200, service.registry.load_config(root)


def _put_config(
    service: StandardsService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    root = str(payload.get("root") or params.get("root") or "").strip() or None
    if root:
        service.registry.discover(root)
    return 200, service.registry.save_config(payload, root)


def _post_run(
    service: StandardsService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    root = str(payload.get("root") or params.get("root") or "").strip()
    if not root:
        raise StandardsError("root is required")
    if not os.path.isdir(root):
        raise StandardsError("root is not a directory")
    service.registry.discover(root)
    names = payload.get("enabled_names")
    if names is not None and not isinstance(names, list):
        raise StandardsError("enabled_names must be a list")
    files = payload.get("files")
    if files is not None and not isinstance(files, list):
        raise StandardsError("files must be a list")
    run = service.registry.run_checks(
        root,
        enabled_names=[str(item) for item in names] if names is not None else None,
        files=[str(item) for item in files] if files is not None else None,
        persist_audit=True,
        project_root=root,
    )
    payload_out = run.as_dict()
    return 200, {
        "run_id": payload_out["run_id"],
        "summary": payload_out["summary"],
        "violations": payload_out["violations"],
        "duration_ms": payload_out["duration_ms"],
        "status": payload_out["status"],
        "started_at": payload_out["started_at"],
        "worktree": payload_out["worktree"],
    }


def _get_audit(
    service: StandardsService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    limit_raw = params.get("limit")
    try:
        limit = int(limit_raw) if limit_raw else DEFAULT_AUDIT_LIMIT
    except ValueError as exc:
        raise StandardsError("limit must be an integer") from exc
    return 200, service.registry.audit.list_violations(
        run_id=params.get("run_id") or None,
        plugin=params.get("plugin") or None,
        severity=params.get("severity") or None,
        file=params.get("file") or None,
        limit=limit,
        before_id=params.get("before_id") or None,
    )


ROUTES: list[tuple[str, re.Pattern[str], Handler]] = [
    ("GET", re.compile(rf"^{PLUGINS_PATH}$"), _get_plugins),
    ("GET", re.compile(rf"^{CONFIG_PATH}$"), _get_config),
    ("PUT", re.compile(rf"^{CONFIG_PATH}$"), _put_config),
    ("POST", re.compile(rf"^{RUN_PATH}$"), _post_run),
    ("GET", re.compile(rf"^{AUDIT_PATH}$"), _get_audit),
]


def handle_request(
    service: StandardsService,
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
    except StandardsError as exc:
        return exc.status, {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class StandardsRequestHandler(BaseHTTPRequestHandler):
    server: "StandardsHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        expected = os.environ.get("OH_SESSION_API_KEYS_0") or os.environ.get(
            "SESSION_API_KEY"
        )
        if expected and self.headers.get("X-Session-API-Key") != expected:
            body = json.dumps({"error": "Unauthorized"}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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


class StandardsHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: StandardsService,
    ) -> None:
        super().__init__(server_address, StandardsRequestHandler)
        self.service = service


def serve_standards(
    host: str,
    port: int,
    service: StandardsService | None = None,
) -> StandardsHTTPServer:
    if service is None:
        registry = get_active_registry() or start_default_registry()
        if registry is None:
            registry = StandardsRegistry(default_db_path())
            registry.discover()
            set_active_registry(registry)
        service = StandardsService(registry)
    return StandardsHTTPServer((host, port), service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standards plugins HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18004)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    registry = StandardsRegistry(args.db or default_db_path())
    registry.discover()
    set_active_registry(registry)
    service = StandardsService(registry)
    server = serve_standards(args.host, args.port, service=service)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
