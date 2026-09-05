"""REST handlers for the feature-developer orchestrator.

Until these routes are mounted on the agent-server, run:

    python3 tools/feature_developer_api.py --host 127.0.0.1 --port 18009
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from feature_developer import FeatureDevError, FeatureDeveloper, default_db_path
from fleet import FleetError
from kanban import KanbanError

JsonBody = dict[str, Any] | None
Handler = Callable[[FeatureDeveloper, dict[str, str], JsonBody], tuple[int, Any]]

RUNS_PATH = "/api/feature-developer/runs"
RUN_PATH_RE = re.compile(r"^/api/feature-developer/runs/(?P<run_id>[^/]+)$")
PAUSE_PATH_RE = re.compile(r"^/api/feature-developer/runs/(?P<run_id>[^/]+)/pause$")
RESUME_PATH_RE = re.compile(r"^/api/feature-developer/runs/(?P<run_id>[^/]+)/resume$")
ABORT_PATH_RE = re.compile(r"^/api/feature-developer/runs/(?P<run_id>[^/]+)/abort$")
REPORT_PATH_RE = re.compile(
    r"^/api/feature-developer/runs/(?P<run_id>[^/]+)/report$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _list_runs(
    store: FeatureDeveloper, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.list_runs(
        project_id=params.get("project_id"),
        status=params.get("status"),
    )


def _start_run(
    store: FeatureDeveloper, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    run = store.start_run(
        project_id=str(payload.get("project_id") or ""),
        spec_text=str(payload.get("spec_text") or ""),
        max_concurrent_agents=payload.get("max_concurrent_agents"),
        continue_on_failure=bool(payload.get("continue_on_failure")),
        cost_cap=payload.get("cost_cap"),
    )
    return 201, run


def _get_run(
    store: FeatureDeveloper, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.get_run(params["run_id"])


def _pause(
    store: FeatureDeveloper, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.pause_run(params["run_id"])


def _resume(
    store: FeatureDeveloper, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.resume_run(params["run_id"])


def _abort(
    store: FeatureDeveloper, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, store.abort_run(params["run_id"])


def _report(
    store: FeatureDeveloper, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, {"markdown": store.report(params["run_id"])}


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{RUNS_PATH}$"), _list_runs),
    ("POST", re.compile(rf"^{RUNS_PATH}$"), _start_run),
    ("POST", PAUSE_PATH_RE, _pause),
    ("POST", RESUME_PATH_RE, _resume),
    ("POST", ABORT_PATH_RE, _abort),
    ("GET", REPORT_PATH_RE, _report),
    ("GET", RUN_PATH_RE, _get_run),
)


def handle_request(
    store: FeatureDeveloper,
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
    except (FeatureDevError, FleetError, KanbanError) as exc:
        return getattr(exc, "status", 400), {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class FeatureDevRequestHandler(BaseHTTPRequestHandler):
    server: "FeatureDevHTTPServer"

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


class FeatureDevHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: FeatureDeveloper,
    ) -> None:
        super().__init__(server_address, FeatureDevRequestHandler)
        self.store = store


def serve_feature_developer(
    host: str,
    port: int,
    store: FeatureDeveloper | None = None,
) -> FeatureDevHTTPServer:
    if store is None:
        store = FeatureDeveloper(db_path=default_db_path())
    return FeatureDevHTTPServer((host, port), store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-developer HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18009)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    store = FeatureDeveloper(db_path=args.db or default_db_path())
    server = serve_feature_developer(args.host, args.port, store=store)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
