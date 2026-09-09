"""REST handlers for perf, manual, and doc loops.

Until these routes are mounted on the agent-server, run:

    python3 tools/perf_manual_doc_api.py --host 127.0.0.1 --port 18012
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from doc_loop import DocLoopError, DocLoopService
from loop_runner import LoopError, LoopStore, default_db_path as loop_db_path
from manual_loop import ManualLoopError, ManualLoopService
from perf_loop import DEFAULT_PERF_RULES, PerfLoopError, PerfLoopService

JsonBody = dict[str, Any] | None


class PerfManualDocService:
    def __init__(self, loop_store: LoopStore | None = None) -> None:
        self.loop_store = loop_store or LoopStore()
        self._owns_loop_store = loop_store is None
        self.perf = PerfLoopService(self.loop_store)
        self.manual = ManualLoopService(self.loop_store)
        self.doc = DocLoopService(self.loop_store)

    def close(self) -> None:
        if self._owns_loop_store:
            self.loop_store.close()

    def setup(
        self, project_id: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = config or {}
        return {
            "perf": self.perf.setup(project_id, payload.get("perf")),
            "manual": self.manual.setup(project_id, payload.get("manual")),
            "doc": self.doc.setup(project_id, payload.get("doc")),
        }

    def status(self, project_id: str) -> dict[str, Any]:
        return {
            "perf": self._safe_status(self.perf.status, project_id),
            "manual": self._safe_status(self.manual.status, project_id),
            "doc": self._safe_status(self.doc.status, project_id),
        }

    def _safe_status(
        self, fn: Callable[[str], dict[str, Any]], project_id: str
    ) -> dict[str, Any] | None:
        try:
            return fn(project_id)
        except (PerfLoopError, ManualLoopError, DocLoopError):
            return None


Handler = Callable[
    [PerfManualDocService, dict[str, str], JsonBody], tuple[int, Any]
]

SETUP_PATH_RE = re.compile(
    r"^/api/perf-manual-doc/projects/(?P<project_id>[^/]+)/setup$"
)
STATUS_PATH_RE = re.compile(
    r"^/api/perf-manual-doc/projects/(?P<project_id>[^/]+)/status$"
)
RULES_PATH = "/api/perf-manual-doc/perf-rules"
FEEDBACK_PATH_RE = re.compile(
    r"^/api/loops/runs/(?P<run_id>[^/]+)/feedback$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _setup(
    service: PerfManualDocService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    return 201, service.setup(params["project_id"], _json_body(body))


def _status(
    service: PerfManualDocService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, service.status(params["project_id"])


def _rules(
    _service: PerfManualDocService, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, DEFAULT_PERF_RULES


def _feedback(
    service: PerfManualDocService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    if "approve" not in payload:
        raise LoopError("approve is required")
    return 200, service.loop_store.submit_feedback(
        params["run_id"],
        approve=bool(payload.get("approve")),
        note=payload.get("note"),
    )


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("POST", SETUP_PATH_RE, _setup),
    ("GET", STATUS_PATH_RE, _status),
    ("GET", re.compile(rf"^{RULES_PATH}$"), _rules),
    ("POST", FEEDBACK_PATH_RE, _feedback),
)


def handle_request(
    service: PerfManualDocService,
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
    except (PerfLoopError, ManualLoopError, DocLoopError, LoopError) as exc:
        return getattr(exc, "status", 400), {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class PerfManualDocRequestHandler(BaseHTTPRequestHandler):
    server: "PerfManualDocHTTPServer"

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


class PerfManualDocHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: PerfManualDocService,
    ) -> None:
        super().__init__(server_address, PerfManualDocRequestHandler)
        self.service = service


def serve_perf_manual_doc(
    host: str,
    port: int,
    service: PerfManualDocService | None = None,
) -> PerfManualDocHTTPServer:
    if service is None:
        service = PerfManualDocService(loop_store=LoopStore(loop_db_path()))
    return PerfManualDocHTTPServer((host, port), service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perf/manual/doc loop HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18012)
    args = parser.parse_args()
    server = serve_perf_manual_doc(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        server.service.close()


if __name__ == "__main__":
    main()
