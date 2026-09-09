"""REST handlers for loop triggers.

Until these routes are mounted on the agent-server, run:

    python3 tools/loop_triggers_api.py --host 127.0.0.1 --port 18010
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from loop_runner import LoopError, LoopStore, default_db_path as loop_db_path
from loop_triggers import (
    LoopTriggerService,
    TriggerError,
    default_db_path,
    set_active_service,
)

JsonBody = dict[str, Any] | None
Handler = Callable[[LoopTriggerService, dict[str, str], JsonBody], tuple[int, Any]]

TRIGGERS_PATH = "/api/loops/triggers"
EVENTS_FEED_PATH = "/api/loops/triggers/events"
TRIGGER_PATH_RE = re.compile(r"^/api/loops/triggers/(?P<trigger_id>[^/]+)$")
TRIGGER_EVENTS_PATH_RE = re.compile(
    r"^/api/loops/triggers/(?P<trigger_id>[^/]+)/events$"
)
TRIGGER_FIRE_PATH_RE = re.compile(
    r"^/api/loops/triggers/(?P<trigger_id>[^/]+)/fire$"
)


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() in ("1", "true", "yes")


def _list_triggers(
    service: LoopTriggerService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, service.list_triggers(
        project_id=params.get("project_id") or None,
        trigger_type=params.get("trigger_type") or None,
        enabled=_optional_bool(params.get("enabled")),
    )


def _create_trigger(
    service: LoopTriggerService, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    trigger = service.create_trigger(
        project_id=str(payload.get("project_id") or ""),
        loop_definition_id=str(payload.get("loop_definition_id") or ""),
        trigger_type=str(payload.get("trigger_type") or ""),
        schedule_type=payload.get("schedule_type"),
        cron_expr=payload.get("cron_expr"),
        interval_seconds=payload.get("interval_seconds"),
        payload=payload.get("payload"),
        enabled=payload.get("enabled", True),
    )
    return 201, trigger


def _patch_trigger(
    service: LoopTriggerService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    return 200, service.update_trigger(params["trigger_id"], **_json_body(body))


def _delete_trigger(
    service: LoopTriggerService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    service.delete_trigger(params["trigger_id"])
    return 204, None


def _list_trigger_events(
    service: LoopTriggerService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, service.list_events(
        trigger_id=params["trigger_id"],
        limit=int(params.get("limit") or 50),
        offset=int(params.get("offset") or 0),
    )


def _list_all_events(
    service: LoopTriggerService, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, service.list_events(
        limit=int(params.get("limit") or 50),
        offset=int(params.get("offset") or 0),
    )


def _fire_trigger(
    service: LoopTriggerService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    result = service.fire_trigger(
        params["trigger_id"],
        _json_body(body),
        ignore_enabled=True,
    )
    return 201, result


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{EVENTS_FEED_PATH}$"), _list_all_events),
    ("GET", re.compile(rf"^{TRIGGERS_PATH}$"), _list_triggers),
    ("POST", re.compile(rf"^{TRIGGERS_PATH}$"), _create_trigger),
    ("GET", TRIGGER_EVENTS_PATH_RE, _list_trigger_events),
    ("POST", TRIGGER_FIRE_PATH_RE, _fire_trigger),
    ("PATCH", TRIGGER_PATH_RE, _patch_trigger),
    ("DELETE", TRIGGER_PATH_RE, _delete_trigger),
)


def handle_request(
    service: LoopTriggerService,
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
    except TriggerError as exc:
        payload = {"error": str(exc), **exc.payload}
        return exc.status, payload
    except LoopError as exc:
        return exc.status, {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class TriggerRequestHandler(BaseHTTPRequestHandler):
    server: "TriggerHTTPServer"

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


class TriggerHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: LoopTriggerService,
    ) -> None:
        super().__init__(server_address, TriggerRequestHandler)
        self.service = service


def serve_loop_triggers(
    host: str,
    port: int,
    service: LoopTriggerService | None = None,
) -> TriggerHTTPServer:
    if service is None:
        service = LoopTriggerService(
            db_path=default_db_path(),
            loop_store=LoopStore(loop_db_path()),
        )
        set_active_service(service)
        service.start_scheduler()
    return TriggerHTTPServer((host, port), service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop-trigger HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18010)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    service = LoopTriggerService(
        db_path=args.db or default_db_path(),
        loop_store=LoopStore(loop_db_path()),
    )
    set_active_service(service)
    service.start_scheduler()
    server = serve_loop_triggers(args.host, args.port, service=service)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
