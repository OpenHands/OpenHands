"""REST handlers for Meetily transcript ingest.

    python3 tools/meetily_api.py --host 127.0.0.1 --port 18013
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from meetily import MeetilyService

JsonBody = dict[str, Any] | None
Handler = Callable[[MeetilyService, dict[str, str], JsonBody], tuple[int, Any]]

TRANSCRIPT_PATH = "/api/meetings/transcript"
ACTION_PATH = "/api/meetings/action"
WEBHOOK_PATH = "/api/meetings/webhook"


def _json_body(body: JsonBody) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


def _preview_or_ingest(
    service: MeetilyService, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    text = str(payload.get("text") or payload.get("transcript") or "")
    if not text and payload.get("url"):
        text = str(payload.get("url"))
    fmt = payload.get("format")
    board_id = payload.get("board_id")
    if board_id:
        return 201, service.ingest(
            text,
            board_id=str(board_id),
            fmt=fmt,
            channel_id=payload.get("channel_id"),
            session_id=payload.get("session_id"),
            channel_ref=payload.get("channel_ref"),
            thread_ref=payload.get("thread_ref"),
        )
    return 200, service.preview(text, fmt)


def _extract_actions(
    service: MeetilyService, _params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    text = str(payload.get("text") or payload.get("transcript") or "")
    return 200, service.preview(
        text,
        payload.get("format"),
        board_id=payload.get("board_id"),
    )


def _webhook(
    service: MeetilyService, params: dict[str, str], body: JsonBody
) -> tuple[int, Any]:
    payload = _json_body(body)
    text = payload.get("text") or payload.get("transcript")
    if not text and isinstance(payload.get("utterances"), list):
        text = json.dumps({"utterances": payload["utterances"]})
    payload = {**payload, "text": text}
    return _preview_or_ingest(service, params, payload)


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("POST", re.compile(rf"^{TRANSCRIPT_PATH}$"), _preview_or_ingest),
    ("POST", re.compile(rf"^{ACTION_PATH}$"), _extract_actions),
    ("POST", re.compile(rf"^{WEBHOOK_PATH}$"), _webhook),
)


def handle_request(
    service: MeetilyService,
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
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class MeetilyRequestHandler(BaseHTTPRequestHandler):
    server: "MeetilyHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
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


class MeetilyHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: MeetilyService,
    ) -> None:
        super().__init__(server_address, MeetilyRequestHandler)
        self.service = service


def main() -> None:
    parser = argparse.ArgumentParser(description="Meetily HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18013)
    args = parser.parse_args()
    from kanban import KanbanStore, default_db_path

    service = MeetilyService(KanbanStore(default_db_path()))
    server = MeetilyHTTPServer((args.host, args.port), service)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
