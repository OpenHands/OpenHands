"""REST handlers for the channel host.

Until these routes are mounted on the agent-server, run:

    python3 tools/channel_host_api.py --host 127.0.0.1 --port 18012
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from channel_host import ChannelError, ChannelHost, default_db_path

JsonBody = dict[str, Any] | None
Handler = Callable[[ChannelHost, dict[str, str], JsonBody], tuple[int, Any]]

CHANNELS_PATH = "/api/channels"
MESSAGES_PATH = "/api/channels/messages"
CHANNEL_PATH_RE = re.compile(r"^/api/channels/(?P<channel_id>[^/]+)$")
START_PATH_RE = re.compile(r"^/api/channels/(?P<channel_id>[^/]+)/start$")
STOP_PATH_RE = re.compile(r"^/api/channels/(?P<channel_id>[^/]+)/stop$")


def _query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    return {key: values[-1] for key, values in parse_qs(parsed.query).items()}


def _list_channels(
    host: ChannelHost, _params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, host.list_channels()


def _get_channel(
    host: ChannelHost, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, host.get_channel(params["channel_id"])


def _start_channel(
    host: ChannelHost, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, host.start(params["channel_id"])


def _stop_channel(
    host: ChannelHost, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    return 200, host.stop(params["channel_id"])


def _list_messages(
    host: ChannelHost, params: dict[str, str], _body: JsonBody
) -> tuple[int, Any]:
    limit = int(params.get("limit") or 50)
    offset = int(params.get("offset") or 0)
    items = host.list_messages(
        channel_id=params.get("channel") or params.get("channel_id"),
        direction=params.get("direction"),
        correlation_id=params.get("correlation_id"),
        limit=limit,
        offset=offset,
    )
    return 200, {"items": items, "limit": limit, "offset": offset}


ROUTES: tuple[tuple[str, re.Pattern[str], Handler], ...] = (
    ("GET", re.compile(rf"^{CHANNELS_PATH}$"), _list_channels),
    ("GET", re.compile(rf"^{MESSAGES_PATH}$"), _list_messages),
    ("POST", START_PATH_RE, _start_channel),
    ("POST", STOP_PATH_RE, _stop_channel),
    ("GET", CHANNEL_PATH_RE, _get_channel),
)


def handle_request(
    host: ChannelHost,
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
            return handler(host, params, body)
        return 404, {"error": f"No route for {method} {pathname}"}
    except ChannelError as exc:
        return exc.status, {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return 400, {"error": str(exc)}


class ChannelRequestHandler(BaseHTTPRequestHandler):
    server: "ChannelHTTPServer"

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
            self.server.host, self.command, self.path, payload
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


class ChannelHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        host: ChannelHost,
    ) -> None:
        super().__init__(server_address, ChannelRequestHandler)
        self.host = host


def serve_channels(
    listen_host: str,
    port: int,
    host: ChannelHost | None = None,
) -> ChannelHTTPServer:
    if host is None:
        host = ChannelHost(default_db_path())
    return ChannelHTTPServer((listen_host, port), host)


def main() -> None:
    parser = argparse.ArgumentParser(description="Channel-host HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18012)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    channel_host = ChannelHost(args.db or default_db_path())
    server = serve_channels(args.host, args.port, host=channel_host)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()
        channel_host.close()


if __name__ == "__main__":
    main()
