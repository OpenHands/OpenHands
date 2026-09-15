#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets>=13,<18"]
# ///
"""MARS port-forward tunnel client (MARSOHS-1427).

v1 Python port of doctl's `agents port-forward` reference implementation
(doctl/commands/agent_port_forward.go): dial the MARS port-forward WebSocket
for one session/port pair with a bearer token, and expose the result as an
ordinary local TCP listener that pipes bytes bidirectionally. Anything that
connects to that local port gets a plain, single-hop connection to whatever
is listening on the guest port.

Spawned as a subprocess by Canvas's Electron main process
(scripts/tunnel-client.mjs) via `uv run tools/mars_tunnel.py`, using the same
bundled Python runtime uv/uvx already provide for running the Agent Server
itself — no new runtime dependency, no dependency on a doctl release. The
inline PEP 723 metadata above lets `uv run` resolve `websockets` on its own,
the same way uvx resolves the Agent Server's own dependencies.

Protocol with the parent process:
  * The bearer token is read from the MARS_TUNNEL_ACCESS_TOKEN environment
    variable, not a CLI flag, so it never shows up in a `ps` listing.
  * Once the local TCP listener is bound, this prints exactly one line to
    stdout and flushes:
        TUNNEL_READY <local_port>
    Everything else on stdout/stderr is human-readable logging, not part of
    the protocol.
  * Shuts down cleanly on SIGTERM/SIGINT: stops accepting new connections and
    closes every connection currently being bridged.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from urllib.parse import urlsplit, urlunsplit

import websockets
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

COPY_BUF_SIZE = 32 * 1024
DEFAULT_API_URL = "https://api.digitalocean.com/"
REJECTION_BODY_MAX = 300


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MARS port-forward tunnel client")
    parser.add_argument("--session-id", required=True, help="MARS session id")
    parser.add_argument(
        "--remote-port", required=True, type=int, help="Port inside the guest sandbox"
    )
    parser.add_argument(
        "--local-port",
        default=0,
        type=int,
        help="Local port to listen on (0 lets the OS pick one)",
    )
    parser.add_argument("--address", default="127.0.0.1", help="Local bind address")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="harness-api base URL (http(s)://...); translated to ws(s)://",
    )
    return parser.parse_args(argv)


def build_tunnel_ws_url(api_url: str, session_id: str, remote_port: int) -> str:
    """Mirror doctl's hostedAgentsWSURL: swap http(s) for ws(s) and append the
    port-forward path, honoring whatever base path (if any) api_url carries.
    """
    parts = urlsplit(api_url)
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme)
    if scheme is None:
        raise ValueError(f"Unsupported API URL scheme: {parts.scheme!r}")
    path = (
        parts.path.rstrip("/")
        + f"/v2/agents/sessions/{session_id}/port-forward/{remote_port}"
    )
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def rejection_message(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    return text[:REJECTION_BODY_MAX]


async def pump_ws_to_local(ws, writer, log) -> None:
    try:
        async for message in ws:
            if isinstance(message, str):
                continue  # binary-only protocol; ignore stray text frames
            writer.write(message)
            await writer.drain()
    except (ConnectionClosed, ConnectionResetError, BrokenPipeError):
        pass
    except Exception as exc:  # defensive: never let one direction crash the process
        log(f"ws->local pump error: {exc}")


async def pump_local_to_ws(reader, ws, log) -> None:
    try:
        while True:
            chunk = await reader.read(COPY_BUF_SIZE)
            if not chunk:
                break
            await ws.send(chunk)
    except (ConnectionClosed, ConnectionResetError, BrokenPipeError):
        pass
    except Exception as exc:
        log(f"local->ws pump error: {exc}")


async def bridge_connection(reader, writer, ws_url: str, headers: dict, log) -> None:
    """Dial the tunnel for one accepted local connection and pump bytes both
    ways until either side closes. A per-connection failure never affects the
    listener or other connections.
    """
    try:
        async with ws_connect(
            ws_url, additional_headers=headers, max_size=None
        ) as ws:
            await asyncio.gather(
                pump_ws_to_local(ws, writer, log),
                pump_local_to_ws(reader, ws, log),
            )
    except InvalidStatus as exc:
        message = rejection_message(exc.response.body)
        log(
            f"server rejected tunnel ({exc.response.status_code} "
            f"{exc.response.reason_phrase}): {message}"
        )
    except WebSocketException as exc:
        log(f"tunnel dial failed: {exc}")
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def run(args: argparse.Namespace) -> int:
    access_token = os.environ.get("MARS_TUNNEL_ACCESS_TOKEN")
    if not access_token:
        print("MARS_TUNNEL_ACCESS_TOKEN is required", file=sys.stderr)
        return 1

    ws_url = build_tunnel_ws_url(args.api_url, args.session_id, args.remote_port)
    headers = {"Authorization": f"Bearer {access_token}"}

    def log(message: str) -> None:
        print(f"[mars-tunnel] {message}", file=sys.stderr, flush=True)

    async def handle_client(reader, writer):
        await bridge_connection(reader, writer, ws_url, headers, log)

    server = await asyncio.start_server(handle_client, args.address, args.local_port)
    local_port = server.sockets[0].getsockname()[1]

    # Printed as soon as the listener is bound, matching doctl's own
    # behavior — this does not wait for a real tunnel dial to succeed, only
    # for the local half to be ready to accept connections. The parent
    # process (scripts/tunnel-client.mjs) does its own TCP-connect health
    # check against this port before treating the tunnel as ready.
    print(f"TUNNEL_READY {local_port}", flush=True)
    log(
        f"forwarding {args.address}:{local_port} -> port {args.remote_port} "
        f"in session {args.session_id}"
    )

    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def request_stop() -> None:
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass  # Windows: no add_signal_handler; Ctrl-C raises KeyboardInterrupt instead

    async with server:
        await stop
        server.close()
        await server.wait_closed()

    return 0


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
