"""Shared HTTP and WebSocket proxy mechanics.

Both the VS Code proxy (vscode_proxy_router) and agent-server proxy
(agent_proxy_router) use the same underlying transport logic.  This module
provides that logic once to avoid duplication.

Each router is responsible for:
  - looking up the correct internal URL for the target service,
  - building the final target URL (VS Code keeps the full path; the agent
    proxy strips its own prefix),
  - assembling the forwarding headers (VS Code adds an Origin header; the
    agent proxy drops it).

Then each router delegates here for the actual HTTP / WebSocket proxying.
"""

import asyncio
import logging

import aiohttp
import httpx
from fastapi import WebSocket, WebSocketDisconnect, status
from fastapi.background import BackgroundTasks
from fastapi.responses import Response, StreamingResponse

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Hop-by-hop headers that must not be forwarded in either direction.
HTTP_HOP_BY_HOP: frozenset[str] = frozenset(
    [
        'connection',
        'keep-alive',
        'proxy-authenticate',
        'proxy-authorization',
        'proxy-connection',
        'te',
        'trailers',
        'transfer-encoding',
        'upgrade',
    ]
)

# Request headers the proxy itself controls; strip from the incoming request.
HTTP_REQUEST_SKIP: frozenset[str] = HTTP_HOP_BY_HOP | frozenset(['host'])


# ---------------------------------------------------------------------------
# HTTP proxy
# ---------------------------------------------------------------------------


async def do_http_proxy(
    method: str,
    target_url: str,
    req_headers: dict[str, str],
    body_stream,
    label: str = 'upstream',
) -> Response:
    """Forward an HTTP request to *target_url* and stream the response back.

    Args:
        method: HTTP method (GET, POST, …).
        target_url: Fully-qualified URL including path and query string.
        req_headers: Pre-filtered headers to forward (caller's responsibility).
        body_stream: Async byte-stream for the request body (``request.stream()``).
        label: Short identifier used in log/error messages (e.g. ``"vscode:abc123"``).
    """
    client = httpx.AsyncClient(follow_redirects=False, timeout=60.0)
    try:
        upstream_req = client.build_request(
            method,
            target_url,
            headers=req_headers,
            content=body_stream,
        )
        upstream = await client.send(upstream_req, stream=True)
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        await client.aclose()
        _logger.warning('HTTP proxy connect error (%s): %s', label, exc)
        return Response(
            content=f'Could not connect to {label}',
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    resp_headers = {
        k: v
        for k, v in upstream.headers.multi_items()
        if k.lower() not in HTTP_HOP_BY_HOP
    }

    async def _cleanup() -> None:
        await upstream.aclose()
        await client.aclose()

    background = BackgroundTasks()
    background.add_task(_cleanup)

    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=resp_headers,
        background=background,
    )


# ---------------------------------------------------------------------------
# WebSocket proxy
# ---------------------------------------------------------------------------


async def _relay_up(
    browser_ws: WebSocket,
    upstream_ws: aiohttp.ClientWebSocketResponse,
    label: str,
) -> None:
    """Forward frames browser → upstream until the browser disconnects."""
    try:
        while True:
            msg = await browser_ws.receive()
            if msg['type'] == 'websocket.disconnect':
                break
            if (text := msg.get('text')) is not None:
                await upstream_ws.send_str(text)
            elif (data := msg.get('bytes')) is not None:
                await upstream_ws.send_bytes(data)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        _logger.warning('WS proxy browser→%s relay error: %s', label, exc)


async def _relay_down(
    browser_ws: WebSocket,
    upstream_ws: aiohttp.ClientWebSocketResponse,
    label: str,
) -> int:
    """Forward frames upstream → browser until the upstream closes.

    Returns the WS close code sent by the upstream (1000 if not available).
    """
    close_code = 1000
    try:
        async for msg in upstream_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await browser_ws.send_text(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await browser_ws.send_bytes(msg.data)
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                close_code = int(msg.data) if msg.data else 1000
                _logger.info(
                    'WS upstream sent CLOSE (%s): code=%s reason=%r',
                    label,
                    close_code,
                    msg.extra,
                )
                break
            elif msg.type in (aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                _logger.info(
                    'WS upstream sent %s (%s)', msg.type.name, label
                )
                break
    except Exception as exc:
        _logger.warning('WS proxy %s→browser relay error: %s', label, exc)
    return close_code


async def do_ws_proxy(
    websocket: WebSocket,
    target_ws_url: str,
    forward_headers: dict[str, str],
    label: str = 'upstream',
) -> None:
    """Proxy a WebSocket connection bidirectionally to *target_ws_url*.

    Accepts the browser WebSocket, connects to the upstream, and relays
    frames in both directions until either side closes.  Sends an explicit
    WS close frame to the browser so it receives code 1000 (normal closure)
    rather than an abrupt 1006 drop, which would be treated as an error by
    the frontend and trigger an unnecessary reconnect error message.

    Args:
        websocket: Incoming browser WebSocket (not yet accepted).
        target_ws_url: Fully-qualified ``ws://`` or ``wss://`` URL.
        forward_headers: Pre-filtered headers to send on the upstream handshake.
        label: Short identifier for log messages.
    """
    _logger.debug('WS proxy: %s → %s', websocket.url, target_ws_url)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                target_ws_url,
                headers=forward_headers,
                heartbeat=30,
                timeout=aiohttp.ClientTimeout(total=None, connect=10),
            ) as upstream_ws:
                await websocket.accept(subprotocol=upstream_ws.protocol)

                t_up = asyncio.create_task(_relay_up(websocket, upstream_ws, label))
                t_down = asyncio.create_task(
                    _relay_down(websocket, upstream_ws, label)
                )

                done, pending = await asyncio.wait(
                    [t_up, t_down], return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                # Determine which side closed first and get the upstream close code.
                done_task = next(iter(done))
                if done_task is t_down:
                    _logger.info('WS proxy upstream closed first (%s)', label)
                    try:
                        close_code = done_task.result()
                    except Exception:
                        close_code = 1000
                else:
                    _logger.info('WS proxy browser closed first (%s)', label)
                    close_code = 1000

        # Explicitly close the browser WS with the upstream's code so the browser
        # receives a proper close frame (code 1000 = normal) rather than an abrupt
        # TCP drop (code 1006), which would trigger the frontend's onError handler.
        try:
            await websocket.close(code=close_code)
        except Exception:
            pass

    except (aiohttp.ClientConnectorError, aiohttp.WSServerHandshakeError) as exc:
        _logger.warning('WS proxy connect error (%s): %s', label, exc)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    except Exception as exc:
        _logger.warning('WS proxy unexpected error (%s): %s', label, exc)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
