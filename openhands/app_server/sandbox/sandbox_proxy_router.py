"""Reverse-proxy router that forwards browser traffic to sandbox agent-servers.

When OpenHands runs behind a reverse proxy (Traefik / nginx / Caddy), the browser
cannot reach the dynamically-assigned host ports that Docker maps to each
agent-server container -- those ports are not in DNS, not routed by the proxy,
and change on every conversation (see issue #14905).

This router lets the *app server* proxy all agent-server HTTP and WebSocket
traffic under the main app domain using stable, path-based URLs of the form::

    /runtime/{sandbox_id}/...

so no extra ports ever need to be exposed to the browser. It is only mounted /
used when ``OH_ENABLE_RUNTIME_PROXY`` is enabled and a public ``web_url`` is
configured; in that case the conversation URLs handed to the frontend already
point at these ``/runtime/{sandbox_id}`` paths.

Security: the ``{sandbox_id}`` is resolved through the sandbox service, so the
proxy can only ever reach the agent server of an existing sandbox -- never an
arbitrary host/port (no SSRF). The agent server itself still authenticates every
request via the ``X-Session-API-Key`` header / ``session_api_key`` query param
that the browser forwards.
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlsplit

import aiohttp
import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from openhands.app_server.config import get_sandbox_service
from openhands.app_server.sandbox.sandbox_service import SandboxService

_logger = logging.getLogger(__name__)

router = APIRouter(tags=['Runtime Proxy'])

# Path prefix used to address a sandbox agent-server through the app server.
RUNTIME_PROXY_PREFIX = 'runtime'

# Hop-by-hop headers that must not be forwarded when proxying (RFC 7230 6.1).
_HOP_BY_HOP_HEADERS = frozenset(
    {
        'connection',
        'keep-alive',
        'proxy-authenticate',
        'proxy-authorization',
        'te',
        'trailer',
        'transfer-encoding',
        'upgrade',
    }
)

# Resolved ``sandbox_id -> agent_server_url`` entries are cached briefly so the
# proxy does not hit the sandbox backend (Docker daemon) on every request while a
# conversation is active. A running sandbox keeps the same host port, so a short
# TTL is safe and stale entries simply fail the next request and get refreshed.
_RESOLUTION_CACHE_TTL_SECONDS = 30.0
_resolution_cache: dict[str, tuple[str, float]] = {}

# Shared client for proxied HTTP requests. A single pooled client minimises TLS /
# TCP handshakes. Read timeout is disabled so long-poll / streaming endpoints are
# not cut off; the connect timeout still guards against a dead agent server.
_proxy_http_client: httpx.AsyncClient | None = None
_PROXY_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0)


def _get_proxy_http_client() -> httpx.AsyncClient:
    global _proxy_http_client
    if _proxy_http_client is None or _proxy_http_client.is_closed:
        _proxy_http_client = httpx.AsyncClient(
            timeout=_PROXY_HTTP_TIMEOUT, follow_redirects=False
        )
    return _proxy_http_client


async def _resolve_agent_server_url(
    sandbox_service: SandboxService, sandbox_id: str
) -> str | None:
    """Resolve (and briefly cache) the agent-server base URL for a sandbox."""
    now = time.monotonic()
    cached = _resolution_cache.get(sandbox_id)
    if cached is not None and cached[1] > now:
        return cached[0]

    url = await sandbox_service.get_agent_server_url_by_id(sandbox_id)
    if url is None:
        # Drop any stale entry so a paused/deleted sandbox is not served from cache.
        _resolution_cache.pop(sandbox_id, None)
        return None

    _resolution_cache[sandbox_id] = (url, now + _RESOLUTION_CACHE_TTL_SECONDS)
    return url


def _forwardable_request_headers(request: Request) -> dict[str, str]:
    """Copy incoming headers, dropping ones the proxy must own/recompute."""
    headers = {}
    for key, value in request.headers.items():
        lower = key.lower()
        # ``host`` is set from the target URL by the client; ``content-length`` is
        # recomputed from the forwarded body. Hop-by-hop headers are dropped.
        if lower in _HOP_BY_HOP_HEADERS or lower in ('host', 'content-length'):
            continue
        headers[key] = value
    return headers


def _forwardable_response_headers(upstream: httpx.Response) -> dict[str, str]:
    """Copy upstream response headers, dropping hop-by-hop ones."""
    headers = {}
    for key, value in upstream.headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        headers[key] = value
    return headers


def _request_has_body(request: Request) -> bool:
    """Whether the incoming request carries a body worth forwarding."""
    if 'transfer-encoding' in request.headers:
        return True
    content_length = request.headers.get('content-length')
    return content_length is not None and content_length != '0'


@router.api_route(
    '/' + RUNTIME_PROXY_PREFIX + '/{sandbox_id}/{path:path}',
    methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'HEAD'],
    include_in_schema=False,
)
async def proxy_http(
    sandbox_id: str,
    path: str,
    request: Request,
) -> StreamingResponse:
    """Proxy an HTTP request to the agent server of ``sandbox_id``."""
    async with get_sandbox_service(request.state, request) as sandbox_service:
        base_url = await _resolve_agent_server_url(sandbox_service, sandbox_id)
    if base_url is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Sandbox not found or not running',
        )

    # Forward the raw query string verbatim so repeated keys (e.g. ``?ids=a&ids=b``)
    # are preserved -- collapsing them through a dict would drop duplicates.
    target_url = f'{base_url.rstrip("/")}/{path}'
    if request.url.query:
        target_url = f'{target_url}?{request.url.query}'

    client = _get_proxy_http_client()
    upstream_request = client.build_request(
        method=request.method,
        url=target_url,
        headers=_forwardable_request_headers(request),
        content=request.stream() if _request_has_body(request) else None,
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        _logger.warning(
            'Runtime proxy failed to reach agent server for sandbox %s: %s',
            sandbox_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Failed to reach sandbox agent server',
        )

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=_forwardable_response_headers(upstream),
        background=BackgroundTask(upstream.aclose),
    )


def _build_upstream_ws_url(base_url: str, path: str, query: str) -> str:
    """Build the agent-server WebSocket URL from the proxied HTTP base URL."""
    parts = urlsplit(base_url)
    scheme = 'wss' if parts.scheme == 'https' else 'ws'
    url = f'{scheme}://{parts.netloc}/{path}'
    if query:
        url = f'{url}?{query}'
    return url


async def _pump_client_to_upstream(
    websocket: WebSocket, upstream: aiohttp.ClientWebSocketResponse
) -> None:
    while True:
        message = await websocket.receive()
        message_type = message.get('type')
        if message_type == 'websocket.disconnect':
            await upstream.close()
            return
        text = message.get('text')
        if text is not None:
            await upstream.send_str(text)
            continue
        data = message.get('bytes')
        if data is not None:
            await upstream.send_bytes(data)


async def _pump_upstream_to_client(
    websocket: WebSocket, upstream: aiohttp.ClientWebSocketResponse
) -> None:
    async for message in upstream:
        if message.type == aiohttp.WSMsgType.TEXT:
            await websocket.send_text(message.data)
        elif message.type == aiohttp.WSMsgType.BINARY:
            await websocket.send_bytes(message.data)
        elif message.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        ):
            return


@router.websocket('/' + RUNTIME_PROXY_PREFIX + '/{sandbox_id}/{path:path}')
async def proxy_websocket(
    websocket: WebSocket,
    sandbox_id: str,
    path: str,
) -> None:
    """Proxy a WebSocket connection to the agent server of ``sandbox_id``."""
    async with get_sandbox_service(websocket.state) as sandbox_service:
        base_url = await _resolve_agent_server_url(sandbox_service, sandbox_id)
    if base_url is None:
        # 1011 = internal error / unable to fulfil the request.
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    target_url = _build_upstream_ws_url(base_url, path, websocket.url.query)
    await websocket.accept()

    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(target_url, max_msg_size=0) as upstream:
            client_task = asyncio.create_task(
                _pump_client_to_upstream(websocket, upstream)
            )
            upstream_task = asyncio.create_task(
                _pump_upstream_to_client(websocket, upstream)
            )
            _, pending = await asyncio.wait(
                {client_task, upstream_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except Exception as exc:
        _logger.warning(
            'Runtime proxy WebSocket error for sandbox %s: %s', sandbox_id, exc
        )
    finally:
        await session.close()
        try:
            await websocket.close()
        except RuntimeError:
            # Socket already closed by the framework.
            pass
