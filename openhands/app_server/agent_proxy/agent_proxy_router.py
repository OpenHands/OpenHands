"""HTTP and WebSocket reverse-proxy for agent-server containers.

Routes ``/agent/{short_sandbox_id}/…`` to the agent-server container running
on the host, selected by sandbox ID.

By default the frontend connects to the agent-server's dynamic host port
directly.  When a reverse proxy (Caddy, nginx, …) sits in front of OpenHands
that port is not reachable from the browser.  Routing agent-server traffic
through the OpenHands server keeps everything on one port.

Enabled only when ``proxy_agent=True`` in DockerSandboxServiceInjector
(env var: ``SANDBOX_PROXY_AGENT=true``) **and** ``WEB_HOST`` is configured
so the server knows the external URL to embed in ``agent_server_url``.

When the sandbox service reports no internal URL for a given sandbox the
routes return 503 / WS close-1013 so the frontend can surface an error.
"""

import logging

from fastapi import APIRouter, Request, WebSocket, status
from fastapi.responses import Response

from openhands.app_server._proxy_core import (
    HTTP_REQUEST_SKIP,
    do_http_proxy,
    do_ws_proxy,
)
from openhands.app_server.config import get_sandbox_service
from openhands.app_server.utils.docker_utils import replace_localhost_hostname_for_docker

_logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket handshake headers managed by the transport layer.
# Origin is also excluded: when WEB_HOST is configured the agent container's
# allow_cors_origins list is non-empty.  The LocalhostCORSMiddleware shortcut
# only fires when allow_origins is empty, so it falls through to the standard
# CORS check.  Our server-side connection has no meaningful browser origin and
# the agent server doesn't need Origin validation for an inbound proxy hop —
# omitting it causes Starlette CORS to pass the request through.
_WS_SKIP = frozenset(
    [
        'connection',
        'host',
        'origin',
        'sec-websocket-extensions',
        'sec-websocket-key',
        'sec-websocket-version',
        'upgrade',
    ]
)

_HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']


# ---------------------------------------------------------------------------
# HTTP proxy
# ---------------------------------------------------------------------------


async def _do_proxy_http(request: Request, short_sandbox_id: str) -> Response:
    async with get_sandbox_service(request.state) as sandbox_service:
        internal_url = await sandbox_service.get_agent_server_internal_url(
            short_sandbox_id
        )

    if internal_url is None:
        return Response(
            content='Agent server proxy not available for this sandbox',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # When running in Docker, localhost refers to the container itself.
    # Replace with host.docker.internal to reach the host.
    internal_url = replace_localhost_hostname_for_docker(internal_url)

    # Strip the /agent/{id} prefix — the agent server serves at its root and
    # has no knowledge of the proxy prefix (unlike VS Code which is configured
    # with OH_VSCODE_BASE_PATH).
    prefix = f'/agent/{short_sandbox_id}'
    forwarded_path = request.url.path[len(prefix):] or '/'
    target = f'{internal_url}{forwarded_path}'
    if qs := str(request.url.query):
        target = f'{target}?{qs}'

    req_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HTTP_REQUEST_SKIP
    }
    req_headers['host'] = internal_url.split('://', 1)[-1]

    return await do_http_proxy(
        request.method,
        target,
        req_headers,
        request.stream(),
        label=f'agent:{short_sandbox_id}',
    )


# Three routes are needed because FastAPI's {path:path} converter uses .+
# (one-or-more), which refuses to match an empty string, and Starlette treats
# trailing-slash and no-trailing-slash as distinct paths.

@router.api_route(
    '/agent/{short_sandbox_id}/{path:path}',
    methods=_HTTP_METHODS,
    include_in_schema=False,
)
async def proxy_agent_http(
    request: Request, short_sandbox_id: str, path: str
) -> Response:
    """Proxy an HTTP request to the agent server (non-empty sub-path)."""
    return await _do_proxy_http(request, short_sandbox_id)


@router.api_route(
    '/agent/{short_sandbox_id}/',
    methods=_HTTP_METHODS,
    include_in_schema=False,
)
async def proxy_agent_http_slash(
    request: Request, short_sandbox_id: str
) -> Response:
    """Proxy an HTTP request to the agent server root (trailing slash)."""
    return await _do_proxy_http(request, short_sandbox_id)


@router.api_route(
    '/agent/{short_sandbox_id}',
    methods=_HTTP_METHODS,
    include_in_schema=False,
)
async def proxy_agent_http_no_slash(
    request: Request, short_sandbox_id: str
) -> Response:
    """Proxy an HTTP request to the agent server root (no trailing slash)."""
    return await _do_proxy_http(request, short_sandbox_id)


# ---------------------------------------------------------------------------
# WebSocket proxy
# ---------------------------------------------------------------------------


async def _do_proxy_ws(websocket: WebSocket, short_sandbox_id: str) -> None:
    async with get_sandbox_service(websocket.state) as sandbox_service:
        internal_url = await sandbox_service.get_agent_server_internal_url(
            short_sandbox_id
        )

    if internal_url is None:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    # When running in Docker, localhost refers to the container itself.
    # Replace with host.docker.internal to reach the host.
    internal_url = replace_localhost_hostname_for_docker(internal_url)

    ws_scheme = 'wss' if internal_url.startswith('https') else 'ws'
    host_part = internal_url.split('://', 1)[-1]

    prefix = f'/agent/{short_sandbox_id}'
    forwarded_path = websocket.url.path[len(prefix):] or '/'
    target_ws = f'{ws_scheme}://{host_part}{forwarded_path}'
    if qs := str(websocket.url.query):
        target_ws = f'{target_ws}?{qs}'

    forward_headers = {
        k: v
        for k, v in websocket.headers.items()
        if k.lower() not in _WS_SKIP
    }

    await do_ws_proxy(
        websocket, target_ws, forward_headers, label=f'agent:{short_sandbox_id}'
    )


@router.websocket('/agent/{short_sandbox_id}/{path:path}')
async def proxy_agent_ws(
    websocket: WebSocket, short_sandbox_id: str, path: str
) -> None:
    """Proxy a WebSocket connection to the agent server (non-empty sub-path)."""
    await _do_proxy_ws(websocket, short_sandbox_id)


@router.websocket('/agent/{short_sandbox_id}/')
async def proxy_agent_ws_slash(
    websocket: WebSocket, short_sandbox_id: str
) -> None:
    """Proxy a WebSocket connection to the agent server root (trailing slash)."""
    await _do_proxy_ws(websocket, short_sandbox_id)


@router.websocket('/agent/{short_sandbox_id}')
async def proxy_agent_ws_no_slash(
    websocket: WebSocket, short_sandbox_id: str
) -> None:
    """Proxy a WebSocket connection to the agent server root (no trailing slash)."""
    await _do_proxy_ws(websocket, short_sandbox_id)
