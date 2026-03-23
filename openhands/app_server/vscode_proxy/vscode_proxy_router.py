"""HTTP and WebSocket reverse-proxy for the embedded VS Code editor.

Routes ``/vscode/{short_sandbox_id}/{path:path}`` to the OpenVSCode Server
container running on the host, selected by sandbox ID.

Making VS Code same-origin with the main OpenHands UI satisfies the
secure-context requirement for Service Workers, which in turn unlocks image
preview in the embedded editor.

Enabled only when ``proxy_vscode=True`` in DockerSandboxServiceInjector
(env var: ``SANDBOX_PROXY_VSCODE=true``).  When the sandbox service
reports no internal URL for a given sandbox the routes return 503/1013 so
the frontend can fall back to the direct-port behaviour.
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

_logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket handshake headers handled by the transport layer.
# Note: Origin is NOT in this set — VS Code validates that the WS Origin
# matches the server host, so we set it explicitly to internal_url below.
_WS_SKIP = frozenset(
    [
        'connection',
        'host',
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
        internal_url = await sandbox_service.get_vscode_internal_url(short_sandbox_id)

    if internal_url is None:
        return Response(
            content='VS Code proxy not available for this sandbox',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Forward the path as-is: VS Code is configured with OH_VSCODE_BASE_PATH
    # so it understands the /vscode/{id} prefix and generates correct redirect
    # and asset URLs.
    target = f'{internal_url}{request.url.path}'
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
        label=f'vscode:{short_sandbox_id}',
    )


# Three routes are needed because FastAPI's {path:path} converter uses .+
# (one-or-more), which refuses to match an empty string, and Starlette treats
# trailing-slash and no-trailing-slash as distinct paths.
#
# URL shapes that must be handled:
#   /vscode/{id}/           VS Code entry URL (?tkn=…, then redirect away)
#   /vscode/{id}            Post-redirect URL after token validation
#   /vscode/{id}/sub/path   Static assets, workbench, extensions, …

@router.api_route(
    '/vscode/{short_sandbox_id}/{path:path}',
    methods=_HTTP_METHODS,
    include_in_schema=False,
)
async def proxy_vscode_http(
    request: Request, short_sandbox_id: str, path: str
) -> Response:
    """Proxy an HTTP request to VS Code (non-empty sub-path)."""
    return await _do_proxy_http(request, short_sandbox_id)


@router.api_route(
    '/vscode/{short_sandbox_id}/',
    methods=_HTTP_METHODS,
    include_in_schema=False,
)
async def proxy_vscode_http_slash(
    request: Request, short_sandbox_id: str
) -> Response:
    """Proxy an HTTP request to VS Code root (trailing slash, empty path)."""
    return await _do_proxy_http(request, short_sandbox_id)


@router.api_route(
    '/vscode/{short_sandbox_id}',
    methods=_HTTP_METHODS,
    include_in_schema=False,
)
async def proxy_vscode_http_no_slash(
    request: Request, short_sandbox_id: str
) -> Response:
    """Proxy an HTTP request to VS Code root (no trailing slash).

    VS Code redirects here after token validation — the browser drops
    the ``?tkn=`` parameter and lands on this URL.
    """
    return await _do_proxy_http(request, short_sandbox_id)


# ---------------------------------------------------------------------------
# WebSocket proxy
# ---------------------------------------------------------------------------


async def _do_proxy_ws(websocket: WebSocket, short_sandbox_id: str) -> None:
    async with get_sandbox_service(websocket.state) as sandbox_service:
        internal_url = await sandbox_service.get_vscode_internal_url(short_sandbox_id)

    if internal_url is None:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    ws_scheme = 'wss' if internal_url.startswith('https') else 'ws'
    host_part = internal_url.split('://', 1)[-1]
    target_ws = f'{ws_scheme}://{host_part}{websocket.url.path}'
    if qs := str(websocket.url.query):
        target_ws = f'{target_ws}?{qs}'

    forward_headers = {
        k: v
        for k, v in websocket.headers.items()
        if k.lower() not in _WS_SKIP
    }
    # VS Code validates that the WS Origin matches the server host; set it to
    # the internal URL so the same-origin check passes from our proxy hop.
    forward_headers['origin'] = internal_url

    await do_ws_proxy(
        websocket, target_ws, forward_headers, label=f'vscode:{short_sandbox_id}'
    )


@router.websocket('/vscode/{short_sandbox_id}/{path:path}')
async def proxy_vscode_ws(
    websocket: WebSocket, short_sandbox_id: str, path: str
) -> None:
    """Proxy a WebSocket connection to VS Code (non-empty sub-path)."""
    await _do_proxy_ws(websocket, short_sandbox_id)


@router.websocket('/vscode/{short_sandbox_id}/')
async def proxy_vscode_ws_slash(
    websocket: WebSocket, short_sandbox_id: str
) -> None:
    """Proxy a WebSocket connection to VS Code root (trailing slash)."""
    await _do_proxy_ws(websocket, short_sandbox_id)


@router.websocket('/vscode/{short_sandbox_id}')
async def proxy_vscode_ws_no_slash(
    websocket: WebSocket, short_sandbox_id: str
) -> None:
    """Proxy a WebSocket connection to VS Code root (no trailing slash)."""
    await _do_proxy_ws(websocket, short_sandbox_id)
