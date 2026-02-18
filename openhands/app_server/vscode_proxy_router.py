"""HTTP and WebSocket reverse proxy for VSCode (openvscode-server) in sandbox containers.

When running in Docker network mode (OH_SANDBOX__NETWORK is set), sandbox containers
are not directly reachable from the browser. This proxy routes VSCode traffic through
/vscode-proxy/{sandbox_id}/ on the app server, forwarding to the container's port 8001.

This is simpler than the conversation-based proxies (http_proxy_router, ws_proxy_router)
because the sandbox_id is directly in the URL — no conversation→sandbox lookup needed.

The container's openvscode-server runs without --server-base-path (set by the compiled
agent-server binary which doesn't support it). So the proxy:
  - Strips the /vscode-proxy/{sandbox_id}/ prefix when forwarding to the container
  - Rewrites Location headers on redirects so they go through the proxy
  - Rewrites HTML responses to set serverBasePath and prefix static asset paths
"""

import asyncio
import logging
import re

import httpx
import websockets
from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from starlette.responses import RedirectResponse
from starlette.websockets import WebSocketState

from openhands.app_server.sandbox.sandbox_models import VSCODE, SandboxStatus
from openhands.app_server.sandbox.sandbox_service import SandboxService

_logger = logging.getLogger(__name__)

router = APIRouter()


async def _resolve_vscode_internal_url(
    sandbox_service: SandboxService,
    sandbox_id: str,
) -> str | None:
    """Resolve the internal URL for openvscode-server in a sandbox container."""
    sandbox = await sandbox_service.get_sandbox(sandbox_id)
    if sandbox is None:
        _logger.warning(f'VSCode proxy: sandbox not found: {sandbox_id}')
        return None

    # Record activity for idle-timeout tracking
    from openhands.app_server.idle_timeout_manager import get_idle_timeout_manager

    manager = get_idle_timeout_manager()
    if manager:
        manager.touch(sandbox_id)

    if sandbox.status != SandboxStatus.RUNNING:
        _logger.warning(
            f'VSCode proxy: sandbox not running: {sandbox_id} ({sandbox.status})'
        )
        return None

    if not sandbox.exposed_urls:
        _logger.warning(f'VSCode proxy: no exposed URLs for sandbox: {sandbox_id}')
        return None

    vscode_eu = next(
        (eu for eu in sandbox.exposed_urls if eu.name == VSCODE),
        None,
    )
    if not vscode_eu or not vscode_eu.internal_url:
        _logger.warning(f'VSCode proxy: no internal URL for sandbox: {sandbox_id}')
        return None

    return vscode_eu.internal_url


def _rewrite_html(body: bytes, proxy_prefix: str) -> bytes:
    """Rewrite openvscode-server HTML to work behind the proxy prefix.

    The container's VSCode server has no --server-base-path, so its HTML references
    root-level paths like /stable-{hash}/.... This function rewrites the HTML so:
      - serverBasePath is set to the proxy prefix (JS uses this for all dynamic requests)
      - Static asset paths (/stable-...) are prefixed so the browser loads them via proxy
    """
    text = body.decode('utf-8', errors='replace')

    # Rewrite serverBasePath in the workbench configuration
    # From: &quot;serverBasePath&quot;:&quot;/&quot;
    # To:   &quot;serverBasePath&quot;:&quot;/vscode-proxy/{sandbox_id}/&quot;
    text = text.replace(
        '&quot;serverBasePath&quot;:&quot;/&quot;',
        f'&quot;serverBasePath&quot;:&quot;{proxy_prefix}/&quot;',
    )

    # Rewrite all absolute paths to /stable-{hash}/... so they go through the proxy.
    # This handles all contexts: HTML attributes (href=", src="), HTML-encoded
    # attributes (&quot;/stable-...), and JavaScript (URL('/stable-...).
    # The /stable-{hash} pattern is unique to openvscode-server versioned paths.
    match = re.search(r'(/stable-[0-9a-f]+)', text)
    if match:
        stable_prefix = match.group(1)
        text = text.replace(stable_prefix, f'{proxy_prefix}{stable_prefix}')

    return text.encode('utf-8')


def _rewrite_location(location: str, proxy_prefix: str) -> str:
    """Rewrite a Location header so redirects go through the proxy."""
    if location.startswith('/'):
        return f'{proxy_prefix}{location}'
    return location


@router.get('/vscode-proxy/{sandbox_id}')
async def vscode_root_redirect(request: Request, sandbox_id: str):
    """Redirect /vscode-proxy/{sandbox_id} to /vscode-proxy/{sandbox_id}/.

    openvscode-server expects a trailing slash on the base path.
    """
    query_string = str(request.url.query)
    target = f'/vscode-proxy/{sandbox_id}/'
    if query_string:
        target = f'{target}?{query_string}'
    return RedirectResponse(url=target, status_code=302)


@router.api_route(
    '/vscode-proxy/{sandbox_id}/{path:path}',
    methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
)
async def vscode_http_proxy(request: Request, sandbox_id: str, path: str):
    """Proxy HTTP requests from the browser to openvscode-server in sandbox containers.

    1. Look up sandbox_id -> SandboxInfo -> VSCODE internal_url
    2. Strip the proxy prefix and forward to http://{container}:8001/{path}
    3. Rewrite Location headers and HTML content for the proxy prefix
    4. Return the rewritten response to the browser
    """
    from openhands.app_server.config import get_sandbox_service

    state = request.state
    try:
        async with get_sandbox_service(state, request) as sandbox_service:
            internal_url = await _resolve_vscode_internal_url(
                sandbox_service, sandbox_id
            )
    except Exception as exc:
        _logger.error(
            f'VSCode proxy: failed to resolve upstream for {sandbox_id}: {exc}'
        )
        return Response(
            content=f'Failed to resolve sandbox: {exc}',
            status_code=502,
        )

    if not internal_url:
        return Response(
            content='Could not resolve VSCode URL',
            status_code=502,
        )

    # Strip the proxy prefix — the container serves VSCode at root paths
    upstream_url = f'{internal_url}/{path}'
    if request.url.query:
        upstream_url = f'{upstream_url}?{request.url.query}'

    # Forward the request
    body = await request.body()
    headers = dict(request.headers)
    # Remove hop-by-hop headers that shouldn't be forwarded
    for header in ('host', 'transfer-encoding'):
        headers.pop(header, None)

    try:
        async with httpx.AsyncClient() as client:
            upstream_response = await client.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                content=body,
                timeout=30.0,
            )
    except httpx.RequestError as exc:
        _logger.error(f'VSCode proxy: upstream request failed: {exc}')
        return Response(content='Upstream request failed', status_code=502)

    proxy_prefix = f'/vscode-proxy/{sandbox_id}'

    # Determine if this is an HTML response that needs rewriting
    content_type = upstream_response.headers.get('content-type', '')
    is_html = 'text/html' in content_type

    # Forward upstream response headers, excluding hop-by-hop headers.
    # For HTML responses, also strip Content-Security-Policy because the
    # original CSP includes SHA-256 hashes of inline scripts — our HTML
    # rewriting changes those scripts, invalidating the hashes.
    skip_headers = {'transfer-encoding', 'content-encoding', 'content-length'}
    if is_html:
        skip_headers.add('content-security-policy')

    response_headers = {}
    for key, value in upstream_response.headers.items():
        if key.lower() not in skip_headers:
            # Rewrite Location headers so redirects go through the proxy
            if key.lower() == 'location':
                value = _rewrite_location(value, proxy_prefix)
            response_headers[key] = value

    # Rewrite HTML responses to set serverBasePath and prefix asset paths
    content = upstream_response.content
    if is_html:
        content = _rewrite_html(content, proxy_prefix)

    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


@router.websocket('/vscode-proxy/{sandbox_id}/{path:path}')
async def vscode_ws_proxy(websocket: WebSocket, sandbox_id: str, path: str):
    """Proxy WebSocket connections from the browser to openvscode-server.

    openvscode-server uses WebSockets for terminal, extensions, and other features.
    """
    from openhands.app_server.config import get_sandbox_service
    from openhands.app_server.services.injector import InjectorState
    from openhands.app_server.user.auth_user_context import AuthUserContext
    from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
    from openhands.server.user_auth.user_auth import get_for_user

    await websocket.accept()

    # Forward query parameters to upstream
    query_string = websocket.scope.get('query_string', b'').decode('utf-8')

    # WebSocket handlers don't receive a Request object, so we pre-seed the
    # user context on the injector state to avoid AuthError.
    state = InjectorState()
    user_auth = await get_for_user('root')
    setattr(state, USER_CONTEXT_ATTR, AuthUserContext(user_auth=user_auth))

    try:
        async with get_sandbox_service(state) as sandbox_service:
            internal_url = await _resolve_vscode_internal_url(
                sandbox_service, sandbox_id
            )
    except Exception as exc:
        _logger.error(
            f'VSCode WS proxy: failed to resolve upstream for {sandbox_id}: {exc}'
        )
        await websocket.close(code=1011, reason=str(exc))
        return

    if not internal_url:
        await websocket.close(code=1008, reason='Could not resolve VSCode URL')
        return

    # Strip the proxy prefix — the container serves WebSockets at root paths
    ws_url = internal_url.replace('http://', 'ws://').replace('https://', 'wss://')
    ws_url = f'{ws_url}/{path}'
    if query_string:
        ws_url = f'{ws_url}?{query_string}'

    _logger.info(f'VSCode WS proxy: connecting {sandbox_id} -> {ws_url}')

    try:
        async with websockets.connect(ws_url) as upstream:
            await _bidirectional_forward(websocket, upstream)
    except websockets.exceptions.InvalidURI as exc:
        _logger.error(f'VSCode WS proxy: invalid upstream URI: {exc}')
        await _safe_close(websocket, 1011, 'Invalid upstream URI')
    except websockets.exceptions.WebSocketException as exc:
        _logger.error(f'VSCode WS proxy: upstream connection failed: {exc}')
        await _safe_close(websocket, 1011, 'Upstream connection failed')
    except WebSocketDisconnect:
        _logger.debug(f'VSCode WS proxy: browser disconnected for {sandbox_id}')
    except Exception as exc:
        _logger.exception(f'VSCode WS proxy: unexpected error for {sandbox_id}: {exc}')
        await _safe_close(websocket, 1011, 'Internal proxy error')


async def _bidirectional_forward(
    client_ws: WebSocket,
    upstream_ws: websockets.ClientConnection,
) -> None:
    """Forward messages bidirectionally between client and upstream WebSockets."""

    async def client_to_upstream():
        """Forward messages from browser client to upstream VSCode server."""
        try:
            while True:
                message = await client_ws.receive()
                if 'text' in message:
                    await upstream_ws.send(message['text'])
                elif 'bytes' in message:
                    await upstream_ws.send(message['bytes'])
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def upstream_to_client():
        """Forward messages from upstream VSCode server to browser client."""
        try:
            async for message in upstream_ws:
                if isinstance(message, str):
                    await client_ws.send_text(message)
                elif isinstance(message, bytes):
                    await client_ws.send_bytes(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass

    # Run both directions concurrently; when either finishes, cancel the other
    task_c2u = asyncio.create_task(client_to_upstream())
    task_u2c = asyncio.create_task(upstream_to_client())

    try:
        done, pending = await asyncio.wait(
            [task_c2u, task_u2c],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Await cancelled tasks to suppress warnings
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        # Clean shutdown
        await _safe_close(client_ws, 1000, 'Proxy session ended')
        try:
            await upstream_ws.close()
        except Exception:
            pass


async def _safe_close(ws: WebSocket, code: int, reason: str) -> None:
    """Close a WebSocket connection, ignoring errors if already closed."""
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=code, reason=reason)
    except Exception:
        pass
