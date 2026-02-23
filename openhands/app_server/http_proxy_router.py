"""HTTP reverse proxy that forwards browser requests to sandbox agent-server containers.

When running in Docker network mode (OH_SANDBOX__NETWORK is set), sandbox containers
are not directly reachable from the browser. This proxy accepts HTTP requests on the
app server and forwards them to the appropriate sandbox container using
container-to-container networking.

This complements ws_proxy_router.py which handles WebSocket connections.
"""

import logging
from uuid import UUID

import httpx
from fastapi import APIRouter, Request, Response

from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.sandbox.sandbox_models import AGENT_SERVER, SandboxStatus
from openhands.app_server.sandbox.sandbox_service import SandboxService

_logger = logging.getLogger(__name__)

router = APIRouter()


@router.api_route(
    '/api/conversations/{conversation_id}/{path:path}',
    methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
)
async def http_proxy(request: Request, conversation_id: str, path: str):
    """Proxy HTTP requests from the browser to sandbox agent-server containers.

    1. Look up conversation_id -> sandbox_id -> SandboxInfo -> AGENT_SERVER internal_url
    2. Forward the request to http://{container}:{port}/api/conversations/{id}/{path}
    3. Return the upstream response to the browser

    This route is registered AFTER the V0 app-server routes, so specific V0 endpoints
    (e.g., /events, /submit-feedback) are matched first. This catch-all only handles
    paths that don't match any existing app-server route — i.e., sandbox-specific
    endpoints like /events/count, /events/search, /pause, /run.
    """
    from openhands.app_server.config import (
        get_app_conversation_info_service,
        get_sandbox_service,
    )

    state = request.state
    try:
        async with (
            get_app_conversation_info_service(state, request) as info_service,
            get_sandbox_service(state, request) as sandbox_service,
        ):
            upstream_base_url = await _resolve_upstream_base_url(
                info_service, sandbox_service, conversation_id
            )
    except Exception as exc:
        _logger.error(
            f'HTTP proxy: failed to resolve upstream for {conversation_id}: {exc}'
        )
        return Response(
            content=f'Failed to resolve sandbox: {exc}',
            status_code=502,
        )

    if not upstream_base_url:
        return Response(
            content='Could not resolve sandbox URL',
            status_code=502,
        )

    # Build upstream URL preserving the full path and query string.
    # Agent-server has two kinds of routes:
    #   - Conversation-scoped: /api/conversations/{id}/events, /pause, /run, etc.
    #   - Top-level: /api/git/*, /api/file/*, /api/vscode/*, etc.
    # For top-level routes, strip the conversation prefix when forwarding.
    _TOP_LEVEL_PREFIXES = (
        'git/',
        'file/',
        'vscode/',
        'desktop/',
        'tools/',
        'bash/',
        'skills',
    )
    if path.startswith(_TOP_LEVEL_PREFIXES):
        upstream_url = f'{upstream_base_url}/api/{path}'
    else:
        upstream_url = f'{upstream_base_url}/api/conversations/{conversation_id}/{path}'
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
        _logger.error(f'HTTP proxy: upstream request failed: {exc}')
        return Response(content='Upstream request failed', status_code=502)

    # Forward upstream response headers, excluding hop-by-hop headers
    response_headers = {}
    for key, value in upstream_response.headers.items():
        if key.lower() not in (
            'transfer-encoding',
            'content-encoding',
            'content-length',
        ):
            response_headers[key] = value

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def _resolve_upstream_base_url(
    info_service: AppConversationInfoService,
    sandbox_service: SandboxService,
    conversation_id: str,
) -> str | None:
    """Resolve the upstream HTTP base URL for a conversation's sandbox."""
    try:
        conv_uuid = UUID(hex=conversation_id)
    except ValueError:
        _logger.warning(
            f'HTTP proxy: invalid conversation_id format: {conversation_id}'
        )
        return None

    info = await info_service.get_app_conversation_info(conv_uuid)
    if info is None:
        _logger.warning(f'HTTP proxy: conversation not found: {conversation_id}')
        return None

    sandbox = await sandbox_service.get_sandbox(info.sandbox_id)
    if sandbox is None:
        _logger.warning(f'HTTP proxy: sandbox not found: {info.sandbox_id}')
        return None

    # Record activity for idle-timeout tracking
    from openhands.app_server.idle_timeout_manager import get_idle_timeout_manager

    manager = get_idle_timeout_manager()
    if manager:
        manager.touch(info.sandbox_id)

    if sandbox.status != SandboxStatus.RUNNING:
        _logger.debug(
            f'HTTP proxy: sandbox not running: {info.sandbox_id} ({sandbox.status})'
        )
        return None

    if not sandbox.exposed_urls:
        _logger.warning(f'HTTP proxy: no exposed URLs for sandbox: {info.sandbox_id}')
        return None

    agent_server_eu = next(
        (eu for eu in sandbox.exposed_urls if eu.name == AGENT_SERVER),
        None,
    )
    if not agent_server_eu or not agent_server_eu.internal_url:
        _logger.warning(f'HTTP proxy: no internal URL for sandbox: {info.sandbox_id}')
        return None

    return agent_server_eu.internal_url
