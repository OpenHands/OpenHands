"""Conversation branch and reset endpoints.

These endpoints live at /api/conversations/ (not /api/v1) to match the frontend API calls.
They support V1 conversations only (V0 has been removed upstream).
"""

import logging
import uuid

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
)
from openhands.app_server.config import (
    depends_app_conversation_info_service,
    depends_app_conversation_service,
    depends_httpx_client,
    depends_sandbox_service,
    depends_sandbox_spec_service,
    depends_user_context,
)
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.logger import openhands_logger as conversation_logger

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/api/conversations', tags=['Conversations'], dependencies=get_dependencies()
)


async def _proxy_to_agent_server(
    request: Request,
    httpx_client: httpx.AsyncClient,
    agent_server_url: str,
    session_api_key: str | None,
    path: str,
    timeout: float | None = None,
) -> Response:
    """Forward an HTTP request to the agent-server inside the sandbox.

    Args:
        timeout: Optional timeout in seconds. If provided, a temporary client
                 is created with this timeout to avoid httpx version
                 incompatibility (send() dropped timeout kwarg in 0.28+).
    """
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ('host', 'connection', 'transfer-encoding')
    }
    if session_api_key:
        headers['X-Session-API-Key'] = session_api_key

    url = f'{agent_server_url}{path}'
    conversation_logger.info('Proxying to agent-server: %s', url)

    # Read body once so it can be reused regardless of which client sends it.
    body = await request.body()

    if timeout is not None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            proxy_request = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                content=body,
            )
            resp = await client.send(proxy_request, stream=False)
    else:
        proxy_request = httpx_client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            params=request.query_params,
            content=body,
        )
        resp = await httpx_client.send(proxy_request, stream=False)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


async def _resolve_agent_server_context(
    conversation_id: str,
    app_conversation_service: AppConversationService,
    sandbox_service: SandboxService,
    sandbox_spec_service: SandboxSpecService,
):
    """Resolve agent-server URL and session key, raising on failure."""
    ctx = await _get_agent_server_context_fn()(
        uuid.UUID(conversation_id),
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    if isinstance(ctx, JSONResponse):
        raise HTTPException(
            status_code=ctx.status_code,
            detail=ctx.body.decode() if ctx.body else None,
        )
    if ctx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Conversation {conversation_id} not running',
        )
    return ctx.agent_server_url, ctx.session_api_key


app_conversation_service_dependency = depends_app_conversation_service()
app_conversation_info_service_dependency = depends_app_conversation_info_service()
httpx_client_dependency = depends_httpx_client()
sandbox_service_dependency = depends_sandbox_service()
sandbox_spec_service_dependency = depends_sandbox_spec_service()
user_context_dependency = depends_user_context()

# Lazy import to avoid circular dependency
_agent_server_context_fn = None


def _get_agent_server_context_fn():
    global _agent_server_context_fn
    if _agent_server_context_fn is None:
        from openhands.app_server.app_conversation.app_conversation_router import (
            _get_agent_server_context,
        )

        _agent_server_context_fn = _get_agent_server_context
    return _agent_server_context_fn


@router.post('/{conversation_id}/pause')
async def pause_conversation(
    request: Request,
    conversation_id: str,
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
    sandbox_service: SandboxService = sandbox_service_dependency,
    sandbox_spec_service: SandboxSpecService = sandbox_spec_service_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
) -> Response:
    """Pause a conversation by proxying to the agent-server inside the sandbox."""
    agent_server_url, session_api_key = await _resolve_agent_server_context(
        conversation_id,
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    try:
        return await _proxy_to_agent_server(
            request,
            httpx_client,
            agent_server_url,
            session_api_key,
            f'/api/conversations/{conversation_id}/pause',
        )
    except Exception as exc:
        conversation_logger.error('Pause proxy failed for %s: %s', conversation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Failed to reach agent-server',
        )


@router.post('/{conversation_id}/run')
async def resume_conversation(
    request: Request,
    conversation_id: str,
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
    sandbox_service: SandboxService = sandbox_service_dependency,
    sandbox_spec_service: SandboxSpecService = sandbox_spec_service_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
) -> Response:
    """Resume a paused conversation by proxying to the agent-server inside the sandbox."""
    agent_server_url, session_api_key = await _resolve_agent_server_context(
        conversation_id,
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    try:
        return await _proxy_to_agent_server(
            request,
            httpx_client,
            agent_server_url,
            session_api_key,
            f'/api/conversations/{conversation_id}/run',
        )
    except Exception as exc:
        conversation_logger.error('Run proxy failed for %s: %s', conversation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Failed to reach agent-server',
        )


@router.get('/{conversation_id}/events/count')
async def events_count(
    request: Request,
    conversation_id: str,
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
    sandbox_service: SandboxService = sandbox_service_dependency,
    sandbox_spec_service: SandboxSpecService = sandbox_spec_service_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
) -> Response:
    """Get event count by proxying to the agent-server inside the sandbox."""
    agent_server_url, session_api_key = await _resolve_agent_server_context(
        conversation_id,
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    try:
        return await _proxy_to_agent_server(
            request,
            httpx_client,
            agent_server_url,
            session_api_key,
            f'/api/conversations/{conversation_id}/events/count',
        )
    except Exception as exc:
        conversation_logger.error(
            'Events count proxy failed for %s: %s', conversation_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Failed to reach agent-server',
        )


@router.post('/{conversation_id}/ask_agent')
async def ask_agent(
    request: Request,
    conversation_id: str,
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
    sandbox_service: SandboxService = sandbox_service_dependency,
    sandbox_spec_service: SandboxSpecService = sandbox_spec_service_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
) -> Response:
    """Ask the agent a side question by proxying to the agent-server inside the sandbox.

    This endpoint can take up to 10 minutes to respond as it involves an
    LLM call on the agent-server side.
    """
    agent_server_url, session_api_key = await _resolve_agent_server_context(
        conversation_id,
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    try:
        return await _proxy_to_agent_server(
            request,
            httpx_client,
            agent_server_url,
            session_api_key,
            f'/api/conversations/{conversation_id}/ask_agent',
            timeout=600,
        )
    except Exception as exc:
        conversation_logger.error(
            'Ask agent proxy failed for %s: %s', conversation_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Failed to reach agent-server',
        )
