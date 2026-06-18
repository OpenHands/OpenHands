"""Conversation branch and reset endpoints.

These endpoints live at /api/conversations/ (not /api/v1) to match the frontend API calls.
They support V1 conversations only (V0 has been removed upstream).
"""

import asyncio
import logging
import os
import subprocess
import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from openhands.agent_server.models import Success
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    ConversationTrigger,
)
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
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.logger import openhands_logger as conversation_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/conversations', tags=['Conversations'], dependencies=get_dependencies())

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
    ctx = await _get_agent_server_context_fn()(
        uuid.UUID(conversation_id),
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    if isinstance(ctx, JSONResponse):
        return ctx
    if ctx is None:
        conversation_logger.warning(
            'Cannot pause conversation %s: sandbox not running',
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Conversation {conversation_id} not running',
        )

    agent_server_url = ctx.agent_server_url
    session_api_key = ctx.session_api_key

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ('host', 'connection', 'transfer-encoding')}
    if session_api_key:
        headers['X-Session-API-Key'] = session_api_key

    url = f'{agent_server_url}/api/conversations/{conversation_id}/pause'
    conversation_logger.info('Proxying pause for conversation %s to %s', conversation_id, url)

    try:
        proxy_request = httpx_client.build_request(
            method='POST',
            url=url,
            headers=headers,
        )
        resp = await httpx_client.send(proxy_request, stream=False)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception as exc:
        conversation_logger.error('Pause proxy failed for %s: %s', conversation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Failed to reach agent-server',
        )


