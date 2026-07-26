"""REST API router for pending messages."""

import logging
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import TypeAdapter, ValidationError

from openhands.agent_server.models import ImageContent, TextContent
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTaskSortOrder,
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.app_conversation_start_task_service import (
    AppConversationStartTaskService,
)
from openhands.app_server.config import (
    depends_app_conversation_start_task_service,
    depends_httpx_client,
    depends_pending_message_service,
    depends_sandbox_service,
)
from openhands.app_server.pending_messages.pending_message_models import (
    PendingMessageResponse,
)
from openhands.app_server.pending_messages.pending_message_service import (
    PendingMessageLimitExceeded,
    PendingMessageService,
    PendingMessageUnavailable,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)

logger = logging.getLogger(__name__)

# Type adapter for validating content from request
_content_type_adapter = TypeAdapter(list[TextContent | ImageContent])

# Create router with authentication dependencies
router = APIRouter(
    prefix='/conversations/{conversation_id}/pending-messages',
    tags=['Pending Messages'],
    dependencies=get_dependencies(),
)

# Create dependency at module level
pending_message_service_dependency = depends_pending_message_service()
start_task_service_dependency = depends_app_conversation_start_task_service()
sandbox_service_dependency = depends_sandbox_service()
httpx_client_dependency = depends_httpx_client()


class _SandboxNotRunning(Exception):
    pass


async def _send_to_ready_task(
    conversation_id: str,
    role: str,
    content: list[TextContent | ImageContent],
    start_task_service: AppConversationStartTaskService,
    sandbox_service: SandboxService,
    httpx_client: httpx.AsyncClient,
) -> PendingMessageResponse | None:
    if conversation_id.startswith('task-'):
        get_task = getattr(start_task_service, 'get_app_conversation_start_task', None)
        if not callable(get_task):
            return None
        try:
            task_id = UUID(conversation_id.removeprefix('task-'))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Invalid conversation task ID',
            ) from exc
        task = await get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Conversation start task not found',
            )
    else:
        search_tasks = getattr(
            start_task_service, 'search_app_conversation_start_tasks', None
        )
        if not callable(search_tasks):
            return None
        try:
            app_conversation_id = UUID(conversation_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Invalid conversation ID',
            ) from exc
        page = await search_tasks(
            conversation_id__eq=app_conversation_id,
            sort_order=AppConversationStartTaskSortOrder.CREATED_AT_DESC,
            limit=1,
        )
        if not page.items:
            return None
        task = page.items[0]
    if task.status == AppConversationStartTaskStatus.ERROR:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Conversation start failed',
        )
    if task.status != AppConversationStartTaskStatus.READY:
        return None
    if (
        task.app_conversation_id is None
        or task.sandbox_id is None
        or task.agent_server_url is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Conversation is not reachable',
        )
    sandbox = await sandbox_service.get_sandbox(task.sandbox_id)
    if sandbox is None or sandbox.status != SandboxStatus.RUNNING:
        raise _SandboxNotRunning
    try:
        response = await httpx_client.post(
            f'{replace_localhost_hostname_for_docker(task.agent_server_url)}'
            f'/api/conversations/{task.app_conversation_id}/events',
            json={
                'role': role,
                'content': [item.model_dump() for item in content],
                'run': True,
            },
            headers=(
                {'X-Session-API-Key': sandbox.session_api_key}
                if sandbox.session_api_key
                else {}
            ),
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Failed to deliver message',
        ) from exc
    return PendingMessageResponse(
        id=str(uuid4()),
        queued=False,
        position=0,
        conversation_id=str(task.app_conversation_id),
    )


async def _validate_task_access(
    conversation_id: str,
    start_task_service: AppConversationStartTaskService,
) -> None:
    if not conversation_id.startswith('task-'):
        try:
            UUID(conversation_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Invalid conversation ID',
            ) from exc
        return
    get_task = getattr(start_task_service, 'get_app_conversation_start_task', None)
    if not callable(get_task):
        return
    try:
        task_id = UUID(conversation_id.removeprefix('task-'))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid conversation task ID',
        ) from exc
    if await get_task(task_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Conversation start task not found',
        )


@router.post(
    '', response_model=PendingMessageResponse, status_code=status.HTTP_201_CREATED
)
async def queue_pending_message(
    conversation_id: str,
    request: Request,
    pending_service: PendingMessageService = pending_message_service_dependency,
    start_task_service: AppConversationStartTaskService = (
        start_task_service_dependency
    ),
    sandbox_service: SandboxService = sandbox_service_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
) -> PendingMessageResponse:
    """Queue a message for delivery when conversation becomes ready.

    This endpoint allows users to submit messages even when the conversation's
    WebSocket connection is not yet established. Messages are stored server-side
    and delivered automatically when the conversation transitions to READY status.

    Args:
        conversation_id: The conversation ID (can be task ID before conversation is ready)
        request: The FastAPI request containing message content

    Returns:
        PendingMessageResponse with the message ID and queue position

    Raises:
        HTTPException 400: If the request body is invalid
        HTTPException 429: If too many pending messages are queued (limit: 10)
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid request body',
        )

    raw_content = body.get('content')
    role = body.get('role', 'user')

    if not raw_content or not isinstance(raw_content, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='content must be a non-empty list',
        )

    # Validate and parse content into typed objects
    try:
        content = _content_type_adapter.validate_python(raw_content)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Invalid content format: {e}',
        ) from e

    await _validate_task_access(conversation_id, start_task_service)

    try:
        response = await pending_service.add_message(
            conversation_id=conversation_id,
            content=content,
            role=role,
        )
    except PendingMessageLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Too many pending messages. Maximum 10 messages per conversation.',
        ) from exc
    except PendingMessageUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Conversation is not available',
        ) from exc

    if not response.queued:
        try:
            delivered = await _send_to_ready_task(
                response.conversation_id or conversation_id,
                role,
                content,
                start_task_service,
                sandbox_service,
                httpx_client,
            )
        except _SandboxNotRunning:
            try:
                return await pending_service.add_message(
                    conversation_id=conversation_id,
                    content=content,
                    role=role,
                    queue_if_ready=True,
                )
            except PendingMessageLimitExceeded as exc:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        'Too many pending messages. Maximum 10 messages per '
                        'conversation.'
                    ),
                ) from exc
            except PendingMessageUnavailable as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail='Conversation is not available',
                ) from exc
        if delivered is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='Conversation is not reachable',
            )
        return delivered

    logger.info(
        f'Queued pending message {response.id} for conversation {conversation_id} '
        f'(position: {response.position})'
    )

    return response
