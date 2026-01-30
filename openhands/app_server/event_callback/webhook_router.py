"""Event Callback router for OpenHands App Server."""

import asyncio
import importlib
import logging
import pkgutil
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import APIKeyHeader
from jwt import InvalidTokenError
from pydantic import SecretStr

from openhands import tools  # type: ignore[attr-defined]
from openhands.agent_server.models import ConversationInfo, Success
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.config import (
    depends_app_conversation_info_service,
    depends_event_service,
    depends_jwt_service,
    depends_sandbox_service,
    get_event_callback_service,
)
from openhands.app_server.errors import AuthError
from openhands.app_server.event.event_service import EventService
from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.app_server.user.specifiy_user_context import (
    USER_CONTEXT_ATTR,
    SpecifyUserContext,
    as_admin,
)
from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderType
from openhands.sdk import ConversationExecutionStatus, Event
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.server.user_auth.default_user_auth import DefaultUserAuth
from openhands.server.user_auth.user_auth import (
    get_for_user as get_user_auth_for_user,
)

router = APIRouter(prefix='/webhooks', tags=['Webhooks'])
sandbox_service_dependency = depends_sandbox_service()
event_service_dependency = depends_event_service()
app_conversation_info_service_dependency = depends_app_conversation_info_service()
jwt_dependency = depends_jwt_service()
_logger = logging.getLogger(__name__)


async def valid_sandbox(
    user_context: UserContext = Depends(as_admin),
    session_api_key: str = Depends(
        APIKeyHeader(name='X-Session-API-Key', auto_error=False)
    ),
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> SandboxInfo:
    if session_api_key is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail='X-Session-API-Key header is required'
        )

    sandbox_info = await sandbox_service.get_sandbox_by_session_api_key(session_api_key)
    if sandbox_info is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail='Invalid session API key'
        )
    return sandbox_info


async def valid_conversation(
    conversation_id: UUID,
    sandbox_info: SandboxInfo,
    app_conversation_info_service: AppConversationInfoService = app_conversation_info_service_dependency,
) -> AppConversationInfo:
    app_conversation_info = (
        await app_conversation_info_service.get_app_conversation_info(conversation_id)
    )
    if not app_conversation_info:
        # Conversation does not yet exist - create a stub
        return AppConversationInfo(
            id=conversation_id,
            sandbox_id=sandbox_info.id,
            created_by_user_id=sandbox_info.created_by_user_id,
        )
    if app_conversation_info.created_by_user_id != sandbox_info.created_by_user_id:
        # Make sure that the conversation and sandbox were created by the same user
        raise AuthError()
    return app_conversation_info


@router.post('/conversations')
async def on_conversation_update(
    conversation_info: ConversationInfo,
    sandbox_info: SandboxInfo = Depends(valid_sandbox),
    app_conversation_info_service: AppConversationInfoService = app_conversation_info_service_dependency,
) -> Success:
    """Webhook callback for when a conversation starts, pauses, resumes, or deletes."""
    existing = await valid_conversation(
        conversation_info.id, sandbox_info, app_conversation_info_service
    )

    # If the conversation is being deleted, no action is required...
    # Later we may consider deleting the conversation if it exists...
    if conversation_info.execution_status == ConversationExecutionStatus.DELETING:
        return Success()

    app_conversation_info = AppConversationInfo(
        id=conversation_info.id,
        title=existing.title or f'Conversation {conversation_info.id.hex}',
        sandbox_id=sandbox_info.id,
        created_by_user_id=sandbox_info.created_by_user_id,
        llm_model=conversation_info.agent.llm.model,
        # Git parameters
        selected_repository=existing.selected_repository,
        selected_branch=existing.selected_branch,
        git_provider=existing.git_provider,
        trigger=existing.trigger,
        pr_number=existing.pr_number,
        # Preserve parent/child relationship and other metadata
        parent_conversation_id=existing.parent_conversation_id,
    )
    await app_conversation_info_service.save_app_conversation_info(
        app_conversation_info
    )

    return Success()


@router.post('/events/{conversation_id}')
async def on_event(
    events: list[Event],
    conversation_id: UUID,
    sandbox_info: SandboxInfo = Depends(valid_sandbox),
    app_conversation_info_service: AppConversationInfoService = app_conversation_info_service_dependency,
    event_service: EventService = event_service_dependency,
) -> Success:
    """Webhook callback for when event stream events occur."""

    app_conversation_info = await valid_conversation(
        conversation_id, sandbox_info, app_conversation_info_service
    )

    try:
        # Save events...
        await asyncio.gather(
            *[event_service.save_event(conversation_id, event) for event in events]
        )

        # Process stats events for V1 conversations
        for event in events:
            if isinstance(event, ConversationStateUpdateEvent) and event.key == 'stats':
                await app_conversation_info_service.process_stats_event(
                    event, conversation_id
                )

        asyncio.create_task(
            _run_callbacks_in_bg_and_close(
                conversation_id, app_conversation_info.created_by_user_id, events
            )
        )

    except Exception:
        _logger.exception('Error in webhook', stack_info=True)

    return Success()


@router.get('/secrets')
async def get_secret(
    access_token: str = Depends(APIKeyHeader(name='X-Access-Token', auto_error=False)),
    jwt_service: JwtService = jwt_dependency,
) -> Response:
    """Given an access token, retrieve a user secret. The access token
    is limited by user and provider type, and may include a timeout, limiting
    the damage in the event that a token is ever leaked"""
    try:
        payload = jwt_service.verify_jws_token(access_token)
        user_id = payload['user_id']
        provider_type = ProviderType(payload['provider_type'])

        # Get UserAuth for the user_id
        if user_id:
            user_auth = await get_user_auth_for_user(user_id)
        else:
            # OpenHands (OSS mode) - use default user auth
            user_auth = DefaultUserAuth()

        # Create UserContext directly
        user_context = AuthUserContext(user_auth=user_auth)

        secret = await user_context.get_latest_token(provider_type)
        if secret is None:
            raise HTTPException(404, 'No such provider')
        if isinstance(secret, SecretStr):
            secret_value = secret.get_secret_value()
        else:
            secret_value = secret

        return Response(content=secret_value, media_type='text/plain')
    except InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)


async def _run_callbacks_in_bg_and_close(
    conversation_id: UUID,
    user_id: str | None,
    events: list[Event],
):
    """Run all callbacks and close the session"""
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, SpecifyUserContext(user_id=user_id))

    async with get_event_callback_service(state) as event_callback_service:
        # We don't use asynio.gather here because callbacks must be run in sequence.
        for event in events:
            await event_callback_service.execute_callbacks(conversation_id, event)


def _import_all_tools():
    """We need to import all tools so that they are available for deserialization in webhooks."""
    for _, name, is_pkg in pkgutil.walk_packages(tools.__path__, tools.__name__ + '.'):
        if is_pkg:  # Check if it's a subpackage
            try:
                importlib.import_module(name)
            except ImportError as e:
                _logger.error(f"Warning: Could not import subpackage '{name}': {e}")


_import_all_tools()
