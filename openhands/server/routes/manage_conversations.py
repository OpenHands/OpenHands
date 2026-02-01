# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import asyncio
import base64
import itertools
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import base62
import httpx
from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
)
from openhands.app_server.config import (
    depends_app_conversation_info_service,
    depends_app_conversation_service,
    depends_db_session,
    depends_httpx_client,
    depends_sandbox_service,
)
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.services.db_session_injector import set_db_session_keep_open
from openhands.app_server.services.httpx_client_injector import (
    set_httpx_client_keep_open,
)
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.mcp_config import MCPConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    ChangeAgentStateAction,
    NullAction,
)
from openhands.events.event_filter import EventFilter
from openhands.events.event_store import EventStore
from openhands.events.observation import (
    AgentStateChangedObservation,
    NullObservation,
)
from openhands.experiments.experiment_manager import ExperimentConfig
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
)
from openhands.integrations.service_types import (
    CreateMicroagent,
    ProviderType,
    SuggestedTask,
)
from openhands.runtime import get_runtime_cls
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.server.data_models.agent_loop_info import AgentLoopInfo
from openhands.server.data_models.conversation_info import ConversationInfo
from openhands.server.data_models.conversation_info_result_set import (
    ConversationInfoResultSet,
)
from openhands.server.dependencies import get_dependencies
from openhands.server.services.conversation_service import (
    create_new_conversation,
    setup_init_conversation_settings,
)
from openhands.server.shared import (
    ConversationStoreImpl,
    config,
    conversation_manager,
    file_store,
)
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.server.user_auth import (
    get_auth_type,
    get_provider_tokens,
    get_secrets,
    get_user_id,
    get_user_settings,
    get_user_settings_store,
)
from openhands.server.user_auth.user_auth import AuthType
from openhands.server.utils import get_conversation as get_conversation_metadata
from openhands.server.utils import get_conversation_store, validate_conversation_id
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.storage.data_models.conversation_status import ConversationStatus
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.data_models.settings import Settings
from openhands.storage.locations import get_experiment_config_filename
from openhands.storage.settings.settings_store import SettingsStore
from openhands.utils.async_utils import wait_all
from openhands.utils.conversation_summary import get_default_conversation_title
from openhands.utils.environment import get_effective_llm_base_url

app = APIRouter(prefix='/api', dependencies=get_dependencies())
app_conversation_service_dependency = depends_app_conversation_service()
app_conversation_info_service_dependency = depends_app_conversation_info_service()
sandbox_service_dependency = depends_sandbox_service()
db_session_dependency = depends_db_session()
httpx_client_dependency = depends_httpx_client()


def _filter_conversations_by_age(
    conversations: list[ConversationMetadata], max_age_seconds: int
) -> list:
    """Filter conversations by age, removing those older than max_age_seconds.

    Args:
        conversations: List of conversations to filter
        max_age_seconds: Maximum age in seconds for conversations to be included

    Returns:
        List of conversations that meet the age criteria
    """
    now = datetime.now(timezone.utc)
    filtered_results = []

    for conversation in conversations:
        # Skip conversations without created_at or older than max_age
        if not hasattr(conversation, 'created_at'):
            continue

        age_seconds = (
            now - conversation.created_at.replace(tzinfo=timezone.utc)
        ).total_seconds()
        if age_seconds > max_age_seconds:
            continue

        filtered_results.append(conversation)

    return filtered_results


async def _build_conversation_result_set(
    filtered_conversations: list, next_page_id: str | None
) -> ConversationInfoResultSet:
    """Build a ConversationInfoResultSet from filtered conversations.

    This function handles the common logic of getting conversation IDs, connections,
    agent loop info, and building the final result set.

    Args:
        filtered_conversations: List of filtered conversations
        next_page_id: Next page ID for pagination

    Returns:
        ConversationInfoResultSet with the processed conversations
    """
    conversation_ids = set(
        conversation.conversation_id for conversation in filtered_conversations
    )
    connection_ids_to_conversation_ids = await conversation_manager.get_connections(
        filter_to_sids=conversation_ids
    )
    agent_loop_info = await conversation_manager.get_agent_loop_info(
        filter_to_sids=conversation_ids
    )
    agent_loop_info_by_conversation_id = {
        info.conversation_id: info for info in agent_loop_info
    }

    result = ConversationInfoResultSet(
        results=await wait_all(
            _get_conversation_info(
                conversation=conversation,
                num_connections=sum(
                    1
                    for conversation_id in connection_ids_to_conversation_ids.values()
                    if conversation_id == conversation.conversation_id
                ),
                agent_loop_info=agent_loop_info_by_conversation_id.get(
                    conversation.conversation_id
                ),
            )
            for conversation in filtered_conversations
        ),
        next_page_id=next_page_id,
    )
    return result


class InitSessionRequest(BaseModel):
    repository: str | None = None
    git_provider: ProviderType | None = None
    selected_branch: str | None = None
    initial_user_msg: str | None = None
    image_urls: list[str] | None = None
    replay_json: str | None = None
    suggested_task: SuggestedTask | None = None
    create_microagent: CreateMicroagent | None = None
    conversation_instructions: str | None = None
    mcp_config: MCPConfig | None = None
    # Only nested runtimes require the ability to specify a conversation id, and it could be a security risk
    if os.getenv('ALLOW_SET_CONVERSATION_ID', '0') == '1':
        conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    model_config = ConfigDict(extra='forbid')


class ConversationResponse(BaseModel):
    status: str
    conversation_id: str
    message: str | None = None
    conversation_status: ConversationStatus | None = None


class ProvidersSetModel(BaseModel):
    providers_set: list[ProviderType] | None = None


@app.post('/conversations', deprecated=True)
async def new_conversation(
    data: InitSessionRequest,
    user_id: str = Depends(get_user_id),
    provider_tokens: PROVIDER_TOKEN_TYPE = Depends(get_provider_tokens),
    user_secrets: Secrets = Depends(get_secrets),
    auth_type: AuthType | None = Depends(get_auth_type),
) -> ConversationResponse:
    """Initialize a new session or join an existing one.

    After successful initialization, the client should connect to the WebSocket
    using the returned conversation ID.

        Use the V1 endpoint ``POST /api/v1/app-conversations`` instead, which provides
        improved conversation management with sandbox lifecycle support.
    """
    logger.info(f'initializing_new_conversation:{data}')
    repository = data.repository
    selected_branch = data.selected_branch
    initial_user_msg = data.initial_user_msg
    image_urls = data.image_urls or []
    replay_json = data.replay_json
    suggested_task = data.suggested_task
    create_microagent = data.create_microagent
    git_provider = data.git_provider
    conversation_instructions = data.conversation_instructions

    conversation_trigger = ConversationTrigger.GUI

    if suggested_task:
        initial_user_msg = suggested_task.get_prompt_for_task()
        conversation_trigger = ConversationTrigger.SUGGESTED_TASK
    elif create_microagent:
        conversation_trigger = ConversationTrigger.MICROAGENT_MANAGEMENT
        # Set repository and git_provider from create_microagent if not already set
        if not repository and create_microagent.repo:
            repository = create_microagent.repo
        if not git_provider and create_microagent.git_provider:
            git_provider = create_microagent.git_provider

    if auth_type == AuthType.BEARER:
        conversation_trigger = ConversationTrigger.REMOTE_API_KEY

    try:
        if repository:
            provider_handler = ProviderHandler(provider_tokens)
            # Check against git_provider, otherwise check all provider apis
            await provider_handler.verify_repo_provider(repository, git_provider)

        conversation_id = getattr(data, 'conversation_id', None) or uuid.uuid4().hex
        agent_loop_info = await create_new_conversation(
            user_id=user_id,
            git_provider_tokens=provider_tokens,
            custom_secrets=user_secrets.custom_secrets if user_secrets else None,
            selected_repository=repository,
            selected_branch=selected_branch,
            initial_user_msg=initial_user_msg,
            image_urls=image_urls,
            replay_json=replay_json,
            conversation_trigger=conversation_trigger,
            conversation_instructions=conversation_instructions,
            git_provider=git_provider,
            conversation_id=conversation_id,
            mcp_config=data.mcp_config,
        )

        return ConversationResponse(
            status='ok',
            conversation_id=conversation_id,
            conversation_status=agent_loop_info.status,
        )
    except MissingSettingsError as e:
        return JSONResponse(
            content={
                'status': 'error',
                'message': str(e),
                'msg_id': 'CONFIGURATION$SETTINGS_NOT_FOUND',
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    except LLMAuthenticationError as e:
        return JSONResponse(
            content={
                'status': 'error',
                'message': str(e),
                'msg_id': RuntimeStatus.ERROR_LLM_AUTHENTICATION.value,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )


@app.get('/conversations', deprecated=True)
async def search_conversations(
    page_id: str | None = None,
    limit: int = 20,
    selected_repository: str | None = None,
    conversation_trigger: ConversationTrigger | None = None,
    include_sub_conversations: Annotated[
        bool,
        Query(
            title='If True, include sub-conversations in the results. If False (default), exclude all sub-conversations.'
        ),
    ] = False,
    conversation_store: ConversationStore = Depends(get_conversation_store),
    app_conversation_service: AppConversationService = app_conversation_service_dependency,
) -> ConversationInfoResultSet:
    """Search and list conversations with pagination support.

    Use the V1 endpoint ``GET /api/v1/app-conversations/search`` instead, which provides
    enhanced filtering, sorting, and pagination capabilities.
    """
    # Parse combined page_id to extract separate page_ids for each source
    v0_page_id = None
    v1_page_id = None

    if page_id:
        try:
            # Try to parse as JSON first
            page_data = json.loads(base64.b64decode(page_id))
            v0_page_id = page_data.get('v0')
            v1_page_id = page_data.get('v1')
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat as v0 page_id for backward compatibility
            v0_page_id = page_id

    # Get results from old conversation store (V0)
    conversation_metadata_result_set = await conversation_store.search(
        v0_page_id, limit
    )

    # Get results from new app conversation service (V1)
    age_filter_date = None
    if config.conversation_max_age_seconds:
        age_filter_date = datetime.now(timezone.utc) - timedelta(
            seconds=config.conversation_max_age_seconds
        )

    app_conversation_page = await app_conversation_service.search_app_conversations(
        page_id=v1_page_id,
        limit=limit,
        # Apply age filter at the service level if possible
        created_at__gte=age_filter_date,
        include_sub_conversations=include_sub_conversations,
    )

    # Convert V1 conversations to ConversationInfo format
    v1_conversations = [
        _to_conversation_info(app_conv) for app_conv in app_conversation_page.items
    ]

    # Apply age filter to V0 conversations
    v0_filtered_results = _filter_conversations_by_age(
        conversation_metadata_result_set.results,
        config.conversation_max_age_seconds,
    )
    v0_conversation_ids = set(
        conversation.conversation_id for conversation in v0_filtered_results
    )
    await conversation_manager.get_connections(filter_to_sids=v0_conversation_ids)
    v0_agent_loop_info = await conversation_manager.get_agent_loop_info(
        filter_to_sids=v0_conversation_ids
    )
    v0_agent_loop_info_by_conversation_id = {
        info.conversation_id: info for info in v0_agent_loop_info
    }
    v0_conversations = await wait_all(
        _get_conversation_info(
            conversation=conversation,
            num_connections=sum(
                1
                for conversation_id in v0_agent_loop_info_by_conversation_id.values()
                if conversation_id == conversation.conversation_id
            ),
            agent_loop_info=v0_agent_loop_info_by_conversation_id.get(
                conversation.conversation_id
            ),
        )
        for conversation in v0_filtered_results
    )

    # Apply additional filters to both V0 and V1 results
    def apply_filters(conversations: list[ConversationInfo]) -> list[ConversationInfo]:
        filtered = []
        for conversation in conversations:
            # Apply repository filter
            if (
                selected_repository is not None
                and conversation.selected_repository != selected_repository
            ):
                continue

            # Apply conversation trigger filter
            if (
                conversation_trigger is not None
                and conversation.trigger != conversation_trigger
            ):
                continue

            filtered.append(conversation)
        return filtered

    v0_final_filtered = apply_filters(v0_conversations)
    v1_final_filtered = apply_filters(v1_conversations)

    # Combine results from both sources
    all_conversations = v0_final_filtered + v1_final_filtered

    # Sort by created_at descending (most recent first)
    all_conversations.sort(
        key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    # Limit to requested number of results
    final_results = all_conversations[:limit]

    # Create combined page_id for next page
    next_page_id = None
    if (
        conversation_metadata_result_set.next_page_id
        or app_conversation_page.next_page_id
    ):
        next_page_data = {
            'v0': conversation_metadata_result_set.next_page_id,
            'v1': app_conversation_page.next_page_id,
        }
        # Only include page_id if at least one source has more pages
        if next_page_data['v0'] or next_page_data['v1']:
            next_page_id = base64.b64encode(
                json.dumps(next_page_data).encode()
            ).decode()

    return ConversationInfoResultSet(results=final_results, next_page_id=next_page_id)


@app.get('/conversations/{conversation_id}', deprecated=True)
async def get_conversation(
    conversation_id: str = Depends(validate_conversation_id),
    conversation_store: ConversationStore = Depends(get_conversation_store),
    app_conversation_service: AppConversationService = app_conversation_service_dependency,
) -> ConversationInfo | None:
    """Get a single conversation by ID.

    Use the V1 endpoint ``GET /api/v1/app-conversations?ids={conversation_id}`` instead,
    which supports batch retrieval of conversations by their IDs.
    """
    try:
        # Shim to add V1 conversations
        try:
            conversation_uuid = uuid.UUID(conversation_id)
            app_conversation = await app_conversation_service.get_app_conversation(
                conversation_uuid
            )
            if app_conversation:
                return _to_conversation_info(app_conversation)
        except (ValueError, TypeError, Exception):
            # Not a V1 conversation or service error
            pass

        metadata = await conversation_store.get_metadata(conversation_id)
        num_connections = len(
            await conversation_manager.get_connections(filter_to_sids={conversation_id})
        )
        agent_loop_infos = await conversation_manager.get_agent_loop_info(
            filter_to_sids={conversation_id}
        )
        agent_loop_info = agent_loop_infos[0] if agent_loop_infos else None
        conversation_info = await _get_conversation_info(
            metadata, num_connections, agent_loop_info
        )
        return conversation_info
    except FileNotFoundError:
        return None


@app.delete('/conversations/{conversation_id}', deprecated=True)
async def delete_conversation(
    request: Request,
    conversation_id: str = Depends(validate_conversation_id),
    user_id: str | None = Depends(get_user_id),
    app_conversation_service: AppConversationService = app_conversation_service_dependency,
    app_conversation_info_service: AppConversationInfoService = app_conversation_info_service_dependency,
    sandbox_service: SandboxService = sandbox_service_dependency,
    db_session: AsyncSession = db_session_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
) -> bool:
    """Delete a conversation by ID.

    For V1 conversations, use ``DELETE /api/v1/sandboxes/{sandbox_id}`` to delete the
    associated sandbox, which will clean up the conversation resources.
    """
    set_db_session_keep_open(request.state, True)
    set_httpx_client_keep_open(request.state, True)

    # Try V1 conversation first
    v1_result = await _try_delete_v1_conversation(
        conversation_id,
        app_conversation_service,
        app_conversation_info_service,
        sandbox_service,
        db_session,
        httpx_client,
    )
    if v1_result is not None:
        return v1_result

    # Close connections
    await db_session.close()
    await httpx_client.aclose()

    # V0 conversation logic
    return await _delete_v0_conversation(conversation_id, user_id)


async def _try_delete_v1_conversation(
    conversation_id: str,
    app_conversation_service: AppConversationService,
    app_conversation_info_service: AppConversationInfoService,
    sandbox_service: SandboxService,
    db_session: AsyncSession,
    httpx_client: httpx.AsyncClient,
) -> bool | None:
    """Try to delete a V1 conversation. Returns None if not a V1 conversation."""
    result = None
    try:
        conversation_uuid = uuid.UUID(conversation_id)
        # Check if it's a V1 conversation by trying to get it
        app_conversation_info = (
            await app_conversation_info_service.get_app_conversation_info(
                conversation_uuid
            )
        )
        if app_conversation_info:
            # This is a V1 conversation, delete it using the app conversation service
            # Pass the conversation ID for secure deletion
            result = await app_conversation_service.delete_app_conversation(
                app_conversation_info.id
            )

            # Manually commit so that the conversation will vanish from the list
            await db_session.commit()

            # Delete the sandbox in the background
            asyncio.create_task(
                _delete_sandbox_and_close_connections(
                    sandbox_service,
                    app_conversation_info.sandbox_id,
                    db_session,
                    httpx_client,
                )
            )
    except Exception:
        # Continue with V0 logic
        pass

    return result


async def _delete_sandbox_and_close_connections(
    sandbox_service: SandboxService,
    sandbox_id: str,
    db_session: AsyncSession,
    httpx_client: httpx.AsyncClient,
):
    try:
        await sandbox_service.delete_sandbox(sandbox_id)
        await db_session.commit()
    finally:
        await asyncio.gather(
            *[
                db_session.aclose(),
                httpx_client.aclose(),
            ]
        )


async def _delete_v0_conversation(conversation_id: str, user_id: str | None) -> bool:
    """Delete a V0 conversation using the legacy logic."""
    conversation_store = await ConversationStoreImpl.get_instance(config, user_id)
    try:
        await conversation_store.get_metadata(conversation_id)
    except FileNotFoundError:
        return False

    # Stop the conversation if it's running
    is_running = await conversation_manager.is_agent_loop_running(conversation_id)
    if is_running:
        await conversation_manager.close_session(conversation_id)

    # Clean up runtime and metadata
    runtime_cls = get_runtime_cls(config.runtime)
    await runtime_cls.delete(conversation_id)
    await conversation_store.delete_metadata(conversation_id)
    return True


@app.get('/conversations/{conversation_id}/remember-prompt')
async def get_prompt(
    event_id: int,
    conversation_id: str = Depends(validate_conversation_id),
    user_settings: SettingsStore = Depends(get_user_settings_store),
    metadata: ConversationMetadata = Depends(get_conversation_metadata),
):
    # get event store for the conversation
    event_store = EventStore(
        sid=conversation_id, file_store=file_store, user_id=metadata.user_id
    )

    # retrieve the relevant events
    stringified_events = _get_contextual_events(event_store, event_id)

    # generate a prompt
    settings = await user_settings.load()
    if settings is None:
        # placeholder for error handling
        raise ValueError('Settings not found')

    settings_base_url = settings.llm_base_url
    effective_base_url = get_effective_llm_base_url(
        settings.llm_model,
        settings_base_url,
    )
    llm_config = LLMConfig(
        model=settings.llm_model or '',
        api_key=settings.llm_api_key,
        base_url=effective_base_url,
    )

    prompt_template = generate_prompt_template(stringified_events)
    prompt = await generate_prompt(llm_config, prompt_template, conversation_id)

    return JSONResponse(
        {
            'status': 'success',
            'prompt': prompt,
        }
    )


def generate_prompt_template(events: str) -> str:
    env = Environment(loader=FileSystemLoader('openhands/microagent/prompts'))
    template = env.get_template('generate_remember_prompt.j2')
    return template.render(events=events)


async def generate_prompt(
    llm_config: LLMConfig, prompt_template: str, conversation_id: str
) -> str:
    messages = [
        {
            'role': 'system',
            'content': prompt_template,
        },
        {
            'role': 'user',
            'content': 'Please generate a prompt for the AI to update the special file based on the events provided.',
        },
    ]

    raw_prompt = await conversation_manager.request_llm_completion(
        'remember_prompt', conversation_id, llm_config, messages
    )
    prompt = re.search(r'<update_prompt>(.*?)</update_prompt>', raw_prompt, re.DOTALL)

    if prompt:
        return prompt.group(1).strip()
    else:
        raise ValueError('No valid prompt found in the response.')


async def _get_conversation_info(
    conversation: ConversationMetadata,
    num_connections: int,
    agent_loop_info: AgentLoopInfo | None,
) -> ConversationInfo | None:
    try:
        title = conversation.title
        if not title:
            title = get_default_conversation_title(conversation.conversation_id)
        return ConversationInfo(
            trigger=conversation.trigger,
            conversation_id=conversation.conversation_id,
            title=title,
            last_updated_at=conversation.last_updated_at,
            created_at=conversation.created_at,
            selected_repository=conversation.selected_repository,
            selected_branch=conversation.selected_branch,
            git_provider=conversation.git_provider,
            status=getattr(agent_loop_info, 'status', ConversationStatus.STOPPED),
            runtime_status=getattr(agent_loop_info, 'runtime_status', None),
            num_connections=num_connections,
            url=agent_loop_info.url if agent_loop_info else None,
            session_api_key=getattr(agent_loop_info, 'session_api_key', None),
            pr_number=conversation.pr_number,
        )
    except Exception as e:
        logger.error(
            f'Error loading conversation {conversation.conversation_id}: {str(e)}',
            extra={'session_id': conversation.conversation_id},
        )
        return None


@app.post('/conversations/{conversation_id}/start', deprecated=True)
async def start_conversation(
    providers_set: ProvidersSetModel,
    conversation_id: str = Depends(validate_conversation_id),
    user_id: str = Depends(get_user_id),
    provider_tokens: PROVIDER_TOKEN_TYPE = Depends(get_provider_tokens),
    settings: Settings = Depends(get_user_settings),
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> ConversationResponse:
    """Start an agent loop for a conversation.

    This endpoint calls the conversation_manager's maybe_start_agent_loop method
    to start a conversation. If the conversation is already running, it will
    return the existing agent loop info.

        Use the V1 endpoint ``POST /api/v1/app-conversations`` instead, which combines
        conversation creation and starting into a single operation with integrated
        sandbox lifecycle management.
    """
    logger.info(
        f'Starting conversation: {conversation_id}',
        extra={'session_id': conversation_id},
    )

    # Log token fetch status
    if provider_tokens:
        logger.info(
            f'/start endpoint: Fetched provider tokens: {list(provider_tokens.keys())}',
            extra={'session_id': conversation_id},
        )
    else:
        logger.warning(
            '/start endpoint: No provider tokens fetched (provider_tokens is None/empty)',
            extra={'session_id': conversation_id},
        )

    try:
        # Check that the conversation exists
        try:
            await conversation_store.get_metadata(conversation_id)
        except Exception:
            return JSONResponse(
                content={
                    'status': 'error',
                    'conversation_id': conversation_id,
                },
                status_code=status.HTTP_404_NOT_FOUND,
            )

        # Set up conversation init data with provider information
        conversation_init_data = await setup_init_conversation_settings(
            user_id, conversation_id, providers_set.providers_set or [], provider_tokens
        )

        # Start the agent loop
        agent_loop_info = await conversation_manager.maybe_start_agent_loop(
            sid=conversation_id,
            settings=conversation_init_data,
            user_id=user_id,
        )

        return ConversationResponse(
            status='ok',
            conversation_id=conversation_id,
            conversation_status=agent_loop_info.status,
        )
    except Exception as e:
        logger.error(
            f'Error starting conversation {conversation_id}: {str(e)}',
            extra={'session_id': conversation_id},
        )
        return JSONResponse(
            content={
                'status': 'error',
                'conversation_id': conversation_id,
                'message': f'Failed to start conversation: {str(e)}',
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.post('/conversations/{conversation_id}/stop', deprecated=True)
async def stop_conversation(
    conversation_id: str = Depends(validate_conversation_id),
    user_id: str = Depends(get_user_id),
) -> ConversationResponse:
    """Stop an agent loop for a conversation.

    This endpoint calls the conversation_manager's close_session method
    to stop a conversation.

        Use the V1 endpoint ``POST /api/v1/sandboxes/{sandbox_id}/pause`` instead to pause
        the sandbox execution, or ``DELETE /api/v1/sandboxes/{sandbox_id}`` to fully stop
        and remove the sandbox.
    """
    logger.info(f'Stopping conversation: {conversation_id}')

    try:
        # Check if the conversation is running
        agent_loop_info = await conversation_manager.get_agent_loop_info(
            user_id=user_id, filter_to_sids={conversation_id}
        )
        conversation_status = (
            agent_loop_info[0].status if agent_loop_info else ConversationStatus.STOPPED
        )

        if conversation_status not in (
            ConversationStatus.STARTING,
            ConversationStatus.RUNNING,
        ):
            return ConversationResponse(
                status='ok',
                conversation_id=conversation_id,
                message='Conversation was not running',
                conversation_status=conversation_status,
            )

        # Stop the conversation
        await conversation_manager.close_session(conversation_id)

        return ConversationResponse(
            status='ok',
            conversation_id=conversation_id,
            message='Conversation stopped successfully',
            conversation_status=conversation_status,
        )
    except Exception as e:
        logger.error(
            f'Error stopping conversation {conversation_id}: {str(e)}',
            extra={'session_id': conversation_id},
        )
        return JSONResponse(
            content={
                'status': 'error',
                'conversation_id': conversation_id,
                'message': f'Failed to stop conversation: {str(e)}',
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _get_contextual_events(event_store: EventStore, event_id: int) -> str:
    # find the specified events to learn from
    # Get X events around the target event
    context_size = 4

    agent_event_filter = EventFilter(
        exclude_hidden=True,
        exclude_types=(
            NullAction,
            NullObservation,
            ChangeAgentStateAction,
            AgentStateChangedObservation,
        ),
    )  # the types of events that can be in an agent's history

    # from event_id - context_size to event_id..
    context_before = event_store.search_events(
        start_id=event_id,
        filter=agent_event_filter,
        reverse=True,
        limit=context_size,
    )

    # from event_id to event_id + context_size + 1
    context_after = event_store.search_events(
        start_id=event_id + 1,
        filter=agent_event_filter,
        limit=context_size + 1,
    )

    # context_before is in reverse chronological order, so convert to list and reverse it.
    ordered_context_before = list(context_before)
    ordered_context_before.reverse()

    all_events = itertools.chain(ordered_context_before, context_after)
    stringified_events = '\n'.join(str(event) for event in all_events)
    return stringified_events


class UpdateConversationRequest(BaseModel):
    """Request model for updating conversation metadata."""

    title: str = Field(
        ..., min_length=1, max_length=200, description='New conversation title'
    )

    model_config = ConfigDict(extra='forbid')


async def _update_v1_conversation(
    conversation_uuid: uuid.UUID,
    new_title: str,
    user_id: str | None,
    app_conversation_info_service: AppConversationInfoService,
    app_conversation_service: AppConversationService,
) -> JSONResponse | bool:
    """Update a V1 conversation title.

    Args:
        conversation_uuid: The conversation ID as a UUID
        new_title: The new title to set
        user_id: The authenticated user ID
        app_conversation_info_service: The app conversation info service
        app_conversation_service: The app conversation service for agent-server communication

    Returns:
        JSONResponse on error, True on success
    """
    conversation_id = str(conversation_uuid)
    logger.info(
        f'Updating V1 conversation {conversation_uuid}',
        extra={'session_id': conversation_id, 'user_id': user_id},
    )

    # Get the V1 conversation info
    app_conversation_info = (
        await app_conversation_info_service.get_app_conversation_info(conversation_uuid)
    )

    if not app_conversation_info:
        # Not a V1 conversation
        return None

    # Validate that the user owns this conversation
    if user_id and app_conversation_info.created_by_user_id != user_id:
        logger.warning(
            f'User {user_id} attempted to update V1 conversation {conversation_uuid} owned by {app_conversation_info.created_by_user_id}',
            extra={'session_id': conversation_id, 'user_id': user_id},
        )
        return JSONResponse(
            content={
                'status': 'error',
                'message': 'Permission denied: You can only update your own conversations',
                'msg_id': 'AUTHORIZATION$PERMISSION_DENIED',
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    # Update the title and timestamp
    original_title = app_conversation_info.title
    app_conversation_info.title = new_title
    app_conversation_info.updated_at = datetime.now(timezone.utc)

    # Save the updated conversation info
    try:
        await app_conversation_info_service.save_app_conversation_info(
            app_conversation_info
        )
    except AssertionError:
        # This happens when user doesn't own the conversation
        logger.warning(
            f'User {user_id} attempted to update V1 conversation {conversation_uuid} - permission denied',
            extra={'session_id': conversation_id, 'user_id': user_id},
        )
        return JSONResponse(
            content={
                'status': 'error',
                'message': 'Permission denied: You can only update your own conversations',
                'msg_id': 'AUTHORIZATION$PERMISSION_DENIED',
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    # Try to update the agent-server as well
    try:
        if hasattr(app_conversation_service, 'update_agent_server_conversation_title'):
            await app_conversation_service.update_agent_server_conversation_title(
                conversation_id=conversation_id,
                new_title=new_title,
                app_conversation_info=app_conversation_info,
            )
    except Exception as e:
        # Log the error but don't fail the database update
        logger.warning(
            f'Failed to update agent-server for conversation {conversation_uuid}: {e}',
            extra={'session_id': conversation_id, 'user_id': user_id},
        )

    logger.info(
        f'Successfully updated V1 conversation {conversation_uuid} title from "{original_title}" to "{app_conversation_info.title}"',
        extra={'session_id': conversation_id, 'user_id': user_id},
    )

    return True


async def _update_v0_conversation(
    conversation_id: str,
    new_title: str,
    user_id: str | None,
    conversation_store: ConversationStore,
) -> JSONResponse | bool:
    """Update a V0 conversation title.

    Args:
        conversation_id: The conversation ID
        new_title: The new title to set
        user_id: The authenticated user ID
        conversation_store: The conversation store

    Returns:
        JSONResponse on error, True on success

    Raises:
        FileNotFoundError: If the conversation is not found
    """
    logger.info(
        f'Updating V0 conversation {conversation_id}',
        extra={'session_id': conversation_id, 'user_id': user_id},
    )

    # Get the existing conversation metadata
    metadata = await conversation_store.get_metadata(conversation_id)

    # Validate that the user owns this conversation
    if user_id and metadata.user_id != user_id:
        logger.warning(
            f'User {user_id} attempted to update conversation {conversation_id} owned by {metadata.user_id}',
            extra={'session_id': conversation_id, 'user_id': user_id},
        )
        return JSONResponse(
            content={
                'status': 'error',
                'message': 'Permission denied: You can only update your own conversations',
                'msg_id': 'AUTHORIZATION$PERMISSION_DENIED',
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    # Update the conversation metadata
    original_title = metadata.title
    metadata.title = new_title
    metadata.last_updated_at = datetime.now(timezone.utc)

    # Save the updated metadata
    await conversation_store.save_metadata(metadata)

    # Emit a status update to connected clients about the title change
    try:
        status_update_dict = {
            'status_update': True,
            'type': 'info',
            'message': conversation_id,
            'conversation_title': metadata.title,
        }
        await conversation_manager.sio.emit(
            'oh_event',
            status_update_dict,
            to=f'room:{conversation_id}',
        )
    except Exception as e:
        logger.error(f'Error emitting title update event: {e}')
        # Don't fail the update if we can't emit the event

    logger.info(
        f'Successfully updated conversation {conversation_id} title from "{original_title}" to "{metadata.title}"',
        extra={'session_id': conversation_id, 'user_id': user_id},
    )

    return True


@app.patch('/conversations/{conversation_id}', deprecated=True)
async def update_conversation(
    data: UpdateConversationRequest,
    conversation_id: str = Depends(validate_conversation_id),
    user_id: str | None = Depends(get_user_id),
    conversation_store: ConversationStore = Depends(get_conversation_store),
    app_conversation_info_service: AppConversationInfoService = app_conversation_info_service_dependency,
    app_conversation_service: AppConversationService = app_conversation_service_dependency,
) -> bool:
    """Update conversation metadata.

    This endpoint allows updating conversation details like title.
    Only the conversation owner can update the conversation.
    Supports both V0 and V1 conversations.

    Args:
        conversation_id: The ID of the conversation to update
        data: The conversation update data (title, etc.)
        user_id: The authenticated user ID
        conversation_store: The conversation store dependency
        app_conversation_info_service: The app conversation info service for V1 conversations
        app_conversation_service: The app conversation service for agent-server communication

    Returns:
        bool: True if the conversation was updated successfully

    Raises:
        HTTPException: If conversation is not found or user lacks permission

        This endpoint is part of the legacy V0 API and will be removed in a future release.
        Use the V1 endpoint ``PATCH /api/v1/app-conversations/{conversation_id}`` instead,
        which provides the same functionality for updating conversation metadata.
    """
    logger.info(
        f'Updating conversation {conversation_id} with title: {data.title}',
        extra={'session_id': conversation_id, 'user_id': user_id},
    )

    new_title = data.title.strip()

    # Try to handle as V1 conversation first
    try:
        conversation_uuid = uuid.UUID(conversation_id)
        result = await _update_v1_conversation(
            conversation_uuid=conversation_uuid,
            new_title=new_title,
            user_id=user_id,
            app_conversation_info_service=app_conversation_info_service,
            app_conversation_service=app_conversation_service,
        )

        # If result is not None, it's a V1 conversation (either success or error)
        if result is not None:
            return result

    except (ValueError, TypeError):
        # Not a valid UUID, fall through to V0 logic
        pass
    except Exception as e:
        logger.warning(
            f'Error checking V1 conversation {conversation_id}: {str(e)}',
            extra={'session_id': conversation_id, 'user_id': user_id},
        )
        # Fall through to V0 logic

    # Handle as V0 conversation
    try:
        return await _update_v0_conversation(
            conversation_id=conversation_id,
            new_title=new_title,
            user_id=user_id,
            conversation_store=conversation_store,
        )
    except FileNotFoundError:
        logger.warning(
            f'Conversation {conversation_id} not found for update',
            extra={'session_id': conversation_id, 'user_id': user_id},
        )
        return JSONResponse(
            content={
                'status': 'error',
                'message': 'Conversation not found',
                'msg_id': 'CONVERSATION$NOT_FOUND',
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )
    except Exception as e:
        logger.error(
            f'Error updating conversation {conversation_id}: {str(e)}',
            extra={'session_id': conversation_id, 'user_id': user_id},
        )
        return JSONResponse(
            content={
                'status': 'error',
                'message': f'Failed to update conversation: {str(e)}',
                'msg_id': 'CONVERSATION$UPDATE_ERROR',
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.post('/conversations/{conversation_id}/exp-config')
def add_experiment_config_for_conversation(
    exp_config: ExperimentConfig,
    conversation_id: str = Depends(validate_conversation_id),
) -> bool:
    exp_config_filepath = get_experiment_config_filename(conversation_id)
    exists = False
    try:
        file_store.read(exp_config_filepath)
        exists = True
    except FileNotFoundError:
        pass

    # Don't modify again if it already exists
    if exists:
        return False

    try:
        file_store.write(exp_config_filepath, exp_config.model_dump_json())
    except Exception as e:
        logger.info(f'Failed to write experiment config for {conversation_id}: {e}')
        return True

    return False


def _parse_combined_page_id(page_id: str | None) -> tuple[str | None, str | None]:
    """Parse combined page_id to extract separate V0 and V1 page_ids.

    Args:
        page_id: Combined page_id (base64-encoded JSON) or legacy V0 page_id

    Returns:
        Tuple of (v0_page_id, v1_page_id)
    """
    v0_page_id = None
    v1_page_id = None

    if page_id:
        try:
            # Try to parse as JSON first
            page_data = json.loads(base64.b64decode(page_id))
            v0_page_id = page_data.get('v0')
            v1_page_id = page_data.get('v1')
        except (json.JSONDecodeError, TypeError, Exception):
            # Fallback: treat as v0 page_id for backward compatibility
            # This catches base64 decode errors and any other parsing issues
            v0_page_id = page_id

    return v0_page_id, v1_page_id


async def _fetch_v1_conversations_safe(
    app_conversation_service: AppConversationService,
    v1_page_id: str | None,
    limit: int,
) -> tuple[list[ConversationInfo], str | None]:
    """Safely fetch V1 conversations with error handling.

    Args:
        app_conversation_service: App conversation service for V1
        v1_page_id: Page ID for V1 pagination
        limit: Maximum number of results
        include_sub_conversations: If True, include sub-conversations in results

    Returns:
        Tuple of (v1_conversations, v1_next_page_id)
    """
    v1_conversations = []
    v1_next_page_id = None

    try:
        age_filter_date = None
        if config.conversation_max_age_seconds:
            age_filter_date = datetime.now(timezone.utc) - timedelta(
                seconds=config.conversation_max_age_seconds
            )

        app_conversation_page = await app_conversation_service.search_app_conversations(
            page_id=v1_page_id,
            limit=limit,
            created_at__gte=age_filter_date,
        )

        v1_conversations = [
            _to_conversation_info(app_conv) for app_conv in app_conversation_page.items
        ]
        v1_next_page_id = app_conversation_page.next_page_id
    except Exception as e:
        # V1 system might not be available or initialized yet
        logger.debug(f'V1 conversation service not available: {str(e)}')

    return v1_conversations, v1_next_page_id


async def _process_v0_conversations(
    conversation_metadata_result_set,
) -> list[ConversationInfo]:
    """Process V0 conversations with age filtering and agent loop info.

    Args:
        conversation_metadata_result_set: Result set from V0 conversation store

    Returns:
        List of processed ConversationInfo objects
    """
    # Apply age filter to V0 conversations
    v0_filtered_results = _filter_conversations_by_age(
        conversation_metadata_result_set.results,
        config.conversation_max_age_seconds,
    )

    v0_conversation_ids = set(
        conversation.conversation_id for conversation in v0_filtered_results
    )

    # Get agent loop info for V0 conversations
    await conversation_manager.get_connections(filter_to_sids=v0_conversation_ids)
    v0_agent_loop_info = await conversation_manager.get_agent_loop_info(
        filter_to_sids=v0_conversation_ids
    )
    v0_agent_loop_info_by_conversation_id = {
        info.conversation_id: info for info in v0_agent_loop_info
    }

    # Convert to ConversationInfo objects
    v0_conversations = await wait_all(
        _get_conversation_info(
            conversation=conversation,
            num_connections=sum(
                1
                for conversation_id in v0_agent_loop_info_by_conversation_id.values()
                if conversation_id == conversation.conversation_id
            ),
            agent_loop_info=v0_agent_loop_info_by_conversation_id.get(
                conversation.conversation_id
            ),
        )
        for conversation in v0_filtered_results
    )

    return v0_conversations


async def _apply_microagent_filters(
    conversations: list[ConversationInfo],
    selected_repository: str,
    provider_handler: ProviderHandler,
) -> list[ConversationInfo]:
    """Apply microagent management specific filters to conversations.

    Filters conversations by:
    - Trigger type (MICROAGENT_MANAGEMENT)
    - Repository match
    - PR status (only open PRs)

    Args:
        conversations: List of conversations to filter
        selected_repository: Repository to filter by
        provider_handler: Handler for checking PR status

    Returns:
        Filtered list of conversations
    """
    filtered = []
    for conversation in conversations:
        # Only include microagent_management conversations
        if conversation.trigger != ConversationTrigger.MICROAGENT_MANAGEMENT:
            continue

        # Apply repository filter
        if conversation.selected_repository != selected_repository:
            continue

        # Check if PR is still open
        if (
            conversation.pr_number
            and len(conversation.pr_number) > 0
            and conversation.selected_repository
            and conversation.git_provider
            and not await provider_handler.is_pr_open(
                conversation.selected_repository,
                conversation.pr_number[-1],  # Get the last PR number
                conversation.git_provider,
            )
        ):
            # Skip this conversation if the PR is closed/merged
            continue

        filtered.append(conversation)

    return filtered


def _create_combined_page_id(
    v0_next_page_id: str | None, v1_next_page_id: str | None
) -> str | None:
    """Create a combined page_id from V0 and V1 page_ids.

    Args:
        v0_next_page_id: Next page ID for V0 conversations
        v1_next_page_id: Next page ID for V1 conversations

    Returns:
        Base64-encoded JSON combining both page_ids, or None if no next pages
    """
    if not v0_next_page_id and not v1_next_page_id:
        return None

    next_page_data = {
        'v0': v0_next_page_id,
        'v1': v1_next_page_id,
    }

    return base64.b64encode(json.dumps(next_page_data).encode()).decode()


@app.get('/microagent-management/conversations')
async def get_microagent_management_conversations(
    selected_repository: str,
    page_id: str | None = None,
    limit: int = 20,
    conversation_store: ConversationStore = Depends(get_conversation_store),
    provider_tokens: PROVIDER_TOKEN_TYPE = Depends(get_provider_tokens),
    app_conversation_service: AppConversationService = app_conversation_service_dependency,
) -> ConversationInfoResultSet:
    """Get conversations for the microagent management page with pagination support.

    This endpoint returns conversations with conversation_trigger = 'microagent_management'
    and only includes conversations with active PRs. Pagination is supported.

    Args:
        page_id: Optional page ID for pagination
        limit: Maximum number of results per page (default: 20)
        selected_repository: Repository filter to limit results to a specific repository
        conversation_store: Conversation store dependency
        provider_tokens: Provider tokens for checking PR status
        app_conversation_service: App conversation service for V1 conversations

    Returns:
        ConversationInfoResultSet with filtered and paginated results
    """
    # Parse page_id to extract V0 and V1 components
    v0_page_id, v1_page_id = _parse_combined_page_id(page_id)

    # Fetch V0 conversations
    conversation_metadata_result_set = await conversation_store.search(
        v0_page_id, limit
    )

    # Fetch V1 conversations (with graceful error handling)
    v1_conversations, v1_next_page_id = await _fetch_v1_conversations_safe(
        app_conversation_service, v1_page_id, limit
    )

    # Process V0 conversations
    v0_conversations = await _process_v0_conversations(conversation_metadata_result_set)

    # Apply microagent-specific filters
    provider_handler = ProviderHandler(provider_tokens)
    v0_filtered = await _apply_microagent_filters(
        v0_conversations, selected_repository, provider_handler
    )
    v1_filtered = await _apply_microagent_filters(
        v1_conversations, selected_repository, provider_handler
    )

    # Combine and sort results
    all_conversations = v0_filtered + v1_filtered
    all_conversations.sort(
        key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    # Limit to requested number of results
    final_results = all_conversations[:limit]

    # Create combined page_id for pagination
    next_page_id = _create_combined_page_id(
        conversation_metadata_result_set.next_page_id, v1_next_page_id
    )

    return ConversationInfoResultSet(results=final_results, next_page_id=next_page_id)


def _to_conversation_info(app_conversation: AppConversation) -> ConversationInfo:
    """Convert a V1 AppConversation into an old style ConversationInfo"""
    from openhands.app_server.sandbox.sandbox_models import SandboxStatus

    # Map SandboxStatus to ConversationStatus
    conversation_status_mapping = {
        SandboxStatus.RUNNING: ConversationStatus.RUNNING,
        SandboxStatus.STARTING: ConversationStatus.STARTING,
        SandboxStatus.PAUSED: ConversationStatus.STOPPED,
        SandboxStatus.ERROR: ConversationStatus.ERROR,
        SandboxStatus.MISSING: ConversationStatus.ARCHIVED,
    }

    conversation_status = conversation_status_mapping.get(
        app_conversation.sandbox_status, ConversationStatus.STOPPED
    )

    if conversation_status == ConversationStatus.RUNNING:
        runtime_status_mapping = {
            ConversationExecutionStatus.ERROR: RuntimeStatus.ERROR,
            ConversationExecutionStatus.IDLE: RuntimeStatus.READY,
            ConversationExecutionStatus.RUNNING: RuntimeStatus.READY,
            ConversationExecutionStatus.PAUSED: RuntimeStatus.READY,
            ConversationExecutionStatus.WAITING_FOR_CONFIRMATION: RuntimeStatus.READY,
            ConversationExecutionStatus.FINISHED: RuntimeStatus.READY,
            ConversationExecutionStatus.STUCK: RuntimeStatus.ERROR,
        }
        runtime_status = runtime_status_mapping.get(
            app_conversation.execution_status, RuntimeStatus.ERROR
        )
    else:
        runtime_status = None

    title = (
        app_conversation.title
        or f'Conversation {base62.encodebytes(app_conversation.id.bytes)}'
    )

    return ConversationInfo(
        conversation_id=app_conversation.id.hex,
        title=title,
        last_updated_at=app_conversation.updated_at,
        status=conversation_status,
        runtime_status=runtime_status,
        selected_repository=app_conversation.selected_repository,
        selected_branch=app_conversation.selected_branch,
        git_provider=app_conversation.git_provider,
        trigger=app_conversation.trigger,
        num_connections=0,  # V1 conversations don't track connections the same way
        url=app_conversation.conversation_url,
        session_api_key=app_conversation.session_api_key,
        created_at=app_conversation.created_at,
        pr_number=app_conversation.pr_number,
        conversation_version='V1',
        sub_conversation_ids=[
            sub_id.hex for sub_id in app_conversation.sub_conversation_ids
        ],
        public=app_conversation.public,
    )
