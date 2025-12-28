import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
)
from openhands.app_server.config import depends_app_conversation_service
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.message import MessageAction
from openhands.events.event_filter import EventFilter
from openhands.events.event_store import EventStore
from openhands.events.serialization.event import event_to_dict
from openhands.memory.memory import Memory
from openhands.microagent.types import InputMetadata
from openhands.runtime.base import Runtime
from openhands.server.dependencies import get_dependencies
from openhands.server.session.conversation import ServerConversation
from openhands.server.shared import conversation_manager, file_store
from openhands.server.user_auth import get_user_id
from openhands.server.utils import get_conversation, get_conversation_metadata
from openhands.storage.data_models.conversation_metadata import ConversationMetadata

app = APIRouter(
    prefix='/api/conversations/{conversation_id}', dependencies=get_dependencies()
)

# Dependency for app conversation service
app_conversation_service_dependency = depends_app_conversation_service()


async def _is_v1_conversation(
    conversation_id: str, app_conversation_service: AppConversationService
) -> bool:
    """Check if the given conversation_id corresponds to a V1 conversation.

    Args:
        conversation_id: The conversation ID to check
        app_conversation_service: Service to query V1 conversations

    Returns:
        True if this is a V1 conversation, False otherwise
    """
    try:
        conversation_uuid = uuid.UUID(conversation_id)
        app_conversation = await app_conversation_service.get_app_conversation(
            conversation_uuid
        )
        return app_conversation is not None
    except (ValueError, TypeError):
        # Not a valid UUID, so it's not a V1 conversation
        return False
    except Exception:
        # Service error, assume it's not a V1 conversation
        return False


async def _get_v1_conversation_config(
    conversation_id: str, app_conversation_service: AppConversationService
) -> dict[str, str | None]:
    """Get configuration for a V1 conversation.

    Args:
        conversation_id: The conversation ID
        app_conversation_service: Service to query V1 conversations

    Returns:
        Dictionary with runtime_id (sandbox_id) and session_id (conversation_id)
    """
    conversation_uuid = uuid.UUID(conversation_id)
    app_conversation = await app_conversation_service.get_app_conversation(
        conversation_uuid
    )

    if app_conversation is None:
        raise ValueError(f'V1 conversation {conversation_id} not found')

    return {
        'runtime_id': app_conversation.sandbox_id,
        'session_id': conversation_id,
    }


def _get_v0_conversation_config(
    conversation: ServerConversation,
) -> dict[str, str | None]:
    """Get configuration for a V0 conversation.

    Args:
        conversation: The server conversation object

    Returns:
        Dictionary with runtime_id and session_id from the runtime
    """
    runtime = conversation.runtime
    runtime_id = runtime.runtime_id if hasattr(runtime, 'runtime_id') else None
    session_id = runtime.sid if hasattr(runtime, 'sid') else None

    return {
        'runtime_id': runtime_id,
        'session_id': session_id,
    }


@app.get('/config')
async def get_remote_runtime_config(
    conversation_id: str,
    app_conversation_service: AppConversationService = app_conversation_service_dependency,
    user_id: str | None = Depends(get_user_id),
) -> JSONResponse:
    """Retrieve the runtime configuration.

    For V0 conversations: returns runtime_id and session_id from the runtime.
    For V1 conversations: returns sandbox_id as runtime_id and conversation_id as session_id.
    """
    # Check if this is a V1 conversation first
    if await _is_v1_conversation(conversation_id, app_conversation_service):
        # This is a V1 conversation
        config = await _get_v1_conversation_config(
            conversation_id, app_conversation_service
        )
    else:
        # V0 conversation - get the conversation and use the existing logic
        conversation = await conversation_manager.attach_to_conversation(
            conversation_id, user_id
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Conversation {conversation_id} not found',
            )
        try:
            config = _get_v0_conversation_config(conversation)
        finally:
            await conversation_manager.detach_from_conversation(conversation)

    return JSONResponse(content=config)


@app.get('/vscode-url')
async def get_vscode_url(
    conversation: ServerConversation = Depends(get_conversation),
) -> JSONResponse:
    """Get the VSCode URL.

    This endpoint allows getting the VSCode URL.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        JSONResponse: A JSON response indicating the success of the operation.
    """
    try:
        runtime: Runtime = conversation.runtime
        logger.debug(f'Runtime type: {type(runtime)}')
        logger.debug(f'Runtime VSCode URL: {runtime.vscode_url}')
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={'vscode_url': runtime.vscode_url}
        )
    except Exception as e:
        logger.error(f'Error getting VSCode URL: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                'vscode_url': None,
                'error': f'Error getting VSCode URL: {e}',
            },
        )


@app.get('/web-hosts')
async def get_hosts(
    conversation: ServerConversation = Depends(get_conversation),
) -> JSONResponse:
    """Get the hosts used by the runtime.

    This endpoint allows getting the hosts used by the runtime.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        JSONResponse: A JSON response indicating the success of the operation.
    """
    try:
        runtime: Runtime = conversation.runtime
        logger.debug(f'Runtime type: {type(runtime)}')
        logger.debug(f'Runtime hosts: {runtime.web_hosts}')
        return JSONResponse(status_code=200, content={'hosts': runtime.web_hosts})
    except Exception as e:
        logger.error(f'Error getting runtime hosts: {e}')
        return JSONResponse(
            status_code=500,
            content={
                'hosts': None,
                'error': f'Error getting runtime hosts: {e}',
            },
        )


@app.get('/events')
async def search_events(
    conversation_id: str,
    start_id: int = 0,
    end_id: int | None = None,
    reverse: bool = False,
    filter: EventFilter | None = None,
    limit: int = 20,
    metadata: ConversationMetadata = Depends(get_conversation_metadata),
    user_id: str | None = Depends(get_user_id),
):
    """Search through the event stream with filtering and pagination.

    Args:
        conversation_id: The conversation ID
        start_id: Starting ID in the event stream. Defaults to 0
        end_id: Ending ID in the event stream
        reverse: Whether to retrieve events in reverse order. Defaults to False.
        filter: Filter for events
        limit: Maximum number of events to return. Must be between 1 and 100. Defaults to 20
        metadata: Conversation metadata (injected by dependency)
        user_id: User ID (injected by dependency)

    Returns:
        dict: Dictionary containing:
            - events: List of matching events
            - has_more: Whether there are more matching events after this batch
    Raises:
        HTTPException: If conversation is not found or access is denied
        ValueError: If limit is less than 1 or greater than 100
    """
    if limit < 0 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid limit'
        )

    # Create an event store to access the events directly
    event_store = EventStore(
        sid=conversation_id,
        file_store=file_store,
        user_id=user_id,
    )

    # Get matching events from the store
    events = list(
        event_store.search_events(
            start_id=start_id,
            end_id=end_id,
            reverse=reverse,
            filter=filter,
            limit=limit + 1,
        )
    )

    # Check if there are more events
    has_more = len(events) > limit
    if has_more:
        events = events[:limit]  # Remove the extra event

    events_json = [event_to_dict(event) for event in events]
    return {
        'events': events_json,
        'has_more': has_more,
    }


@app.post('/events')
async def add_event(
    request: Request, conversation: ServerConversation = Depends(get_conversation)
):
    data = await request.json()
    await conversation_manager.send_event_to_conversation(conversation.sid, data)
    return JSONResponse({'success': True})


class AddMessageRequest(BaseModel):
    """Request model for adding a message to a conversation."""

    message: str


@app.post('/message')
async def add_message(
    data: AddMessageRequest,
    conversation: ServerConversation = Depends(get_conversation),
):
    """Add a message to an existing conversation.

    This endpoint allows adding a user message to an existing conversation.
    The message will be processed by the agent in the conversation.

    Args:
        data: The request data containing the message text
        conversation: The conversation to add the message to (injected by dependency)

    Returns:
        JSONResponse: A JSON response indicating the success of the operation
    """
    try:
        # Create a MessageAction from the provided message text
        message_action = MessageAction(content=data.message)

        # Convert the action to a dictionary for sending to the conversation
        message_data = event_to_dict(message_action)

        # Send the message to the conversation
        await conversation_manager.send_event_to_conversation(
            conversation.sid, message_data
        )

        return JSONResponse({'success': True})
    except Exception as e:
        logger.error(f'Error adding message to conversation: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                'success': False,
                'error': f'Error adding message to conversation: {e}',
            },
        )


class MicroagentResponse(BaseModel):
    """Response model for microagents endpoint."""

    name: str
    type: str
    content: str
    triggers: list[str] = []
    inputs: list[InputMetadata] = []
    tools: list[str] = []


@app.get('/microagents')
async def get_microagents(
    conversation: ServerConversation = Depends(get_conversation),
) -> JSONResponse:
    """Get all microagents associated with the conversation.

    This endpoint returns all repository and knowledge microagents that are loaded for the conversation.

    Returns:
        JSONResponse: A JSON response containing the list of microagents.
    """
    try:
        # Get the agent session for this conversation
        agent_session = conversation_manager.get_agent_session(conversation.sid)

        if not agent_session:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={'error': 'Agent session not found for this conversation'},
            )

        # Access the memory to get the microagents
        memory: Memory | None = agent_session.memory
        if memory is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    'error': 'Memory is not yet initialized for this conversation'
                },
            )

        # Prepare the response
        microagents = []

        # Add repo microagents
        for name, r_agent in memory.repo_microagents.items():
            microagents.append(
                MicroagentResponse(
                    name=name,
                    type='repo',
                    content=r_agent.content,
                    triggers=[],
                    inputs=r_agent.metadata.inputs,
                    tools=(
                        [
                            server.name
                            for server in r_agent.metadata.mcp_tools.stdio_servers
                        ]
                        if r_agent.metadata.mcp_tools
                        else []
                    ),
                )
            )

        # Add knowledge microagents
        for name, k_agent in memory.knowledge_microagents.items():
            microagents.append(
                MicroagentResponse(
                    name=name,
                    type='knowledge',
                    content=k_agent.content,
                    triggers=k_agent.triggers,
                    inputs=k_agent.metadata.inputs,
                    tools=(
                        [
                            server.name
                            for server in k_agent.metadata.mcp_tools.stdio_servers
                        ]
                        if k_agent.metadata.mcp_tools
                        else []
                    ),
                )
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'microagents': [m.dict() for m in microagents]},
        )
    except Exception as e:
        logger.error(f'Error getting microagents: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error getting microagents: {e}'},
        )
