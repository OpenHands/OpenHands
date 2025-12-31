import base64
import json
import pickle
from datetime import datetime

from server.logger import logger
from storage.conversation_callback import (
    CallbackStatus,
    ConversationCallback,
    ConversationCallbackProcessor,
)
from storage.conversation_work import ConversationWork
from storage.database import session_maker
from storage.stored_conversation_metadata import StoredConversationMetadata

from openhands.core.config import load_openhands_config
from openhands.core.schema.agent import AgentState
from openhands.events.event_store import EventStore
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.serialization.event import event_from_dict
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage import get_file_store
from openhands.storage.files import FileStore
from openhands.storage.locations import (
    get_conversation_agent_state_filename,
    get_conversation_dir,
)
from openhands.utils.async_utils import call_sync_from_async

config = load_openhands_config()
file_store = get_file_store(config.file_store, config.file_store_path)


async def process_event(
    user_id: str, conversation_id: str, subpath: str, content: dict
):
    """
    Process a conversation event and invoke any registered callbacks.

    Args:
        user_id: The user ID associated with the conversation
        conversation_id: The conversation ID
        subpath: The event subpath
        content: The event content
    """
    logger.debug(
        'process_event',
        extra={
            'user_id': user_id,
            'conversation_id': conversation_id,
            'content': content,
        },
    )
    write_path = get_conversation_dir(conversation_id, user_id) + subpath

    # This writes to the google cloud storage, so we do this in a background thread to not block the main runloop...
    await call_sync_from_async(file_store.write, write_path, json.dumps(content))

    event = event_from_dict(content)
    if isinstance(event, AgentStateChangedObservation):
        # Load and invoke all active callbacks for this conversation
        await invoke_conversation_callbacks(conversation_id, event)

        # Update active working seconds if agent state is not Running
        if event.agent_state != AgentState.RUNNING:
            event_store = EventStore(conversation_id, file_store, user_id)
            update_active_working_seconds(
                event_store, conversation_id, user_id, file_store
            )


async def invoke_conversation_callbacks(
    conversation_id: str, observation: AgentStateChangedObservation
):
    """
    Load and invoke all active callbacks for a conversation.

    Args:
        conversation_id: The conversation ID to process callbacks for
        observation: The AgentStateChangedObservation that triggered the callback
    """
    with session_maker() as session:
        callbacks = (
            session.query(ConversationCallback)
            .filter(
                ConversationCallback.conversation_id == conversation_id,
                ConversationCallback.status == CallbackStatus.ACTIVE,
            )
            .all()
        )

        for callback in callbacks:
            try:
                processor = callback.get_processor()
                await processor.__call__(callback, observation)
                logger.info(
                    'callback_invoked_successfully',
                    extra={
                        'conversation_id': conversation_id,
                        'callback_id': callback.id,
                        'processor_type': callback.processor_type,
                    },
                )
            except Exception as e:
                logger.error(
                    'callback_invocation_failed',
                    extra={
                        'conversation_id': conversation_id,
                        'callback_id': callback.id,
                        'processor_type': callback.processor_type,
                        'error': str(e),
                    },
                )
                # Mark callback as error status
                callback.status = CallbackStatus.ERROR
                callback.updated_at = datetime.now()

        session.commit()


def update_conversation_metadata(conversation_id: str, content: dict):
    """
    Update conversation metadata with new content.

    Args:
        conversation_id: The conversation ID to update
        content: The metadata content to update
    """
    logger.debug(
        'update_conversation_metadata',
        extra={
            'conversation_id': conversation_id,
            'content': content,
        },
    )
    with session_maker() as session:
        conversation = (
            session.query(StoredConversationMetadata)
            .filter(StoredConversationMetadata.conversation_id == conversation_id)
            .first()
        )
        conversation.title = content.get('title') or conversation.title
        conversation.last_updated_at = datetime.now()
        conversation.accumulated_cost = (
            content.get('accumulated_cost') or conversation.accumulated_cost
        )
        conversation.prompt_tokens = (
            content.get('prompt_tokens') or conversation.prompt_tokens
        )
        conversation.completion_tokens = (
            content.get('completion_tokens') or conversation.completion_tokens
        )
        conversation.total_tokens = (
            content.get('total_tokens') or conversation.total_tokens
        )
        session.commit()


def register_callback_processor(
    conversation_id: str, processor: ConversationCallbackProcessor
) -> int:
    """
    Register a callback processor for a conversation.

    Args:
        conversation_id: The conversation ID to register the callback for
        processor: The ConversationCallbackProcessor instance to register

    Returns:
        int: The ID of the created callback
    """
    with session_maker() as session:
        callback = ConversationCallback(
            conversation_id=conversation_id, status=CallbackStatus.ACTIVE
        )
        callback.set_processor(processor)
        session.add(callback)
        session.commit()
        return callback.id


def update_active_working_seconds(
    event_store: EventStore, conversation_id: str, user_id: str, file_store: FileStore
):
    """
    Calculate and update the total active working seconds for a conversation.

    This function reads all events for the conversation, looks for AgentStateChanged
    observations, and calculates the total time spent in a running state.

    Args:
        event_store: The EventStore instance for reading events
        conversation_id: The conversation ID to process
        user_id: The user ID associated with the conversation
        file_store: The FileStore instance for accessing conversation data
    """
    try:
        # Track agent state changes and calculate running time
        running_start_time = None
        total_running_seconds = 0.0

        for event in event_store.search_events():
            if isinstance(event, AgentStateChangedObservation) and event.timestamp:
                event_timestamp = datetime.fromisoformat(event.timestamp).timestamp()

                if event.agent_state == AgentState.RUNNING:
                    # Agent started running
                    if running_start_time is None:
                        running_start_time = event_timestamp
                elif running_start_time is not None:
                    # Agent stopped running, calculate duration
                    duration = event_timestamp - running_start_time
                    total_running_seconds += duration
                    running_start_time = None

        # If agent is still running at the end, don't count that time yet
        # (it will be counted when the agent stops)

        # Create or update the conversation_work record
        with session_maker() as session:
            conversation_work = (
                session.query(ConversationWork)
                .filter(ConversationWork.conversation_id == conversation_id)
                .first()
            )

            if conversation_work:
                # Update existing record
                conversation_work.seconds = total_running_seconds
                conversation_work.updated_at = datetime.now().isoformat()
            else:
                # Create new record
                conversation_work = ConversationWork(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    seconds=total_running_seconds,
                )
                session.add(conversation_work)

            session.commit()

        logger.info(
            'updated_active_working_seconds',
            extra={
                'conversation_id': conversation_id,
                'user_id': user_id,
                'total_seconds': total_running_seconds,
            },
        )

    except Exception as e:
        logger.error(
            'failed_to_update_active_working_seconds',
            extra={
                'conversation_id': conversation_id,
                'user_id': user_id,
                'error': str(e),
            },
        )


def update_agent_state(user_id: str, conversation_id: str, content: bytes):
    """
    Update agent state file for a conversation.

    Args:
        user_id: The user ID associated with the conversation
        conversation_id: The conversation ID
        content: The agent state content as bytes
    """
    logger.debug(
        'update_agent_state',
        extra={
            'user_id': user_id,
            'conversation_id': conversation_id,
            'content_size': len(content),
        },
    )
    write_path = get_conversation_agent_state_filename(conversation_id, user_id)
    file_store.write(write_path, content)


def update_conversation_stats(user_id: str, conversation_id: str, content: bytes):
    existing_convo_stats = ConversationStats(
        file_store=file_store, conversation_id=conversation_id, user_id=user_id
    )

    incoming_convo_stats = ConversationStats(None, conversation_id, None)
    pickled = base64.b64decode(content)
    incoming_convo_stats.restored_metrics = pickle.loads(pickled)

    # Merging automatically saves to file store
    existing_convo_stats.merge_and_save(incoming_convo_stats)
