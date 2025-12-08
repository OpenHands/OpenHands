"""Utility functions for generating conversation summaries."""

from typing import Optional

from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.message import MessageAction
from openhands.events.event import EventSource
from openhands.events.event_store import EventStore
from openhands.llm.llm_registry import LLMRegistry
from openhands.storage.data_models.settings import Settings
from openhands.storage.files import FileStore
from openhands.utils.environment import get_effective_llm_base_url


async def generate_conversation_title(
    message: str,
    llm_config: LLMConfig,
    llm_registry: LLMRegistry,
    max_length: int = 50,
) -> Optional[str]:
    """Generate a concise title for a conversation based on the first user message.

    Args:
        message: The first user message in the conversation.
        llm_config: The LLM configuration to use for generating the title.
        max_length: The maximum length of the generated title.

    Returns:
        A concise title for the conversation, or None if generation fails.
    """
    if not message or message.strip() == '':
        return None

    # Truncate very long messages to avoid excessive token usage
    if len(message) > 1000:
        truncated_message = message[:1000] + '...(truncated)'
    else:
        truncated_message = message

    try:
        # Create a simple prompt for the LLM to generate a title
        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that generates concise, descriptive titles for conversations with OpenHands. OpenHands is a helpful AI agent that can interact with a computer to solve tasks using bash terminal, file editor, and browser. Given a user message (which may be truncated), generate a concise, descriptive title for the conversation. Return only the title, with no additional text, quotes, or explanations.',
            },
            {
                'role': 'user',
                'content': f'Generate a title (maximum {max_length} characters) for a conversation that starts with this message:\n\n{truncated_message}',
            },
        ]

        title = llm_registry.request_extraneous_completion(
            'conversation_title_creator', llm_config, messages
        )

        # Ensure the title isn't too long
        if len(title) > max_length:
            title = title[: max_length - 3] + '...'

        return title
    except Exception as e:
        logger.error(f'Error generating conversation title: {e}')
        return None


def get_default_conversation_title(conversation_id: str) -> str:
    """Generate a default title for a conversation based on its ID.

    Args:
        conversation_id: The ID of the conversation

    Returns:
        A default title string
    """
    return f'Conversation {conversation_id[:5]}'


async def auto_generate_title(
    conversation_id: str,
    user_id: str | None,
    file_store: FileStore,
    settings: Settings,
    llm_registry: LLMRegistry,
) -> str:
    """Auto-generate a title for a conversation based on the first user message.
    Uses LLM-based title generation if available, otherwise falls back to a simple truncation.

    Args:
        conversation_id: The ID of the conversation
        user_id: The ID of the user

    Returns:
        A generated title string
    """
    try:
        # Create an event store for the conversation
        event_store = EventStore(conversation_id, file_store, user_id)

        # Find the first user message
        first_user_message = None
        for event in event_store.search_events():
            if (
                event.source == EventSource.USER
                and isinstance(event, MessageAction)
                and event.content
                and event.content.strip()
            ):
                first_user_message = event.content
                break

        if first_user_message:
            # Get LLM config from user settings
            try:
                if settings and settings.llm_model:
                    # Create LLM config from settings
                    settings_base_url = settings.llm_base_url
                    effective_base_url = get_effective_llm_base_url(
                        settings.llm_model,
                        settings_base_url,
                    )
                    llm_config = LLMConfig(
                        model=settings.llm_model,
                        api_key=settings.llm_api_key,
                        base_url=effective_base_url,
                    )

                    # Try to generate title using LLM
                    llm_title = await generate_conversation_title(
                        first_user_message, llm_config, llm_registry
                    )
                    if llm_title:
                        logger.info(f'Generated title using LLM: {llm_title}')
                        return llm_title
            except Exception as e:
                logger.error(f'Error using LLM for title generation: {e}')

            # Fall back to simple truncation if LLM generation fails or is unavailable
            first_user_message = first_user_message.strip()
            title = first_user_message[:30]
            if len(first_user_message) > 30:
                title += '...'
            logger.info(f'Generated title using truncation: {title}')
            return title
    except Exception as e:
        logger.error(f'Error generating title: {str(e)}')
    return ''
