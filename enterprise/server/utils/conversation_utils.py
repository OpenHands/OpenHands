"""Utility functions for conversation operations."""

from uuid import UUID

from storage.database import session_maker
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas


def get_user_id(conversation_id: str) -> str:
    """Get the user ID for a conversation from the metadata.

    Args:
        conversation_id: The conversation ID

    Returns:
        The user ID as a string
    """
    with session_maker() as session:
        conversation_metadata_saas = (
            session.query(StoredConversationMetadataSaas)
            .filter(StoredConversationMetadataSaas.conversation_id == conversation_id)
            .first()
        )
        if not conversation_metadata_saas:
            raise ValueError(f'Conversation not found: {conversation_id}')
        return str(conversation_metadata_saas.user_id)


async def get_session_api_key(conversation_id: str) -> str | None:
    """Get the session API key for a conversation.

    This retrieves the session API key from the V1 sandbox system.

    Args:
        conversation_id: The conversation ID

    Returns:
        The session API key, or None if not available
    """
    from openhands.app_server.config import (
        get_app_conversation_info_service,
        get_sandbox_service,
    )
    from openhands.app_server.services.injector import InjectorState
    from openhands.app_server.user.specifiy_user_context import (
        ADMIN,
        USER_CONTEXT_ATTR,
    )

    # Create injector state for dependency injection
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)

    async with (
        get_app_conversation_info_service(state) as app_conversation_info_service,
        get_sandbox_service(state) as sandbox_service,
    ):
        # Get the conversation info to find the sandbox_id
        app_conversation_info = (
            await app_conversation_info_service.get_app_conversation_info(
                UUID(conversation_id)
            )
        )
        if not app_conversation_info:
            return None

        # Get the sandbox to retrieve the session API key
        sandbox = await sandbox_service.get_sandbox(app_conversation_info.sandbox_id)
        if not sandbox:
            return None

        return sandbox.session_api_key
