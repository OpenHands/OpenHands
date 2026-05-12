"""Conversation path helpers for consistent path construction.

This module provides helper functions for constructing conversation-related
storage paths. Use these helpers instead of hardcoding path patterns to ensure
consistency across the codebase.
"""

from pathlib import Path
from uuid import UUID

# The base directory name for v1 conversation storage
V1_CONVERSATIONS_DIR = 'v1_conversations'


def get_conversation_dir(conversation_id: UUID | str) -> str:
    """Get the conversation directory path segment.

    Args:
        conversation_id: The conversation ID (UUID or hex string)

    Returns:
        Path segment like 'v1_conversations/{conversation_id_hex}'

    Example:
        >>> get_conversation_dir(UUID('12345678-1234-5678-1234-567812345678'))
        'v1_conversations/12345678123456781234567812345678'
        >>> get_conversation_dir('12345678123456781234567812345678')
        'v1_conversations/12345678123456781234567812345678'
    """
    if isinstance(conversation_id, UUID):
        conversation_id_hex = conversation_id.hex
    else:
        # Already a hex string
        conversation_id_hex = conversation_id
    return f'{V1_CONVERSATIONS_DIR}/{conversation_id_hex}'


def get_conversation_path(
    conversation_id: UUID | str,
    user_id: str | None = None,
    prefix: Path | str | None = None,
) -> Path:
    """Get the full conversation path.

    Args:
        conversation_id: The conversation ID (UUID or hex string)
        user_id: Optional user ID to include in path
        prefix: Optional prefix path

    Returns:
        Full path like '{prefix}/{user_id}/v1_conversations/{conversation_id_hex}'

    Example:
        >>> get_conversation_path(UUID('...'), user_id='user123', prefix=Path('/data'))
        Path('/data/user123/v1_conversations/...')
    """
    if isinstance(conversation_id, UUID):
        conversation_id_hex = conversation_id.hex
    else:
        conversation_id_hex = conversation_id

    parts: list[str] = []

    if prefix:
        parts.append(str(prefix))

    if user_id:
        parts.append(user_id)

    parts.append(V1_CONVERSATIONS_DIR)
    parts.append(conversation_id_hex)

    return Path(*parts) if parts else Path(V1_CONVERSATIONS_DIR) / conversation_id_hex
