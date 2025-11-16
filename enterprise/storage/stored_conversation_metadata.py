"""StoredConversationMetadata import for enterprise telemetry framework.

This module provides access to the StoredConversationMetadata class from the
main OpenHands codebase for use in enterprise telemetry collectors.
"""

from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    StoredConversationMetadata as _StoredConversationMetadata,
)

StoredConversationMetadata = _StoredConversationMetadata


__all__ = ['StoredConversationMetadata']
