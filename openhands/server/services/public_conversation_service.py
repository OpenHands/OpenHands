"""Service for handling public conversation operations."""

import secrets
from datetime import datetime, timezone

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.message import MessageAction
from openhands.events.event_store import EventStore
from openhands.server.data_models.public_conversation import (
    PublicConversationDetail,
    PublicConversationInfo,
    PublicMessageInfo,
)
from openhands.server.shared import file_store
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.conversation_metadata import ConversationMetadata


class PublicConversationService:
    """Service for managing public conversation sharing."""

    def __init__(self, conversation_store: ConversationStore):
        self.conversation_store = conversation_store

    async def make_conversation_public(self, conversation_id: str, user_id: str) -> str:
        """Make a conversation public and return the share token.

        Args:
            conversation_id: The conversation to make public
            user_id: The user making the conversation public

        Returns:
            The public share token

        Raises:
            FileNotFoundError: If conversation doesn't exist
            PermissionError: If user doesn't own the conversation
        """
        # Validate user owns the conversation
        if not await self.conversation_store.validate_metadata(
            conversation_id, user_id
        ):
            raise PermissionError('User does not own this conversation')

        # Get existing metadata
        metadata = await self.conversation_store.get_metadata(conversation_id)

        # Generate share token if not already public
        if not metadata.is_public:
            metadata.is_public = True
            metadata.public_share_token = secrets.token_urlsafe(32)
            metadata.shared_at = datetime.now(timezone.utc)

            # Save updated metadata
            await self.conversation_store.save_metadata(metadata)

        return metadata.public_share_token or ''

    async def make_conversation_private(
        self, conversation_id: str, user_id: str
    ) -> None:
        """Make a conversation private again.

        Args:
            conversation_id: The conversation to make private
            user_id: The user making the conversation private

        Raises:
            FileNotFoundError: If conversation doesn't exist
            PermissionError: If user doesn't own the conversation
        """
        # Validate user owns the conversation
        if not await self.conversation_store.validate_metadata(
            conversation_id, user_id
        ):
            raise PermissionError('User does not own this conversation')

        # Get existing metadata
        metadata = await self.conversation_store.get_metadata(conversation_id)

        # Make private
        metadata.is_public = False
        metadata.public_share_token = None
        metadata.shared_at = None

        # Save updated metadata
        await self.conversation_store.save_metadata(metadata)

    async def get_public_conversation(
        self, conversation_id: str
    ) -> PublicConversationInfo | None:
        """Get public conversation info by ID.

        Args:
            conversation_id: The conversation ID

        Returns:
            Public conversation info or None if not public/not found
        """
        try:
            metadata = await self.conversation_store.get_metadata(conversation_id)
            if not metadata.is_public:
                return None
            return self._to_public_conversation_info(metadata)
        except FileNotFoundError:
            return None

    async def get_public_conversation_by_token(
        self, share_token: str
    ) -> PublicConversationInfo | None:
        """Get public conversation info by share token.

        Args:
            share_token: The public share token

        Returns:
            Public conversation info or None if not found
        """
        # Note: This would require a more efficient lookup in a real implementation
        # For now, we'll need to search through conversations
        # In production, you'd want an index on public_share_token
        logger.warning('Token-based lookup not efficiently implemented yet')
        return None

    async def get_public_conversation_messages(
        self, conversation_id: str
    ) -> list[PublicMessageInfo]:
        """Get filtered public messages for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            List of public-safe messages
        """
        try:
            # First verify conversation is public
            metadata = await self.conversation_store.get_metadata(conversation_id)
            if not metadata.is_public:
                return []

            # Get event store for this conversation
            event_store = EventStore(conversation_id, file_store, '')
            events = event_store.get_events()

            messages = []
            for event in events:
                # Filter to only include safe message types
                if isinstance(event, MessageAction):
                    # Only include user messages and safe assistant responses
                    if event.source in ['user', 'assistant']:
                        # Filter out sensitive content
                        content = self._filter_sensitive_content(event.content)
                        if content:  # Only add if content remains after filtering
                            # Ensure timestamp is a datetime object
                            event_timestamp = event.timestamp
                            if isinstance(event_timestamp, str):
                                try:
                                    parsed_timestamp = datetime.fromisoformat(
                                        event_timestamp.replace('Z', '+00:00')
                                    )
                                except ValueError:
                                    parsed_timestamp = datetime.now(timezone.utc)
                            else:
                                # event_timestamp is None
                                parsed_timestamp = datetime.now(timezone.utc)

                            messages.append(
                                PublicMessageInfo(
                                    id=str(event.id)
                                    if hasattr(event, 'id')
                                    else str(hash(event)),
                                    timestamp=parsed_timestamp,
                                    source=event.source,
                                    content=content,
                                )
                            )

            return messages

        except FileNotFoundError:
            return []

    async def get_public_conversation_detail(
        self, conversation_id: str
    ) -> PublicConversationDetail | None:
        """Get complete public conversation with messages.

        Args:
            conversation_id: The conversation ID

        Returns:
            Complete public conversation or None if not public/not found
        """
        conversation_info = await self.get_public_conversation(conversation_id)
        if not conversation_info:
            return None

        messages = await self.get_public_conversation_messages(conversation_id)

        return PublicConversationDetail(
            conversation=conversation_info, messages=messages
        )

    def _to_public_conversation_info(
        self, metadata: ConversationMetadata
    ) -> PublicConversationInfo:
        """Convert ConversationMetadata to public-safe info."""
        return PublicConversationInfo(
            conversation_id=metadata.conversation_id,
            title=metadata.title or f'Conversation {metadata.conversation_id[:8]}',
            selected_repository=metadata.selected_repository,
            selected_branch=metadata.selected_branch,
            git_provider=metadata.git_provider,
            trigger=metadata.trigger,
            created_at=metadata.created_at,
            last_updated_at=metadata.last_updated_at,
            shared_at=metadata.shared_at,
        )

    def _filter_sensitive_content(self, content: str) -> str:
        """Filter out sensitive information from message content.

        Args:
            content: Original message content

        Returns:
            Filtered content with sensitive data removed
        """
        # Basic filtering - in production you'd want more sophisticated filtering
        sensitive_patterns = [
            'api_key',
            'token',
            'password',
            'secret',
            'key=',
            'authorization:',
            'bearer ',
            'x-api-key',
        ]

        filtered_content = content
        for pattern in sensitive_patterns:
            if pattern.lower() in filtered_content.lower():
                logger.warning(f'Filtering sensitive content containing: {pattern}')
                # For now, just log and return a placeholder
                # In production, you'd want more sophisticated filtering
                filtered_content = '[Content filtered for security]'
                break

        return filtered_content
