from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.storage.data_models.conversation_metadata import ConversationMetadata
from openhands.storage.data_models.conversation_metadata_result_set import (
    ConversationMetadataResultSet,
)
from openhands.utils.async_utils import wait_all


class ConversationStore(ABC):
    """Abstract base class for conversation metadata storage.

    This is an extension point in OpenHands that allows applications to customize how
    conversation metadata is stored. Applications can substitute their own implementation by:
    1. Creating a class that inherits from ConversationStore
    2. Implementing all required methods
    3. Setting server_config.conversation_store_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.

    The implementation may or may not support multiple users depending on the environment.
    """

    @abstractmethod
    async def save_metadata(self, metadata: ConversationMetadata) -> None:
        """Store conversation metadata."""

    @abstractmethod
    async def get_metadata(self, conversation_id: str) -> ConversationMetadata:
        """Load conversation metadata."""

    async def validate_metadata(self, conversation_id: str, user_id: str) -> bool:
        """Validate that conversation belongs to the current user."""
        metadata = await self.get_metadata(conversation_id)
        if not metadata.user_id or metadata.user_id != user_id:
            return False
        else:
            return True

    @abstractmethod
    async def delete_metadata(self, conversation_id: str) -> None:
        """Delete conversation metadata."""

    @abstractmethod
    async def exists(self, conversation_id: str) -> bool:
        """Check if conversation exists."""

    @abstractmethod
    async def search(
        self,
        page_id: str | None = None,
        limit: int = 20,
    ) -> ConversationMetadataResultSet:
        """Search conversations."""

    async def get_all_metadata(
        self, conversation_ids: Iterable[str]
    ) -> list[ConversationMetadata]:
        """Get metadata for multiple conversations in parallel."""
        return await wait_all([self.get_metadata(cid) for cid in conversation_ids])

    @classmethod
    @abstractmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> ConversationStore:
        """Get a store for the user represented by the token given."""
