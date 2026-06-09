import asyncio
from abc import ABC, abstractmethod
from uuid import UUID

from server.sharing.shared_conversation_models import (
    SharedConversation,
)

from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class SharedConversationInfoService(ABC):
    """Service for accessing shared conversation info without user restrictions."""

    @abstractmethod
    async def get_shared_conversation_info(
        self, conversation_id: UUID
    ) -> SharedConversation | None:
        """Get a single shared conversation info, returning None if missing or not shared."""

    async def batch_get_shared_conversation_info(
        self, conversation_ids: list[UUID]
    ) -> list[SharedConversation | None]:
        """Get a batch of shared conversation info, return None for any missing or non-shared."""
        return await asyncio.gather(
            *[
                self.get_shared_conversation_info(conversation_id)
                for conversation_id in conversation_ids
            ]
        )


class SharedConversationInfoServiceInjector(
    DiscriminatedUnionMixin, Injector[SharedConversationInfoService], ABC
):
    pass
