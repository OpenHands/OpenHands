import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from openhands.app_server.services.injector import Injector
from openhands.app_server.sharing.public_conversation_models import (
    PublicConversation,
    PublicConversationPage,
    PublicConversationSortOrder,
)
# Simple implementation of DiscriminatedUnionMixin for now
class DiscriminatedUnionMixin:
    """Simple mixin for discriminated unions."""
    pass


class PublicConversationInfoService(ABC):
    """Service for accessing public conversation info without user restrictions."""

    @abstractmethod
    async def search_public_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sort_order: PublicConversationSortOrder = PublicConversationSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 100,
        include_sub_conversations: bool = False,
    ) -> PublicConversationPage:
        """Search for public conversations."""

    @abstractmethod
    async def count_public_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
    ) -> int:
        """Count public conversations."""

    @abstractmethod
    async def get_public_conversation_info(
        self, conversation_id: UUID
    ) -> PublicConversation | None:
        """Get a single public conversation info, returning None if missing or not public."""

    async def batch_get_public_conversation_info(
        self, conversation_ids: list[UUID]
    ) -> list[PublicConversation | None]:
        """Get a batch of public conversation info, return None for any missing or non-public."""
        return await asyncio.gather(
            *[
                self.get_public_conversation_info(conversation_id)
                for conversation_id in conversation_ids
            ]
        )


class PublicConversationInfoServiceInjector(
    DiscriminatedUnionMixin, Injector[PublicConversationInfoService], ABC
):
    pass