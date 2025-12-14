import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from openhands.app_server.public_conversations.public_conversation_models import (
    PublicConversationInfo,
    PublicConversationInfoPage,
    PublicConversationSortOrder,
)
from openhands.app_server.services.injector import Injector
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class PublicConversationInfoService(ABC):
    """Service for accessing info on public conversations without their current status."""

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
    ) -> PublicConversationInfoPage:
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
    ) -> PublicConversationInfo | None:
        """Get a single public conversation info, returning None if missing or not public."""

    @abstractmethod
    async def get_public_conversation_info_by_token(
        self, token: str
    ) -> PublicConversationInfo | None:
        """Get a single public conversation info by share token, returning None if missing or not public."""


def depends_public_conversation_info_service() -> PublicConversationInfoService:
    """Dependency injection for PublicConversationInfoService."""
    return Injector.get(PublicConversationInfoService)


async def update_public_conversation_info_on_conversation_state_update(
    event: ConversationStateUpdateEvent,
):
    """Update public conversation info when conversation state changes."""
    if not isinstance(event, ConversationStateUpdateEvent):
        return

    public_conversation_info_service = depends_public_conversation_info_service()
    if hasattr(public_conversation_info_service, 'update_public_conversation_info'):
        await public_conversation_info_service.update_public_conversation_info(event)


# Register the event handler
DiscriminatedUnionMixin.register_event_handler(
    ConversationStateUpdateEvent,
    update_public_conversation_info_on_conversation_state_update,
)