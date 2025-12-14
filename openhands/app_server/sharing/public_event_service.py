import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.services.injector import Injector
from openhands.sdk import Event
from openhands.sdk.utils.models import DiscriminatedUnionMixin

_logger = logging.getLogger(__name__)


class PublicEventService(ABC):
    """Event Service for getting events from public conversations only."""

    @abstractmethod
    async def get_public_event(
        self, conversation_id: UUID, event_id: str
    ) -> Event | None:
        """Given a conversation_id and event_id, retrieve an event if the conversation is public."""

    @abstractmethod
    async def search_public_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        """Search events for a specific public conversation."""

    @abstractmethod
    async def count_public_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
    ) -> int:
        """Count events for a specific public conversation."""

    async def batch_get_public_events(
        self, conversation_id: UUID, event_ids: list[str]
    ) -> list[Event | None]:
        """Given a conversation_id and list of event_ids, get events if the conversation is public."""
        return await asyncio.gather(
            *[
                self.get_public_event(conversation_id, event_id)
                for event_id in event_ids
            ]
        )


class PublicEventServiceInjector(
    DiscriminatedUnionMixin, Injector[PublicEventService], ABC
):
    pass
