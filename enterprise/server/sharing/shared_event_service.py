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


class SharedEventService(ABC):
    """Event Service for getting events from shared conversations only."""

    @abstractmethod
    async def get_shared_event(
        self, conversation_id: UUID, event_id: UUID
    ) -> Event | None:
        """Given a conversation_id and event_id, retrieve an event if the conversation is shared."""

    @abstractmethod
    async def search_shared_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        """Search events for a specific shared conversation."""

    @abstractmethod
    async def count_shared_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
    ) -> int:
        """Count events for a specific shared conversation."""

    async def batch_get_shared_events(
        self, conversation_id: UUID, event_ids: list[UUID]
    ) -> list[Event | None]:
        """Given a conversation_id and list of event_ids, get events if the conversation is shared."""
        return await asyncio.gather(
            *[
                self.get_shared_event(conversation_id, event_id)
                for event_id in event_ids
            ]
        )


class SharedEventServiceInjector(
    DiscriminatedUnionMixin, Injector[SharedEventService], ABC
):
    pass
