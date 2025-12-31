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


class EventService(ABC):
    """Event Service for getting events."""

    @abstractmethod
    async def get_event(self, event_id: str) -> Event | None:
        """Given an id, retrieve an event."""

    @abstractmethod
    async def search_events(
        self,
        conversation_id__eq: UUID | None = None,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        """Search events matching the given filters."""

    @abstractmethod
    async def count_events(
        self,
        conversation_id__eq: UUID | None = None,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
    ) -> int:
        """Count events matching the given filters."""

    @abstractmethod
    async def save_event(self, conversation_id: UUID, event: Event):
        """Save an event. Internal method intended not be part of the REST api."""

    async def batch_get_events(self, event_ids: list[str]) -> list[Event | None]:
        """Given a list of ids, get events (Or none for any which were not found)."""
        return await asyncio.gather(
            *[self.get_event(event_id) for event_id in event_ids]
        )


class EventServiceInjector(DiscriminatedUnionMixin, Injector[EventService], ABC):
    pass
