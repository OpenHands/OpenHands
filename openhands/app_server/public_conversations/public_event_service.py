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
    """Event Service for getting events from public conversations."""

    @abstractmethod
    async def get_public_event(self, event_id: str) -> Event | None:
        """Given an id, retrieve an event from a public conversation."""

    @abstractmethod
    async def search_public_events(
        self,
        conversation_id__eq: UUID | None = None,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        """Search events from public conversations matching the given filters."""

    @abstractmethod
    async def count_public_events(
        self,
        conversation_id__eq: UUID | None = None,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
    ) -> int:
        """Count events from public conversations matching the given filters."""

    @abstractmethod
    async def batch_get_public_events(self, event_ids: list[str]) -> list[Event | None]:
        """Get a batch of events from public conversations given their ids."""

    @abstractmethod
    async def search_public_events_by_token(
        self,
        token: str,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        """Search events from a public conversation by share token."""


def depends_public_event_service() -> PublicEventService:
    """Dependency injection for PublicEventService."""
    return Injector.get(PublicEventService)