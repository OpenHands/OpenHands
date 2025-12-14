"""Implementation of PublicEventService.

This implementation provides read-only access to events from public conversations:
- Validates that the conversation is public before returning events
- Uses existing EventService for actual event retrieval
- Uses PublicConversationInfoService for public conversation validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.sharing.public_conversation_info_service import (
    PublicConversationInfoService,
)
from openhands.app_server.sharing.public_event_service import (
    PublicEventService,
    PublicEventServiceInjector,
)
from openhands.sdk import Event

logger = logging.getLogger(__name__)


@dataclass
class PublicEventServiceImpl(PublicEventService):
    """Implementation of PublicEventService that validates public access."""

    public_conversation_service: PublicConversationInfoService
    event_service: EventService

    async def get_public_event(self, conversation_id: UUID, event_id: str) -> Event | None:
        """Given a conversation_id and event_id, retrieve an event if the conversation is public."""
        # First check if the conversation is public
        public_conversation = await self.public_conversation_service.get_public_conversation_info(
            conversation_id
        )
        if public_conversation is None:
            return None

        # If conversation is public, get the event
        return await self.event_service.get_event(event_id)

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
        # First check if the conversation is public
        public_conversation = await self.public_conversation_service.get_public_conversation_info(
            conversation_id
        )
        if public_conversation is None:
            # Return empty page if conversation is not public
            return EventPage(items=[], next_page_id=None)

        # If conversation is public, search events for this conversation
        return await self.event_service.search_events(
            conversation_id__eq=conversation_id,
            kind__eq=kind__eq,
            timestamp__gte=timestamp__gte,
            timestamp__lt=timestamp__lt,
            sort_order=sort_order,
            page_id=page_id,
            limit=limit,
        )

    async def count_public_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
    ) -> int:
        """Count events for a specific public conversation."""
        # First check if the conversation is public
        public_conversation = await self.public_conversation_service.get_public_conversation_info(
            conversation_id
        )
        if public_conversation is None:
            return 0

        # If conversation is public, count events for this conversation
        return await self.event_service.count_events(
            conversation_id__eq=conversation_id,
            kind__eq=kind__eq,
            timestamp__gte=timestamp__gte,
            timestamp__lt=timestamp__lt,
            sort_order=sort_order,
        )


class PublicEventServiceImplInjector(PublicEventServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[PublicEventService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import (
            get_event_service,
            get_public_conversation_info_service,
        )

        async with (
            get_public_conversation_info_service(state, request) as public_conversation_service,
            get_event_service(state, request) as event_service,
        ):
            service = PublicEventServiceImpl(
                public_conversation_service=public_conversation_service,
                event_service=event_service,
            )
            yield service