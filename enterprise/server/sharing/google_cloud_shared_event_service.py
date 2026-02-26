"""Implementation of SharedEventService.

This implementation provides read-only access to events from shared conversations:
- Validates that the conversation is shared before returning events
- Uses existing EventService for actual event retrieval
- Uses SharedConversationInfoService for shared conversation validation
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client
from pydantic import Field
from server.sharing.shared_conversation_info_service import (
    SharedConversationInfoService,
)
from server.sharing.shared_event_service import (
    SharedEventService,
    SharedEventServiceInjector,
)
from server.sharing.sql_shared_conversation_info_service import (
    SQLSharedConversationInfoService,
)

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event.google_cloud_event_service import (
    GoogleCloudEventService,
)
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.services.injector import InjectorState
from openhands.sdk import Event

logger = logging.getLogger(__name__)


@dataclass
class GoogleCloudSharedEventService(SharedEventService):
    """Implementation of SharedEventService that validates shared access."""

    shared_conversation_info_service: SharedConversationInfoService
    bucket: Bucket

    async def get_event_service(self, conversation_id: UUID) -> EventService | None:
        shared_conversation_info = (
            await self.shared_conversation_info_service.get_shared_conversation_info(
                conversation_id
            )
        )
        if shared_conversation_info is None:
            return None

        return GoogleCloudEventService(
            bucket=self.bucket,
            prefix=Path('users'),
            user_id=shared_conversation_info.created_by_user_id,
            app_conversation_info_service=None,
            app_conversation_info_load_tasks={},
        )

    async def get_shared_event(
        self, conversation_id: UUID, event_id: UUID
    ) -> Event | None:
        """Given a conversation_id and event_id, retrieve an event if the conversation is shared."""
        # First check if the conversation is shared
        event_service = await self.get_event_service(conversation_id)
        if event_service is None:
            return None

        # If conversation is shared, get the event
        return await event_service.get_event(conversation_id, event_id)

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
        # First check if the conversation is shared
        event_service = await self.get_event_service(conversation_id)
        if event_service is None:
            # Return empty page if conversation is not shared
            return EventPage(items=[], next_page_id=None)

        # If conversation is shared, search events for this conversation
        return await event_service.search_events(
            conversation_id=conversation_id,
            kind__eq=kind__eq,
            timestamp__gte=timestamp__gte,
            timestamp__lt=timestamp__lt,
            sort_order=sort_order,
            page_id=page_id,
            limit=limit,
        )

    async def count_shared_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
    ) -> int:
        """Count events for a specific shared conversation."""
        # First check if the conversation is shared
        event_service = await self.get_event_service(conversation_id)
        if event_service is None:
            # Return empty page if conversation is not shared
            return 0

        # If conversation is shared, count events for this conversation
        return await event_service.count_events(
            conversation_id=conversation_id,
            kind__eq=kind__eq,
            timestamp__gte=timestamp__gte,
            timestamp__lt=timestamp__lt,
        )


class GoogleCloudSharedEventServiceInjector(SharedEventServiceInjector):
    bucket_name: str | None = Field(
        default_factory=lambda: os.environ.get('FILE_STORE_PATH')
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SharedEventService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import get_db_session

        async with get_db_session(state, request) as db_session:
            shared_conversation_info_service = SQLSharedConversationInfoService(
                db_session=db_session
            )

            bucket_name = self.bucket_name
            storage_client: Client = storage.Client()
            bucket: Bucket = storage_client.bucket(bucket_name)

            service = GoogleCloudSharedEventService(
                shared_conversation_info_service=shared_conversation_info_service,
                bucket=bucket,
            )
            yield service
