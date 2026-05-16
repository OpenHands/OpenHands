import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.conversation_paths import V1_CONVERSATIONS_DIR
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.sdk import Event
from openhands.sdk.utils.paging import page_iterator


@dataclass
class EventSearchMetadata:
    """Lightweight event data needed to filter, sort, and page search results."""

    path: Path
    kind: str
    timestamp: str
    event: Event | None = None


@dataclass
class EventServiceBase(EventService, ABC):
    """Event Service for getting events - the only check on permissions for events is
    in the strict prefix for storage.
    """

    prefix: Path
    user_id: str | None
    app_conversation_info_service: AppConversationInfoService | None
    app_conversation_info_load_tasks: dict[
        UUID, asyncio.Task[AppConversationInfo | None]
    ]

    @abstractmethod
    def _load_event(self, path: Path) -> Event | None:
        """Get the event at the path given."""

    @abstractmethod
    def _store_event(self, path: Path, event: Event):
        """Store the event given at the path given."""

    @abstractmethod
    def _search_paths(self, prefix: Path) -> list[Path]:
        """Search paths."""

    def _load_event_search_metadata(self, path: Path) -> EventSearchMetadata | None:
        """Load lightweight metadata for event search.

        Storage backends may override this to avoid hydrating full event models
        for files that will be filtered out or paginated away.
        """
        event = self._load_event(path)
        if not event:
            return None
        return EventSearchMetadata(
            path=path,
            kind=event.kind,
            timestamp=event.timestamp,
            event=event,
        )

    async def get_conversation_path(self, conversation_id: UUID) -> Path:
        """Get a path for a conversation. Ensure user_id is included if possible."""
        path = self.prefix
        if self.user_id:
            path /= self.user_id
        elif self.app_conversation_info_service:
            task = self.app_conversation_info_load_tasks.get(conversation_id)
            if task is None:
                task = asyncio.create_task(
                    self.app_conversation_info_service.get_app_conversation_info(
                        conversation_id
                    )
                )
                self.app_conversation_info_load_tasks[conversation_id] = task
            conversation_info = await task
            if conversation_info and conversation_info.created_by_user_id:
                path /= conversation_info.created_by_user_id
        path = path / V1_CONVERSATIONS_DIR / conversation_id.hex
        return path

    async def get_event(self, conversation_id: UUID, event_id: UUID) -> Event | None:
        """Get the event with the given id, or None if not found."""
        conversation_path = await self.get_conversation_path(conversation_id)
        path = conversation_path / f'{event_id.hex}.json'
        loop = asyncio.get_running_loop()
        event: Event = await loop.run_in_executor(None, self._load_event, path)  # type: ignore[arg-type]
        return event

    async def search_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        """Search events matching the given filters."""
        loop = asyncio.get_running_loop()
        prefix = await self.get_conversation_path(conversation_id)
        paths = await loop.run_in_executor(None, self._search_paths, prefix)

        metadata_results = await asyncio.gather(
            *[
                loop.run_in_executor(None, self._load_event_search_metadata, path)
                for path in paths
            ]
        )
        # Convert datetime filters to ISO strings so they can be compared
        # against event.timestamp (which is stored as an ISO 8601 string).
        timestamp_gte_str = timestamp__gte.isoformat() if timestamp__gte else None
        timestamp_lt_str = timestamp__lt.isoformat() if timestamp__lt else None

        matching_metadata: list[EventSearchMetadata] = []
        for metadata in metadata_results:
            if not metadata:
                continue
            if kind__eq and metadata.kind != kind__eq:
                continue
            if timestamp_gte_str and metadata.timestamp < timestamp_gte_str:
                continue
            if timestamp_lt_str and metadata.timestamp >= timestamp_lt_str:
                continue
            matching_metadata.append(metadata)

        if sort_order:
            matching_metadata.sort(
                key=lambda metadata: metadata.timestamp,
                reverse=(sort_order == EventSortOrder.TIMESTAMP_DESC),
            )

        # Apply pagination before hydrating full Event models for the page.
        start_offset = 0
        next_page_id = None
        if page_id:
            start_offset = int(page_id)
            matching_metadata = matching_metadata[start_offset:]
        if len(matching_metadata) > limit:
            next_page_id = str(start_offset + limit)
            matching_metadata = matching_metadata[:limit]

        items = []
        for metadata in matching_metadata:
            if metadata.event:
                items.append(metadata.event)
                continue
            event = await loop.run_in_executor(None, self._load_event, metadata.path)  # type: ignore[arg-type]
            if event:
                items.append(event)

        return EventPage(items=items, next_page_id=next_page_id)

    async def count_events(
        self,
        conversation_id: UUID,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
    ) -> int:
        """Count events matching the given filters."""
        # If we are not filtering, we can simply count the paths
        if not (kind__eq or timestamp__gte or timestamp__lt):
            conversation_path = await self.get_conversation_path(conversation_id)
            result = await self._count_events_no_filter(conversation_path)
            return result

        events = page_iterator(
            self.search_events,
            conversation_id=conversation_id,
            kind__eq=kind__eq,
            timestamp__gte=timestamp__gte,
            timestamp__lt=timestamp__lt,
        )
        result = 0
        async for event in events:
            result += 1
        return result

    async def _count_events_no_filter(self, conversation_path: Path) -> int:
        """Count all event files in the conversation directory without filtering."""
        loop = asyncio.get_running_loop()
        paths = await loop.run_in_executor(None, self._search_paths, conversation_path)
        return len(paths)

    async def save_event(self, conversation_id: UUID, event: Event):
        if isinstance(event.id, str):
            id_hex = event.id.replace('-', '')
        else:
            id_hex = event.id.hex  # type: ignore[unreachable]
        path = (await self.get_conversation_path(conversation_id)) / f'{id_hex}.json'
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._store_event, path, event)

    async def batch_get_events(
        self, conversation_id: UUID, event_ids: list[UUID]
    ) -> list[Event | None]:
        """Given a list of ids, get events (Or none for any which were not found)."""
        return await asyncio.gather(
            *[self.get_event(conversation_id, event_id) for event_id in event_ids]
        )
