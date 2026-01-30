import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.agent_server.sockets import page_iterator
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.sdk import Event


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
        path = path / 'v1_conversations' / conversation_id.hex
        return path

    async def get_event(self, conversation_id: UUID, event_id: UUID) -> Event | None:
        """Get the event with the given id, or None if not found."""
        conversation_path = await self.get_conversation_path(conversation_id)
        path = conversation_path / f'{event_id.hex}.json'
        loop = asyncio.get_running_loop()
        event: Event = await loop.run_in_executor(None, self._load_event, path)
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
        events = await asyncio.gather(
            *[loop.run_in_executor(None, self._load_event, path) for path in paths]
        )
        items = []
        for event in events:
            if not event:
                continue
            if kind__eq and event.kind != kind__eq:
                continue
            if timestamp__gte and event.timestamp < timestamp__gte:
                continue
            if timestamp__lt and event.timestamp >= timestamp__lt:
                continue
            items.append(event)

        if sort_order:
            items.sort(
                key=lambda e: e.timestamp,
                reverse=(sort_order == EventSortOrder.TIMESTAMP_DESC),
            )

        start_offset = 0
        next_page_id = None
        if page_id:
            start_offset = int(page_id)
            paths = paths[start_offset:]
        if len(paths) > limit:
            paths = paths[:limit]
            next_page_id = str(start_offset + limit)

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
        result = sum(1 for event in events)
        return result

    async def _count_events_no_filter(self, conversation_path: Path) -> int:
        paths = page_iterator(self._search_paths, conversation_path)
        result = 0
        async for _ in paths:
            result += 1
        return result

    async def save_event(self, conversation_id: UUID, event: Event):
        if isinstance(event.id, str):
            id_hex = event.id.replace('-', '')
        else:
            id_hex = event.id.hex
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
