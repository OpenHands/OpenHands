"""Filesystem-based EventService implementation."""

import asyncio
import glob
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.errors import OpenHandsError
from openhands.app_server.event.event_service import EventService, EventServiceInjector
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.services.injector import InjectorState
from openhands.sdk import Event

_logger = logging.getLogger(__name__)


@dataclass
class FilesystemEventService(EventService):
    """Filesystem-based implementation of EventService.

    Events are stored in files with the naming format:
    {conversation_id}/{YYYYMMDDHHMMSS}_{kind}_{id.hex}

    Uses an AppConversationInfoService to lookup conversations
    """

    app_conversation_info_service: AppConversationInfoService
    events_dir: Path

    def _ensure_events_dir(self, conversation_id: UUID | None = None) -> Path:
        """Ensure the events directory exists."""
        if conversation_id:
            events_path = self.events_dir / str(conversation_id)
        else:
            events_path = self.events_dir
        events_path.mkdir(parents=True, exist_ok=True)
        return events_path

    def _timestamp_to_str(self, timestamp: datetime | str) -> str:
        """Convert timestamp to YYYYMMDDHHMMSS format."""
        if isinstance(timestamp, str):
            # Parse ISO format timestamp string
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y%m%d%H%M%S')
        return timestamp.strftime('%Y%m%d%H%M%S')

    def _get_event_filename(self, conversation_id: UUID, event: Event) -> str:
        """Generate filename using YYYYMMDDHHMMSS_kind_id.hex format."""
        timestamp_str = self._timestamp_to_str(event.timestamp)
        kind = event.__class__.__name__
        # Handle both UUID objects and string UUIDs
        if isinstance(event.id, str):
            id_hex = event.id.replace('-', '')
        else:
            id_hex = event.id.hex
        return f'{timestamp_str}_{kind}_{id_hex}'

    def _save_event_to_file(self, conversation_id: UUID, event: Event) -> None:
        """Save an event to a file."""
        events_path = self._ensure_events_dir(conversation_id)
        filename = self._get_event_filename(conversation_id, event)
        filepath = events_path / filename

        with open(filepath, 'w') as f:
            # Use model_dump with mode='json' to handle UUID serialization
            data = event.model_dump(mode='json')
            f.write(json.dumps(data, indent=2))

    def _load_events_from_files(self, file_paths: list[Path]) -> list[Event]:
        events = []
        for file_path in file_paths:
            event = self._load_event_from_file(file_path)
            if event is not None:
                events.append(event)
        return events

    def _load_event_from_file(self, filepath: Path) -> Event | None:
        """Load an event from a file."""
        try:
            json_data = filepath.read_text()
            return Event.model_validate_json(json_data)
        except Exception:
            return None

    def _get_event_files_by_pattern(
        self, pattern: str, conversation_id: UUID | None = None
    ) -> list[Path]:
        """Get event files matching a glob pattern, sorted by timestamp."""
        if conversation_id:
            search_path = self.events_dir / str(conversation_id) / pattern
        else:
            search_path = self.events_dir / '*' / pattern

        files = glob.glob(str(search_path))
        return sorted([Path(f) for f in files])

    def _parse_filename(self, filename: str) -> dict[str, str] | None:
        """Parse filename to extract timestamp, kind, and event_id."""
        try:
            parts = filename.split('_')
            if len(parts) >= 3:
                timestamp_str = parts[0]
                kind = '_'.join(parts[1:-1])  # Handle kinds with underscores
                event_id = parts[-1]
                return {'timestamp': timestamp_str, 'kind': kind, 'event_id': event_id}
        except Exception:
            pass
        return None

    def _get_conversation_id(self, file: Path) -> UUID | None:
        try:
            return UUID(file.parent.name)
        except Exception:
            return None

    def _get_conversation_ids(self, files: list[Path]) -> set[UUID]:
        result = set()
        for file in files:
            conversation_id = self._get_conversation_id(file)
            if conversation_id:
                result.add(conversation_id)
        return result

    async def _filter_files_by_conversation(self, files: list[Path]) -> list[Path]:
        conversation_ids = list(self._get_conversation_ids(files))
        conversations = (
            await self.app_conversation_info_service.batch_get_app_conversation_info(
                conversation_ids
            )
        )
        permitted_conversation_ids = set()
        for conversation in conversations:
            if conversation:
                permitted_conversation_ids.add(conversation.id)
        result = [
            file
            for file in files
            if self._get_conversation_id(file) in permitted_conversation_ids
        ]
        return result

    def _filter_files_by_criteria(
        self,
        files: list[Path],
        conversation_id__eq: UUID | None = None,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
    ) -> list[Path]:
        """Filter files based on search criteria."""
        filtered_files = []

        for file_path in files:
            # Check conversation_id filter
            if conversation_id__eq:
                if str(conversation_id__eq) not in str(file_path):
                    continue

            # Parse filename for additional filtering
            filename_info = self._parse_filename(file_path.name)
            if not filename_info:
                continue

            # Check kind filter
            if kind__eq and filename_info['kind'] != kind__eq:
                continue

            # Check timestamp filters
            if timestamp__gte or timestamp__lt:
                try:
                    file_timestamp = datetime.strptime(
                        filename_info['timestamp'], '%Y%m%d%H%M%S'
                    )
                    if timestamp__gte and file_timestamp < timestamp__gte:
                        continue
                    if timestamp__lt and file_timestamp >= timestamp__lt:
                        continue
                except ValueError:
                    continue

            filtered_files.append(file_path)

        return filtered_files

    async def get_event(self, event_id: str) -> Event | None:
        """Get the event with the given id, or None if not found."""
        # Convert event_id to hex format (remove dashes) for filename matching
        if isinstance(event_id, str) and '-' in event_id:
            id_hex = event_id.replace('-', '')
        else:
            id_hex = event_id

        # Use glob pattern to find files ending with the event_id
        pattern = f'*_{id_hex}'
        files = self._get_event_files_by_pattern(pattern)

        if not files:
            return None

        # If there is no access to the conversation do not return the event
        file = files[0]
        conversation_id = self._get_conversation_id(file)
        if not conversation_id:
            return None
        conversation = (
            await self.app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
        )
        if not conversation:
            return None

        # Load and return the first matching event
        return self._load_event_from_file(file)

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
        """Search for events matching the given filters."""
        # Build the search pattern
        pattern = '*'
        files = self._get_event_files_by_pattern(pattern, conversation_id__eq)

        files = await self._filter_files_by_conversation(files)

        files = self._filter_files_by_criteria(
            files, conversation_id__eq, kind__eq, timestamp__gte, timestamp__lt
        )

        files.sort(
            key=lambda f: f.name,
            reverse=(sort_order == EventSortOrder.TIMESTAMP_DESC),
        )

        # Handle pagination
        start_index = 0
        if page_id:
            for i, file_path in enumerate(files):
                if file_path.name == page_id:
                    start_index = i + 1
                    break

        # Collect items for this page
        page_files = files[start_index : start_index + limit]
        next_page_id = None
        if start_index + limit < len(files):
            next_page_id = files[start_index + limit].name

        # Load all events from files in a background thread.
        loop = asyncio.get_running_loop()
        page_events = await loop.run_in_executor(
            None, self._load_events_from_files, page_files
        )

        return EventPage(items=page_events, next_page_id=next_page_id)

    async def count_events(
        self,
        conversation_id__eq: UUID | None = None,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
    ) -> int:
        """Count events matching the given filters."""
        # Build the search pattern
        pattern = '*'
        files = self._get_event_files_by_pattern(pattern, conversation_id__eq)

        files = await self._filter_files_by_conversation(files)

        files = self._filter_files_by_criteria(
            files, conversation_id__eq, kind__eq, timestamp__gte, timestamp__lt
        )

        return len(files)

    async def save_event(self, conversation_id: UUID, event: Event):
        """Save an event. Internal method intended not be part of the REST api."""
        conversation = (
            await self.app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
        )
        if not conversation:
            # This is either an illegal state or somebody is trying to hack
            raise OpenHandsError('No such conversation: {conversaiont_id}')
        self._save_event_to_file(conversation_id, event)


class FilesystemEventServiceInjector(EventServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[EventService, None]:
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_global_config,
        )

        async with get_app_conversation_info_service(
            state, request
        ) as app_conversation_info_service:
            persistence_dir = get_global_config().persistence_dir

            yield FilesystemEventService(
                app_conversation_info_service=app_conversation_info_service,
                events_dir=persistence_dir / 'v1' / 'events',
            )
