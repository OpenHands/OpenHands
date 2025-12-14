"""Filesystem-based PublicEventService implementation."""

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
from openhands.app_server.errors import OpenHandsError
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.public_conversations.public_conversation_info_service import (
    PublicConversationInfoService,
)
from openhands.app_server.public_conversations.public_event_service import (
    PublicEventService,
)
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.sdk import Event

_logger = logging.getLogger(__name__)


@dataclass
class FilesystemPublicEventService(PublicEventService):
    """Filesystem-based implementation of PublicEventService.

    Events are stored in files with the naming format:
    {conversation_id}/{YYYYMMDDHHMMSS}_{kind}_{id.hex}

    Uses a PublicConversationInfoService to lookup public conversations
    """

    public_conversation_info_service: PublicConversationInfoService
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
        """Convert timestamp to string format used in filenames."""
        if isinstance(timestamp, str):
            # Try to parse and reformat
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime('%Y%m%d%H%M%S')
            except ValueError:
                return timestamp
        return timestamp.strftime('%Y%m%d%H%M%S')

    def _find_event_files(
        self,
        pattern: str = '*',
        conversation_id: UUID | None = None,
    ) -> list[Path]:
        """Find event files matching the pattern."""
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

    async def _filter_files_by_public_conversation(self, files: list[Path]) -> list[Path]:
        """Filter files to only include those from public conversations."""
        conversation_ids = list(self._get_conversation_ids(files))
        
        # Check which conversations are public
        permitted_conversation_ids = set()
        for conversation_id in conversation_ids:
            conversation = await self.public_conversation_info_service.get_public_conversation_info(
                conversation_id
            )
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

    async def get_public_event(self, event_id: str) -> Event | None:
        """Get the event with the given id from a public conversation, or None if not found."""
        # Convert event_id to hex format (remove dashes) for filename matching
        if isinstance(event_id, str) and '-' in event_id:
            id_hex = event_id.replace('-', '')
        else:
            id_hex = str(event_id)

        # Search for files ending with this event ID
        files = self._find_event_files(pattern=f'*_{id_hex}')
        
        # Filter to only public conversations
        files = await self._filter_files_by_public_conversation(files)

        if not files:
            return None

        # Load the first matching file
        try:
            with open(files[0], 'r') as f:
                event_data = json.load(f)
                return Event.model_validate(event_data)
        except Exception as e:
            _logger.error(f'Error loading event from {files[0]}: {e}')
            return None

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
        # Find all event files
        files = self._find_event_files(conversation_id=conversation_id__eq)
        
        # Filter to only public conversations
        files = await self._filter_files_by_public_conversation(files)
        
        # Apply search criteria
        files = self._filter_files_by_criteria(
            files, conversation_id__eq, kind__eq, timestamp__gte, timestamp__lt
        )

        # Apply sorting
        if sort_order == EventSortOrder.TIMESTAMP_DESC:
            files = sorted(files, reverse=True)
        else:
            files = sorted(files)

        # Apply pagination
        start_index = 0
        if page_id:
            try:
                start_index = int(page_id)
            except ValueError:
                start_index = 0

        end_index = start_index + limit
        page_files = files[start_index:end_index]

        # Load events from files
        events = []
        for file_path in page_files:
            try:
                with open(file_path, 'r') as f:
                    event_data = json.load(f)
                    event = Event.model_validate(event_data)
                    events.append(event)
            except Exception as e:
                _logger.error(f'Error loading event from {file_path}: {e}')
                continue

        # Determine next page
        next_page_id = None
        if end_index < len(files):
            next_page_id = str(end_index)

        return EventPage(items=events, next_page_id=next_page_id)

    async def count_public_events(
        self,
        conversation_id__eq: UUID | None = None,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
    ) -> int:
        """Count events from public conversations matching the given filters."""
        # Find all event files
        files = self._find_event_files(conversation_id=conversation_id__eq)
        
        # Filter to only public conversations
        files = await self._filter_files_by_public_conversation(files)
        
        # Apply search criteria
        files = self._filter_files_by_criteria(
            files, conversation_id__eq, kind__eq, timestamp__gte, timestamp__lt
        )

        return len(files)

    async def batch_get_public_events(self, event_ids: list[str]) -> list[Event | None]:
        """Get a batch of events from public conversations given their ids."""
        events = []
        for event_id in event_ids:
            event = await self.get_public_event(event_id)
            events.append(event)
        return events

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
        # First, get the conversation by token
        conversation = await self.public_conversation_info_service.get_public_conversation_info_by_token(
            token
        )
        if not conversation:
            return EventPage(items=[], next_page_id=None)

        # Then search events for that conversation
        return await self.search_public_events(
            conversation_id__eq=conversation.id,
            kind__eq=kind__eq,
            timestamp__gte=timestamp__gte,
            timestamp__lt=timestamp__lt,
            sort_order=sort_order,
            page_id=page_id,
            limit=limit,
        )


class FilesystemPublicEventServiceInjector:
    """Dependency injection for FilesystemPublicEventService."""

    async def __call__(self, request: Request) -> AsyncGenerator[FilesystemPublicEventService, None]:
        """Create and yield a FilesystemPublicEventService instance."""
        public_conversation_info_service = Injector.get(PublicConversationInfoService)
        events_dir = Path('/tmp/openhands_events')  # TODO: Make configurable
        yield FilesystemPublicEventService(
            public_conversation_info_service=public_conversation_info_service,
            events_dir=events_dir,
        )


def register_filesystem_public_event_service():
    """Register the filesystem implementation of PublicEventService."""
    Injector.register(
        PublicEventService,
        FilesystemPublicEventServiceInjector(),
        InjectorState.ASYNC_GENERATOR,
    )