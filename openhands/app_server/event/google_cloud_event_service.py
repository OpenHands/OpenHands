"""Google Cloud Storage-based EventService implementation."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client

from openhands.agent_server.models import EventPage, EventSortOrder
from more_itertools import bucket
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.errors import OpenHandsError
from openhands.app_server.event.event_service import EventService, EventServiceInjector
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.services.injector import InjectorState
from openhands.sdk import Event

from pydantic import Field

_logger = logging.getLogger(__name__)


@dataclass
class GoogleCloudEventService(EventService):
    """Google Cloud Storage-based implementation of EventService.

    Events are stored in GCS with the naming format:
    users/{user_id}/v1_conversations/{conversation_id}/{YYYYMMDDHHMMSS}_{kind}_{id.hex}

    Uses an AppConversationInfoService to lookup conversations and get user_id.
    """

    app_conversation_info_service: AppConversationInfoService
    bucket: Bucket
    _conversation_user_cache: dict[UUID, str | None] = field(
        default_factory=dict, init=False
    )

    async def _get_user_id_for_conversation(self, conversation_id: UUID) -> str | None:
        """Get the user_id for a conversation, using cache for efficiency."""
        if conversation_id in self._conversation_user_cache:
            return self._conversation_user_cache[conversation_id]

        conversation = (
            await self.app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
        )
        if conversation:
            user_id = conversation.created_by_user_id
            self._conversation_user_cache[conversation_id] = user_id
            return user_id
        return None

    def _get_event_path(self, user_id: str, conversation_id: UUID, event: Event) -> str:
        """Generate the full GCS path for an event."""
        filename = self._get_event_filename(event)
        return f'users/{user_id}/v1_conversations/{conversation_id}/{filename}'

    def _get_event_filename(self, event: Event) -> str:
        """Generate filename using YYYYMMDDHHMMSS_kind_id.hex format."""
        timestamp_str = self._timestamp_to_str(event.timestamp)
        kind = event.__class__.__name__
        if isinstance(event.id, str):
            id_hex = event.id.replace('-', '')
        else:
            id_hex = event.id.hex
        return f'{timestamp_str}_{kind}_{id_hex}'

    def _timestamp_to_str(self, timestamp: datetime | str) -> str:
        """Convert timestamp to YYYYMMDDHHMMSS format."""
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y%m%d%H%M%S')
        return timestamp.strftime('%Y%m%d%H%M%S')

    def _get_conversation_prefix(self, user_id: str, conversation_id: UUID) -> str:
        """Get the GCS prefix for a conversation's events."""
        return f'users/{user_id}/v1_conversations/{conversation_id}/'

    def _write_event(self, path: str, event: Event) -> None:
        """Write an event to GCS."""
        blob: Blob = self.bucket.blob(path)
        data = event.model_dump(mode='json')
        with blob.open('w') as f:
            f.write(json.dumps(data, indent=2))

    def _read_event(self, path: str) -> Event | None:
        """Read an event from GCS."""
        blob: Blob = self.bucket.blob(path)
        try:
            with blob.open('r') as f:
                json_data = f.read()
            return Event.model_validate_json(json_data)
        except NotFound:
            return None
        except Exception:
            _logger.exception(f'Error reading event from {path}')
            return None

    def _list_event_blobs(self, prefix: str) -> list[Blob]:
        """List all event blobs under a prefix."""
        return list(self.bucket.list_blobs(prefix=prefix))

    def _parse_filename(self, filename: str) -> dict[str, str] | None:
        """Parse filename to extract timestamp, kind, and event_id."""
        try:
            parts = filename.split('_')
            if len(parts) >= 3:
                timestamp_str = parts[0]
                kind = '_'.join(parts[1:-1])
                event_id = parts[-1]
                return {'timestamp': timestamp_str, 'kind': kind, 'event_id': event_id}
        except Exception:
            pass
        return None

    def _get_filename_from_blob(self, blob: Blob) -> str:
        """Extract the filename from a blob's name."""
        return blob.name.split('/')[-1]

    def _get_conversation_id_from_blob(self, blob: Blob) -> UUID | None:
        """Extract conversation_id from blob path."""
        try:
            parts = blob.name.split('/')
            for i, part in enumerate(parts):
                if part == 'v1_conversations' and i + 1 < len(parts):
                    return UUID(parts[i + 1])
        except Exception:
            pass
        return None

    def _filter_blobs_by_criteria(
        self,
        blobs: list[Blob],
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
    ) -> list[Blob]:
        """Filter blobs based on search criteria."""
        filtered = []
        for blob in blobs:
            filename = self._get_filename_from_blob(blob)
            filename_info = self._parse_filename(filename)
            if not filename_info:
                continue

            if kind__eq and filename_info['kind'] != kind__eq:
                continue

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

            filtered.append(blob)
        return filtered

    async def get_event(self, event_id: str) -> Event | None:
        """Get the event with the given id, or None if not found."""
        if isinstance(event_id, str) and '-' in event_id:
            id_hex = event_id.replace('-', '')
        else:
            id_hex = event_id

        loop = asyncio.get_running_loop()

        def find_and_load_event() -> Event | None:
            for blob in self.bucket.list_blobs():
                filename = self._get_filename_from_blob(blob)
                if filename.endswith(f'_{id_hex}'):
                    return self._read_event(blob.name)
            return None

        event = await loop.run_in_executor(None, find_and_load_event)
        if event is None:
            return None

        # Verify conversation access
        conversation_id = (
            event.conversation_id if hasattr(event, 'conversation_id') else None
        )
        if conversation_id:
            conversation = (
                await self.app_conversation_info_service.get_app_conversation_info(
                    conversation_id
                    if isinstance(conversation_id, UUID)
                    else UUID(conversation_id)
                )
            )
            if not conversation:
                return None

        return event

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
        if conversation_id__eq:
            user_id = await self._get_user_id_for_conversation(conversation_id__eq)
            if not user_id:
                return EventPage(items=[], next_page_id=None)
            prefix = self._get_conversation_prefix(user_id, conversation_id__eq)
        else:
            # Search across all events - need to search conversations user has access to
            # Get all conversations the user has access to and aggregate results
            return await self._search_events_across_conversations(
                kind__eq, timestamp__gte, timestamp__lt, sort_order, page_id, limit
            )

        loop = asyncio.get_running_loop()

        def list_and_filter() -> list[Blob]:
            blobs = self._list_event_blobs(prefix)
            return self._filter_blobs_by_criteria(
                blobs, kind__eq, timestamp__gte, timestamp__lt
            )

        filtered_blobs = await loop.run_in_executor(None, list_and_filter)

        # Sort by filename (which contains timestamp)
        filtered_blobs.sort(
            key=lambda b: self._get_filename_from_blob(b),
            reverse=(sort_order == EventSortOrder.TIMESTAMP_DESC),
        )

        # Handle pagination
        start_index = 0
        if page_id:
            for i, blob in enumerate(filtered_blobs):
                if self._get_filename_from_blob(blob) == page_id:
                    start_index = i + 1
                    break

        page_blobs = filtered_blobs[start_index : start_index + limit]
        next_page_id = None
        if start_index + limit < len(filtered_blobs):
            next_page_id = self._get_filename_from_blob(
                filtered_blobs[start_index + limit]
            )

        def load_events() -> list[Event]:
            events = []
            for blob in page_blobs:
                event = self._read_event(blob.name)
                if event:
                    events.append(event)
            return events

        page_events = await loop.run_in_executor(None, load_events)
        return EventPage(items=page_events, next_page_id=next_page_id)

    async def _search_events_across_conversations(
        self,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
        sort_order: EventSortOrder = EventSortOrder.TIMESTAMP,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        """Search events across all accessible conversations."""
        conversations_page = (
            await self.app_conversation_info_service.search_app_conversation_info(
                limit=1000
            )
        )
        all_blobs: list[tuple[Blob, UUID]] = []

        loop = asyncio.get_running_loop()

        for conversation in conversations_page.items:
            if not conversation.created_by_user_id:
                continue
            prefix = self._get_conversation_prefix(
                conversation.created_by_user_id, conversation.id
            )

            def list_blobs(p: str = prefix) -> list[Blob]:
                return self._list_event_blobs(p)

            blobs = await loop.run_in_executor(None, list_blobs)
            blobs = self._filter_blobs_by_criteria(
                blobs, kind__eq, timestamp__gte, timestamp__lt
            )
            all_blobs.extend((blob, conversation.id) for blob in blobs)

        # Sort all blobs by filename
        all_blobs.sort(
            key=lambda x: self._get_filename_from_blob(x[0]),
            reverse=(sort_order == EventSortOrder.TIMESTAMP_DESC),
        )

        # Handle pagination
        start_index = 0
        if page_id:
            for i, (blob, _) in enumerate(all_blobs):
                if self._get_filename_from_blob(blob) == page_id:
                    start_index = i + 1
                    break

        page_items = all_blobs[start_index : start_index + limit]
        next_page_id = None
        if start_index + limit < len(all_blobs):
            next_page_id = self._get_filename_from_blob(
                all_blobs[start_index + limit][0]
            )

        def load_events() -> list[Event]:
            events = []
            for blob, _ in page_items:
                event = self._read_event(blob.name)
                if event:
                    events.append(event)
            return events

        page_events = await loop.run_in_executor(None, load_events)
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
        if conversation_id__eq:
            user_id = await self._get_user_id_for_conversation(conversation_id__eq)
            if not user_id:
                return 0
            prefix = self._get_conversation_prefix(user_id, conversation_id__eq)
        else:
            # Count across all conversations
            return await self._count_events_across_conversations(
                kind__eq, timestamp__gte, timestamp__lt
            )

        loop = asyncio.get_running_loop()

        def list_and_filter() -> int:
            blobs = self._list_event_blobs(prefix)
            filtered = self._filter_blobs_by_criteria(
                blobs, kind__eq, timestamp__gte, timestamp__lt
            )
            return len(filtered)

        return await loop.run_in_executor(None, list_and_filter)

    async def _count_events_across_conversations(
        self,
        kind__eq: EventKind | None = None,
        timestamp__gte: datetime | None = None,
        timestamp__lt: datetime | None = None,
    ) -> int:
        """Count events across all accessible conversations."""
        conversations_page = (
            await self.app_conversation_info_service.search_app_conversation_info(
                limit=1000
            )
        )
        total = 0

        loop = asyncio.get_running_loop()

        for conversation in conversations_page.items:
            if not conversation.created_by_user_id:
                continue
            prefix = self._get_conversation_prefix(
                conversation.created_by_user_id, conversation.id
            )

            def count_blobs(p: str = prefix) -> int:
                blobs = self._list_event_blobs(p)
                filtered = self._filter_blobs_by_criteria(
                    blobs, kind__eq, timestamp__gte, timestamp__lt
                )
                return len(filtered)

            total += await loop.run_in_executor(None, count_blobs)

        return total

    async def save_event(self, conversation_id: UUID, event: Event):
        """Save an event. Internal method intended not be part of the REST api."""
        conversation = (
            await self.app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
        )
        if not conversation:
            raise OpenHandsError(f'No such conversation: {conversation_id}')

        user_id = conversation.created_by_user_id
        if not user_id:
            raise OpenHandsError(
                f'Conversation {conversation_id} has no associated user'
            )

        path = self._get_event_path(user_id, conversation_id, event)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_event, path, event)


class GoogleCloudEventServiceInjector(EventServiceInjector):
    bucket_name: str

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[EventService, None]:
        from openhands.app_server.config import get_app_conversation_info_service

        async with get_app_conversation_info_service(
            state, request
        ) as app_conversation_info_service:
            bucket_name = self.bucket_name
            storage_client: Client = storage.Client()
            bucket: Bucket = storage_client.bucket(bucket_name)

            yield GoogleCloudEventService(
                app_conversation_info_service=app_conversation_info_service,
                bucket=bucket,
            )
