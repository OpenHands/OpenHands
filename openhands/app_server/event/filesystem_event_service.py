"""Filesystem-based EventService implementation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request

from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.errors import OpenHandsError
from openhands.app_server.event.event_service import EventService, EventServiceInjector
from openhands.app_server.event.filesystem_event_service_base import (
    FilesystemEventServiceBase,
)
from openhands.app_server.services.injector import InjectorState
from openhands.sdk import Event


@dataclass
class FilesystemEventService(FilesystemEventServiceBase, EventService):
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

    def _save_event_to_file(self, conversation_id: UUID, event: Event) -> None:
        """Save an event to a file."""
        events_path = self._ensure_events_dir(conversation_id)
        filename = self._get_event_filename(conversation_id, event)
        filepath = events_path / filename

        with open(filepath, 'w') as f:
            # Use model_dump with mode='json' to handle UUID serialization
            data = event.model_dump(mode='json')
            f.write(json.dumps(data, indent=2))

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
