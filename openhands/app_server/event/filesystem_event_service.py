import asyncio
import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Request

from openhands.app_server.event.event_service import EventService, EventServiceInjector
from openhands.app_server.event.event_service_base import EventServiceBase
from openhands.app_server.services.injector import InjectorState
from openhands.sdk import Event

_logger = logging.getLogger(__name__)


@dataclass
class FilesystemEventService(EventServiceBase):
    """Event service based on file system"""

    limit: int = 500

    def _load_event(self, path: Path) -> Event | None:
        try:
            content = path.read_text()
            content = Event.model_validate_json(content)  # type: ignore[assignment]
            return content  # type: ignore[return-value]
        except Exception:
            if path.exists():
                _logger.exception('Error reading event', stack_info=True)
            return None

    def _store_event(self, path: Path, event: Event):
        path.parent.mkdir(parents=True, exist_ok=True)
        content = event.model_dump_json(indent=2)
        path.write_text(content)

    def _search_paths(self, prefix: Path, page_id: str | None = None) -> list[Path]:
        search_path = f'{prefix}/*'
        files = glob.glob(str(search_path))
        paths = [Path(file) for file in files]
        return paths

    async def _scan_event_timestamps(
        self, paths: list[Path]
    ) -> list[tuple[Path, str]] | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_event_timestamps, paths)

    def _read_event_timestamps(self, paths: list[Path]) -> list[tuple[Path, str]]:
        """Read only the timestamp of each stored event.

        A plain ``json.loads`` of the stored payload is an order of magnitude
        cheaper than validating the full ``Event`` discriminated union, which
        bounds unfiltered ``search_events`` validation to the requested page's
        ordered prefix instead of every stored event. Reads happen in a single
        executor call because the per-file work is too small to amortize one
        task per file.
        """
        scanned = []
        for path in paths:
            try:
                timestamp = json.loads(path.read_text())['timestamp']
            except Exception:
                if path.exists():
                    _logger.exception('Error reading event', stack_info=True)
                continue
            if isinstance(timestamp, str):
                scanned.append((path, timestamp))
        return scanned


class FilesystemEventServiceInjector(EventServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[EventService, None]:
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_global_config,
            get_user_context,
        )

        async with (
            get_user_context(state, request) as user_context,
            get_app_conversation_info_service(
                state, request
            ) as app_conversation_info_service,
        ):
            # Set up a service with a path {persistence_dir}/{user_id}/v1_conversations
            prefix = get_global_config().persistence_dir
            user_id = await user_context.get_user_id()

            yield FilesystemEventService(
                prefix=prefix,
                user_id=user_id,
                app_conversation_info_service=app_conversation_info_service,
                app_conversation_info_load_tasks={},
            )
