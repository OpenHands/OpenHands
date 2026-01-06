"""Google Cloud Storage-based EventService implementation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Iterator

from fastapi import Request
from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client

from openhands.app_server.config import get_app_conversation_info_service
from openhands.app_server.event.event_service import EventService, EventServiceInjector
from openhands.app_server.event.event_service_base import EventServiceBase
from openhands.app_server.services.injector import InjectorState
from openhands.sdk import Event

_logger = logging.getLogger(__name__)


@dataclass
class GoogleCloudEventService(EventServiceBase):
    """Google Cloud Storage-based implementation of EventService."""

    bucket: Bucket

    def _load_event(self, path: Path) -> Event | None:
        """Get the event at the path given."""
        blob: Blob = self.bucket.blob(str(path))
        try:
            with blob.open('r') as f:
                json_data = f.read()
            event = Event.model_validate_json(json_data)
            return event
        except NotFound:
            return None
        except Exception:
            _logger.exception(f'Error reading event from {path}')
            return None

    def _store_event(self, path: Path, event: Event):
        """Store the event given at the path given."""
        blob: Blob = self.bucket.blob(str(path))
        data = event.model_dump(mode='json')
        with blob.open('w') as f:
            f.write(json.dumps(data, indent=2))

    def _search_paths(self, prefix: Path, page_id: str | None = None) -> list[Path]:
        """Search paths."""
        blobs: Iterator[Blob] = self.bucket.list_blobs(
            page_token=page_id, prefix=str(prefix)
        )
        paths = list(Path(blob.name) for blob in blobs)
        return paths


class GoogleCloudEventServiceInjector(EventServiceInjector):
    bucket_name: str
    prefix: Path = Path('users')

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[EventService, None]:
        from openhands.app_server.config import (
            get_user_context,
        )

        async with (
            get_user_context(state, request) as user_context,
            get_app_conversation_info_service(
                state, request
            ) as app_conversation_info_service,
        ):
            user_id = await user_context.get_user_id()

            bucket_name = self.bucket_name
            storage_client: Client = storage.Client()
            bucket: Bucket = storage_client.bucket(bucket_name)

            yield GoogleCloudEventService(
                prefix=self.prefix,
                user_id=user_id,
                app_conversation_info_service=app_conversation_info_service,
                bucket=bucket,
                app_conversation_info_load_tasks={},
            )
