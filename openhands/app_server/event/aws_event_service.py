"""AWS S3-based EventService implementation.

This implementation uses role-based authentication (no credentials needed).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

import boto3
import botocore.exceptions
from fastapi import Request
from pydantic import Field

from openhands.app_server.config import get_app_conversation_info_service
from openhands.app_server.event.event_service import EventService, EventServiceInjector
from openhands.app_server.event.event_service_base import EventServiceBase
from openhands.app_server.services.injector import InjectorState
from openhands.sdk import Event

_logger = logging.getLogger(__name__)


@dataclass
class AwsEventService(EventServiceBase):
    """AWS S3-based implementation of EventService.

    Uses role-based authentication, so no explicit credentials are needed.
    """

    s3_client: Any
    bucket_name: str

    def _load_event(self, path: Path) -> Event | None:
        """Get the event at the path given."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=str(path))
            with response['Body'] as stream:
                json_data = stream.read().decode('utf-8')
            event = Event.model_validate_json(json_data)
            return event
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            _logger.exception(f'Error reading event from {path}')
            return None
        except Exception:
            _logger.exception(f'Error reading event from {path}')
            return None

    def _store_event(self, path: Path, event: Event):
        """Store the event given at the path given."""
        data = event.model_dump(mode='json')
        json_str = json.dumps(data, indent=2)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=str(path),
            Body=json_str.encode('utf-8'),
        )

    def _search_paths(self, prefix: Path, page_id: str | None = None) -> list[Path]:
        """Search paths."""
        kwargs: dict[str, Any] = {
            'Bucket': self.bucket_name,
            'Prefix': str(prefix),
        }
        if page_id:
            kwargs['ContinuationToken'] = page_id

        response = self.s3_client.list_objects_v2(**kwargs)
        contents = response.get('Contents', [])
        paths = [Path(obj['Key']) for obj in contents]
        return paths


def _get_default_aws_endpoint_url() -> str | None:
    """Legacy fallback for aws endpoint url based on V0"""
    endpoint_url = os.getenv('AWS_S3_ENDPOINT')
    if not endpoint_url:
        return None
    secure = os.getenv('AWS_S3_SECURE', 'true').lower() == 'true'
    if secure:
        if not endpoint_url.startswith('https://'):
            endpoint_url = 'https://' + endpoint_url.removeprefix('http://')
    else:
        if not endpoint_url.startswith('http://'):
            endpoint_url = 'http://' + endpoint_url.removeprefix('https://')
    return endpoint_url


class AwsEventServiceInjector(EventServiceInjector):
    bucket_name: str
    prefix: Path = Path('users')
    endpoint_url: str | None = Field(default_factory=_get_default_aws_endpoint_url)

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

            # Use role-based authentication - boto3 will automatically
            # use IAM role credentials when running in AWS
            s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
            )

            yield AwsEventService(
                prefix=self.prefix,
                user_id=user_id,
                app_conversation_info_service=app_conversation_info_service,
                s3_client=s3_client,
                bucket_name=bucket_name,
                app_conversation_info_load_tasks={},
            )
