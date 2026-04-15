"""Tests for AwsEventService.

This module tests the AWS S3-based implementation of EventService,
focusing on search functionality and S3 operations.
"""

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import botocore.exceptions
import pytest

from openhands.app_server.event import aws_event_service
from openhands.app_server.event.aws_event_service import (
    AwsEventService,
    AwsEventServiceInjector,
)
from openhands.sdk.event import PauseEvent, TokenEvent


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    return MagicMock()


@pytest.fixture
def service(mock_s3_client) -> AwsEventService:
    """Create an AwsEventService instance for testing."""
    return AwsEventService(
        prefix=Path('users'),
        user_id='test_user',
        app_conversation_info_service=None,
        s3_client=mock_s3_client,
        bucket_name='test-bucket',
        app_conversation_info_load_tasks={},
    )


@pytest.fixture
def service_no_user(mock_s3_client) -> AwsEventService:
    """Create an AwsEventService instance without user_id."""
    return AwsEventService(
        prefix=Path('users'),
        user_id=None,
        app_conversation_info_service=None,
        s3_client=mock_s3_client,
        bucket_name='test-bucket',
        app_conversation_info_load_tasks={},
    )


def create_token_event() -> TokenEvent:
    """Helper to create a TokenEvent for testing."""
    return TokenEvent(
        source='agent', prompt_token_ids=[1, 2], response_token_ids=[3, 4]
    )


def create_pause_event() -> PauseEvent:
    """Helper to create a PauseEvent for testing."""
    return PauseEvent(source='user')


class TestAwsEventServiceLoadEvent:
    """Test cases for _load_event method."""

    def test_load_event_success(self, service: AwsEventService, mock_s3_client):
        """Test that _load_event successfully loads an event from S3."""
        event = create_token_event()
        json_data = event.model_dump_json()

        # Mock the S3 response
        mock_body = MagicMock()
        mock_body.read.return_value = json_data.encode('utf-8')
        mock_body.__enter__ = MagicMock(return_value=mock_body)
        mock_body.__exit__ = MagicMock(return_value=False)
        mock_s3_client.get_object.return_value = {'Body': mock_body}

        result = service._load_event(Path('some/path/event.json'))

        assert result is not None
        assert result.kind == 'TokenEvent'
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='test-bucket', Key='some/path/event.json'
        )

    def test_load_event_not_found(self, service: AwsEventService, mock_s3_client):
        """Test that _load_event returns None when event doesn't exist."""
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}
        mock_s3_client.get_object.side_effect = botocore.exceptions.ClientError(
            error_response, 'GetObject'
        )

        result = service._load_event(Path('some/path/missing.json'))

        assert result is None

    def test_load_event_other_error(self, service: AwsEventService, mock_s3_client):
        """Test that _load_event returns None and logs error on other S3 errors."""
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_s3_client.get_object.side_effect = botocore.exceptions.ClientError(
            error_response, 'GetObject'
        )

        result = service._load_event(Path('some/path/denied.json'))

        assert result is None


class TestAwsEventServiceStoreEvent:
    """Test cases for _store_event method."""

    def test_store_event_success(self, service: AwsEventService, mock_s3_client):
        """Test that _store_event successfully stores an event to S3."""
        event = create_token_event()

        service._store_event(Path('some/path/event.json'), event)

        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args.kwargs['Bucket'] == 'test-bucket'
        assert call_args.kwargs['Key'] == 'some/path/event.json'
        # Verify the body is valid JSON
        body = call_args.kwargs['Body'].decode('utf-8')
        data = json.loads(body)
        assert data['kind'] == 'TokenEvent'


class TestAwsEventServiceSearchPaths:
    """Test cases for _search_paths method."""

    def test_search_paths_returns_paths(self, service: AwsEventService, mock_s3_client):
        """Test that _search_paths returns paths from S3."""
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'users/test_user/v1_conversations/abc123/event1.json'},
                {'Key': 'users/test_user/v1_conversations/abc123/event2.json'},
            ]
        }

        result = service._search_paths(Path('users/test_user/v1_conversations/abc123'))

        assert len(result) == 2
        assert result[0] == Path('users/test_user/v1_conversations/abc123/event1.json')
        assert result[1] == Path('users/test_user/v1_conversations/abc123/event2.json')

    def test_search_paths_empty_bucket(self, service: AwsEventService, mock_s3_client):
        """Test that _search_paths handles empty results."""
        mock_s3_client.list_objects_v2.return_value = {}

        result = service._search_paths(Path('users/test_user/v1_conversations/abc123'))

        assert len(result) == 0

    def test_search_paths_with_page_id(self, service: AwsEventService, mock_s3_client):
        """Test that _search_paths uses continuation token."""
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [{'Key': 'event.json'}]
        }

        service._search_paths(Path('prefix'), page_id='continuation_token')

        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket='test-bucket',
            Prefix='prefix',
            ContinuationToken='continuation_token',
        )


class TestAwsEventServiceIntegration:
    """Integration tests for AwsEventService."""

    @pytest.mark.asyncio
    async def test_get_conversation_path_with_user_id(self, service: AwsEventService):
        """Test conversation path generation with user_id."""
        conversation_id = uuid4()

        path = await service.get_conversation_path(conversation_id)

        assert 'users' in str(path)
        assert 'test_user' in str(path)
        assert 'v1_conversations' in str(path)
        assert conversation_id.hex in str(path)

    @pytest.mark.asyncio
    async def test_get_conversation_path_without_user_id(
        self, service_no_user: AwsEventService
    ):
        """Test conversation path generation without user_id."""
        conversation_id = uuid4()

        path = await service_no_user.get_conversation_path(conversation_id)

        assert 'users' in str(path)
        assert 'test_user' not in str(path)
        assert 'v1_conversations' in str(path)
        assert conversation_id.hex in str(path)


class TestAwsEventServiceInjector:
    """Test cases for AwsEventServiceInjector."""

    def test_injector_has_bucket_name(self):
        """Test that injector has bucket_name attribute."""
        injector = AwsEventServiceInjector(bucket_name='my-bucket')
        assert injector.bucket_name == 'my-bucket'

    def test_injector_has_default_prefix(self):
        """Test that injector has default prefix."""
        injector = AwsEventServiceInjector(bucket_name='my-bucket')
        assert injector.prefix == Path('users')


class TestGetDefaultAwsEndpointUrl:
    """Test cases for _get_default_aws_endpoint_url function."""

    def test_no_env_vars_returns_none(self, monkeypatch):
        """Test that function returns None when no env vars are set."""
        monkeypatch.delenv('AWS_S3_ENDPOINT', raising=False)
        monkeypatch.delenv('AWS_S3_SECURE', raising=False)

        # Need to reload to get fresh default factory
        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result is None

    def test_endpoint_with_https_prefix_secure(self, monkeypatch):
        """Test endpoint with https:// prefix when secure=true."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'https://minio.example.com:9000')
        monkeypatch.setenv('AWS_S3_SECURE', 'true')

        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result == 'https://minio.example.com:9000'

    def test_endpoint_without_https_prefix_secure(self, monkeypatch):
        """Test endpoint without https:// prefix when secure=true adds it."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'minio.example.com:9000')
        monkeypatch.setenv('AWS_S3_SECURE', 'true')

        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result == 'https://minio.example.com:9000'

    def test_endpoint_with_http_prefix_insecure(self, monkeypatch):
        """Test endpoint with http:// prefix when secure=false."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'http://minio.example.com:9000')
        monkeypatch.setenv('AWS_S3_SECURE', 'false')

        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result == 'http://minio.example.com:9000'

    def test_endpoint_without_http_prefix_insecure(self, monkeypatch):
        """Test endpoint without http:// prefix when secure=false adds it."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'minio.example.com:9000')
        monkeypatch.setenv('AWS_S3_SECURE', 'false')

        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result == 'http://minio.example.com:9000'

    def test_endpoint_with_http_converted_to_https(self, monkeypatch):
        """Test http:// is converted to https:// when secure=true."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'http://minio.example.com:9000')
        monkeypatch.setenv('AWS_S3_SECURE', 'true')

        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result == 'https://minio.example.com:9000'

    def test_endpoint_with_https_converted_to_http(self, monkeypatch):
        """Test https:// is converted to http:// when secure=false."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'https://minio.example.com:9000')
        monkeypatch.setenv('AWS_S3_SECURE', 'false')

        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result == 'http://minio.example.com:9000'

    def test_secure_default_is_true(self, monkeypatch):
        """Test that secure defaults to true when not set."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'minio.example.com:9000')
        monkeypatch.delenv('AWS_S3_SECURE', raising=False)

        importlib.reload(aws_event_service)

        result = aws_event_service._get_default_aws_endpoint_url()
        assert result == 'https://minio.example.com:9000'


class TestAwsEventServiceInjectorEndpointUrl:
    """Test cases for AwsEventServiceInjector endpoint_url field."""

    def test_injector_endpoint_url_from_env(self, monkeypatch):
        """Test that endpoint_url is populated from environment variables."""
        monkeypatch.setenv('AWS_S3_ENDPOINT', 'minio.example.com:9000')
        monkeypatch.setenv('AWS_S3_SECURE', 'false')

        importlib.reload(aws_event_service)

        injector = aws_event_service.AwsEventServiceInjector(bucket_name='my-bucket')
        assert injector.endpoint_url == 'http://minio.example.com:9000'

    def test_injector_accepts_custom_endpoint_url(self, monkeypatch):
        """Test that injector accepts custom endpoint_url parameter."""
        monkeypatch.delenv('AWS_S3_ENDPOINT', raising=False)
        monkeypatch.delenv('AWS_S3_SECURE', raising=False)

        importlib.reload(aws_event_service)

        injector = aws_event_service.AwsEventServiceInjector(
            bucket_name='my-bucket', endpoint_url='https://custom.example.com:9000'
        )
        assert injector.endpoint_url == 'https://custom.example.com:9000'

    def test_injector_endpoint_url_none_when_no_env(self, monkeypatch):
        """Test that endpoint_url is None when no env vars set."""
        monkeypatch.delenv('AWS_S3_ENDPOINT', raising=False)
        monkeypatch.delenv('AWS_S3_SECURE', raising=False)

        importlib.reload(aws_event_service)

        injector = aws_event_service.AwsEventServiceInjector(bucket_name='my-bucket')
        assert injector.endpoint_url is None
