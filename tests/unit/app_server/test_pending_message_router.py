"""Unit tests for the pending_message_router endpoints.

This module tests the queue_pending_message endpoint,
focusing on request validation and rate limiting.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException, status

from openhands.agent_server.models import TextContent
from openhands.app_server.pending_messages.pending_message_models import (
    PendingMessageResponse,
)
from openhands.app_server.pending_messages.pending_message_router import (
    queue_pending_message,
)


def _make_mock_service(
    add_message_return=None,
    count_pending_messages_return=0,
):
    """Create a mock PendingMessageService for testing."""
    service = MagicMock()
    service.add_message = AsyncMock(return_value=add_message_return)
    service.count_pending_messages = AsyncMock(
        return_value=count_pending_messages_return
    )
    return service


def _make_mock_request(body: dict):
    """Create a mock FastAPI Request with given JSON body."""
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
class TestQueuePendingMessage:
    """Test suite for queue_pending_message endpoint."""

    async def test_queues_message_successfully(self):
        """Test that a valid message is queued successfully."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Hello, world!'}]
        expected_response = PendingMessageResponse(
            id=str(uuid4()),
            queued=True,
            position=1,
        )
        mock_service = _make_mock_service(
            add_message_return=expected_response,
            count_pending_messages_return=0,
        )
        mock_request = _make_mock_request({'content': raw_content, 'role': 'user'})

        # Act
        result = await queue_pending_message(
            conversation_id=conversation_id,
            request=mock_request,
            pending_service=mock_service,
        )

        # Assert
        assert result == expected_response
        mock_service.add_message.assert_called_once()
        call_kwargs = mock_service.add_message.call_args.kwargs
        assert call_kwargs['conversation_id'] == conversation_id
        assert call_kwargs['role'] == 'user'
        # Content should be parsed into typed objects
        assert len(call_kwargs['content']) == 1
        assert isinstance(call_kwargs['content'][0], TextContent)
        assert call_kwargs['content'][0].text == 'Hello, world!'

    async def test_uses_default_role_when_not_provided(self):
        """Test that 'user' role is used by default."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Test message'}]
        expected_response = PendingMessageResponse(
            id=str(uuid4()),
            queued=True,
            position=1,
        )
        mock_service = _make_mock_service(
            add_message_return=expected_response,
            count_pending_messages_return=0,
        )
        mock_request = _make_mock_request({'content': raw_content})

        # Act
        await queue_pending_message(
            conversation_id=conversation_id,
            request=mock_request,
            pending_service=mock_service,
        )

        # Assert
        mock_service.add_message.assert_called_once()
        call_kwargs = mock_service.add_message.call_args.kwargs
        assert call_kwargs['conversation_id'] == conversation_id
        assert call_kwargs['role'] == 'user'
        assert isinstance(call_kwargs['content'][0], TextContent)

    async def test_returns_400_for_invalid_json_body(self):
        """Test that invalid JSON body returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = MagicMock()
        mock_request.json = AsyncMock(side_effect=Exception('Invalid JSON'))

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid request body' in exc_info.value.detail

    async def test_returns_400_when_content_is_missing(self):
        """Test that missing content returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = _make_mock_request({'role': 'user'})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'content must be a non-empty list' in exc_info.value.detail

    async def test_returns_400_when_content_is_not_a_list(self):
        """Test that non-list content returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = _make_mock_request({'content': 'not a list'})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'content must be a non-empty list' in exc_info.value.detail

    async def test_returns_400_when_content_is_empty_list(self):
        """Test that empty list content returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = _make_mock_request({'content': []})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'content must be a non-empty list' in exc_info.value.detail

    async def test_returns_429_when_rate_limit_exceeded(self):
        """Test that exceeding rate limit returns 429 Too Many Requests."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Test message'}]
        mock_service = _make_mock_service(count_pending_messages_return=10)
        mock_request = _make_mock_request({'content': raw_content})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert 'Maximum 10 messages' in exc_info.value.detail

    async def test_allows_up_to_10_messages(self):
        """Test that 9 existing messages still allows adding one more."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Test message'}]
        expected_response = PendingMessageResponse(
            id=str(uuid4()),
            queued=True,
            position=10,
        )
        mock_service = _make_mock_service(
            add_message_return=expected_response,
            count_pending_messages_return=9,
        )
        mock_request = _make_mock_request({'content': raw_content})

        # Act
        result = await queue_pending_message(
            conversation_id=conversation_id,
            request=mock_request,
            pending_service=mock_service,
        )

        # Assert
        assert result == expected_response
        mock_service.add_message.assert_called_once()
