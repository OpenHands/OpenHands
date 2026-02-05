"""Unit tests for the app_conversation_router endpoints.

This module tests the batch_get_app_conversations endpoint,
focusing on UUID string parsing, validation, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException, status

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
)
from openhands.app_server.app_conversation.app_conversation_router import (
    batch_get_app_conversations,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus


def _make_mock_app_conversation(conversation_id=None, user_id='test-user'):
    """Create a mock AppConversation for testing."""
    if conversation_id is None:
        conversation_id = uuid4()
    return AppConversation(
        id=conversation_id,
        created_by_user_id=user_id,
        sandbox_id=str(uuid4()),
        sandbox_status=SandboxStatus.RUNNING,
    )


def _make_mock_service(
    get_conversation_return=None,
    batch_get_return=None,
):
    """Create a mock AppConversationService for testing."""
    service = MagicMock()
    service.get_app_conversation = AsyncMock(return_value=get_conversation_return)
    service.batch_get_app_conversations = AsyncMock(return_value=batch_get_return or [])
    return service


@pytest.mark.asyncio
class TestBatchGetAppConversations:
    """Test suite for batch_get_app_conversations endpoint."""

    async def test_accepts_uuids_with_dashes(self):
        """Test that standard UUIDs with dashes are accepted.

        Arrange: Create UUIDs with dashes and mock service
        Act: Call batch_get_app_conversations
        Assert: Service is called with parsed UUIDs
        """
        # Arrange
        uuid1 = uuid4()
        uuid2 = uuid4()
        ids = [str(uuid1), str(uuid2)]

        mock_conversations = [
            _make_mock_app_conversation(uuid1),
            _make_mock_app_conversation(uuid2),
        ]
        mock_service = _make_mock_service(batch_get_return=mock_conversations)

        # Act
        result = await batch_get_app_conversations(
            ids=ids,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.batch_get_app_conversations.assert_called_once()
        call_args = mock_service.batch_get_app_conversations.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == uuid1
        assert call_args[1] == uuid2
        assert result == mock_conversations

    async def test_accepts_uuids_without_dashes(self):
        """Test that UUIDs without dashes are accepted and correctly parsed.

        Arrange: Create UUIDs without dashes
        Act: Call batch_get_app_conversations
        Assert: Service is called with correctly parsed UUIDs
        """
        # Arrange
        uuid1 = uuid4()
        uuid2 = uuid4()
        # Remove dashes from UUID strings
        ids = [str(uuid1).replace('-', ''), str(uuid2).replace('-', '')]

        mock_conversations = [
            _make_mock_app_conversation(uuid1),
            _make_mock_app_conversation(uuid2),
        ]
        mock_service = _make_mock_service(batch_get_return=mock_conversations)

        # Act
        result = await batch_get_app_conversations(
            ids=ids,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.batch_get_app_conversations.assert_called_once()
        call_args = mock_service.batch_get_app_conversations.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == uuid1
        assert call_args[1] == uuid2
        assert result == mock_conversations

    async def test_returns_400_for_invalid_uuid_strings(self):
        """Test that invalid UUID strings return 400 Bad Request.

        Arrange: Create list with invalid UUID strings
        Act: Call batch_get_app_conversations
        Assert: HTTPException is raised with 400 status and details about invalid IDs
        """
        # Arrange
        valid_uuid = str(uuid4())
        invalid_ids = ['not-a-uuid', 'also-invalid', '12345']
        ids = [valid_uuid] + invalid_ids

        mock_service = _make_mock_service()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await batch_get_app_conversations(
                ids=ids,
                app_conversation_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid UUID format' in exc_info.value.detail
        # All invalid IDs should be mentioned in the error
        for invalid_id in invalid_ids:
            assert invalid_id in exc_info.value.detail

    async def test_returns_400_for_too_many_ids(self):
        """Test that requesting too many IDs returns 400 Bad Request.

        Arrange: Create list with 100+ IDs
        Act: Call batch_get_app_conversations
        Assert: HTTPException is raised with 400 status
        """
        # Arrange
        ids = [str(uuid4()) for _ in range(100)]
        mock_service = _make_mock_service()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await batch_get_app_conversations(
                ids=ids,
                app_conversation_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Too many ids' in exc_info.value.detail

    async def test_returns_empty_list_for_empty_input(self):
        """Test that empty input returns empty list.

        Arrange: Create empty list of IDs
        Act: Call batch_get_app_conversations
        Assert: Empty list is returned
        """
        # Arrange
        mock_service = _make_mock_service(batch_get_return=[])

        # Act
        result = await batch_get_app_conversations(
            ids=[],
            app_conversation_service=mock_service,
        )

        # Assert
        assert result == []
        mock_service.batch_get_app_conversations.assert_called_once_with([])

    async def test_returns_none_for_missing_conversations(self):
        """Test that None is returned for conversations that don't exist.

        Arrange: Request IDs where some don't exist
        Act: Call batch_get_app_conversations
        Assert: Result contains None for missing conversations
        """
        # Arrange
        uuid1 = uuid4()
        uuid2 = uuid4()
        ids = [str(uuid1), str(uuid2)]

        # Only first conversation exists
        mock_service = _make_mock_service(
            batch_get_return=[_make_mock_app_conversation(uuid1), None]
        )

        # Act
        result = await batch_get_app_conversations(
            ids=ids,
            app_conversation_service=mock_service,
        )

        # Assert
        assert len(result) == 2
        assert result[0] is not None
        assert result[0].id == uuid1
        assert result[1] is None
