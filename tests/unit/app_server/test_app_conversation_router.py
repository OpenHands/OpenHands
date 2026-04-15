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
    AppConversationPage,
)
from openhands.app_server.app_conversation.app_conversation_router import (
    batch_get_app_conversations,
    count_app_conversations,
    search_app_conversations,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus


def _make_mock_app_conversation(
    conversation_id=None, user_id='test-user', sandbox_id=None
):
    """Create a mock AppConversation for testing."""
    if conversation_id is None:
        conversation_id = uuid4()
    if sandbox_id is None:
        sandbox_id = str(uuid4())
    return AppConversation(
        id=conversation_id,
        created_by_user_id=user_id,
        sandbox_id=sandbox_id,
        sandbox_status=SandboxStatus.RUNNING,
    )


def _make_mock_service(
    get_conversation_return=None,
    batch_get_return=None,
    search_return=None,
    count_return=0,
):
    """Create a mock AppConversationService for testing."""
    service = MagicMock()
    service.get_app_conversation = AsyncMock(return_value=get_conversation_return)
    service.batch_get_app_conversations = AsyncMock(return_value=batch_get_return or [])
    service.search_app_conversations = AsyncMock(
        return_value=search_return or AppConversationPage(items=[])
    )
    service.count_app_conversations = AsyncMock(return_value=count_return)
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


@pytest.mark.asyncio
class TestSearchAppConversations:
    """Test suite for search_app_conversations endpoint."""

    async def test_search_with_sandbox_id_filter(self):
        """Test that sandbox_id__eq filter is passed to the service.

        Arrange: Create mock service and specific sandbox_id
        Act: Call search_app_conversations with sandbox_id__eq
        Assert: Service is called with the sandbox_id__eq parameter
        """
        # Arrange
        sandbox_id = 'test-sandbox-123'
        mock_conversation = _make_mock_app_conversation(sandbox_id=sandbox_id)
        mock_service = _make_mock_service(
            search_return=AppConversationPage(items=[mock_conversation])
        )

        # Act
        result = await search_app_conversations(
            sandbox_id__eq=sandbox_id,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.search_app_conversations.assert_called_once()
        call_kwargs = mock_service.search_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert len(result.items) == 1
        assert result.items[0].sandbox_id == sandbox_id

    async def test_search_without_sandbox_id_filter(self):
        """Test that sandbox_id__eq defaults to None when not provided.

        Arrange: Create mock service
        Act: Call search_app_conversations without sandbox_id__eq
        Assert: Service is called with sandbox_id__eq=None
        """
        # Arrange
        mock_service = _make_mock_service()

        # Act
        await search_app_conversations(
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.search_app_conversations.assert_called_once()
        call_kwargs = mock_service.search_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') is None

    async def test_search_with_sandbox_id_and_other_filters(self):
        """Test that sandbox_id__eq works correctly with other filters.

        Arrange: Create mock service
        Act: Call search_app_conversations with sandbox_id__eq and other filters
        Assert: Service is called with all parameters correctly
        """
        # Arrange
        sandbox_id = 'test-sandbox-456'
        mock_service = _make_mock_service()

        # Act
        await search_app_conversations(
            title__contains='test',
            sandbox_id__eq=sandbox_id,
            limit=50,
            include_sub_conversations=True,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.search_app_conversations.assert_called_once()
        call_kwargs = mock_service.search_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert call_kwargs.get('title__contains') == 'test'
        assert call_kwargs.get('limit') == 50
        assert call_kwargs.get('include_sub_conversations') is True


@pytest.mark.asyncio
class TestCountAppConversations:
    """Test suite for count_app_conversations endpoint."""

    async def test_count_with_sandbox_id_filter(self):
        """Test that sandbox_id__eq filter is passed to the service.

        Arrange: Create mock service with count return value
        Act: Call count_app_conversations with sandbox_id__eq
        Assert: Service is called with the sandbox_id__eq parameter
        """
        # Arrange
        sandbox_id = 'test-sandbox-789'
        mock_service = _make_mock_service(count_return=5)

        # Act
        result = await count_app_conversations(
            sandbox_id__eq=sandbox_id,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.count_app_conversations.assert_called_once()
        call_kwargs = mock_service.count_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert result == 5

    async def test_count_without_sandbox_id_filter(self):
        """Test that sandbox_id__eq defaults to None when not provided.

        Arrange: Create mock service
        Act: Call count_app_conversations without sandbox_id__eq
        Assert: Service is called with sandbox_id__eq=None
        """
        # Arrange
        mock_service = _make_mock_service(count_return=10)

        # Act
        result = await count_app_conversations(
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.count_app_conversations.assert_called_once()
        call_kwargs = mock_service.count_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') is None
        assert result == 10

    async def test_count_with_sandbox_id_and_other_filters(self):
        """Test that sandbox_id__eq works correctly with other filters.

        Arrange: Create mock service
        Act: Call count_app_conversations with sandbox_id__eq and other filters
        Assert: Service is called with all parameters correctly
        """
        # Arrange
        sandbox_id = 'test-sandbox-abc'
        mock_service = _make_mock_service(count_return=3)

        # Act
        result = await count_app_conversations(
            title__contains='test',
            sandbox_id__eq=sandbox_id,
            app_conversation_service=mock_service,
        )

        # Assert
        mock_service.count_app_conversations.assert_called_once()
        call_kwargs = mock_service.count_app_conversations.call_args[1]
        assert call_kwargs.get('sandbox_id__eq') == sandbox_id
        assert call_kwargs.get('title__contains') == 'test'
        assert result == 3
