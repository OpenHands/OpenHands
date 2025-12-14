"""Tests for public conversation service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from openhands.server.services.public_conversation_service import (
    PublicConversationService,
)
from openhands.storage.data_models.conversation_metadata import ConversationMetadata


@pytest.fixture
def mock_conversation_store():
    """Mock conversation store."""
    store = AsyncMock()
    return store


@pytest.fixture
def public_service(mock_conversation_store):
    """Public conversation service with mocked store."""
    return PublicConversationService(mock_conversation_store)


@pytest.fixture
def sample_metadata():
    """Sample conversation metadata."""
    return ConversationMetadata(
        conversation_id='test-conversation-123',
        title='Test Conversation',
        user_id='user-123',
        selected_repository='test/repo',
        is_public=False,
        public_share_token=None,
        shared_at=None,
        created_at=datetime.now(timezone.utc),
    )


class TestPublicConversationService:
    """Test cases for PublicConversationService."""

    @pytest.mark.asyncio
    async def test_make_conversation_public_success(
        self, public_service, mock_conversation_store, sample_metadata
    ):
        """Test successfully making a conversation public."""
        # Setup
        conversation_id = 'test-conversation-123'
        user_id = 'user-123'

        mock_conversation_store.validate_metadata.return_value = True
        mock_conversation_store.get_metadata.return_value = sample_metadata
        mock_conversation_store.save_metadata.return_value = None

        # Execute
        share_token = await public_service.make_conversation_public(
            conversation_id, user_id
        )

        # Verify
        assert share_token is not None
        assert len(share_token) > 0
        assert sample_metadata.is_public is True
        assert sample_metadata.public_share_token == share_token
        assert sample_metadata.shared_at is not None

        mock_conversation_store.validate_metadata.assert_called_once_with(
            conversation_id, user_id
        )
        mock_conversation_store.get_metadata.assert_called_once_with(conversation_id)
        mock_conversation_store.save_metadata.assert_called_once_with(sample_metadata)

    @pytest.mark.asyncio
    async def test_make_conversation_public_permission_denied(
        self, public_service, mock_conversation_store
    ):
        """Test permission denied when user doesn't own conversation."""
        # Setup
        conversation_id = 'test-conversation-123'
        user_id = 'user-123'

        mock_conversation_store.validate_metadata.return_value = False

        # Execute & Verify
        with pytest.raises(
            PermissionError, match='User does not own this conversation'
        ):
            await public_service.make_conversation_public(conversation_id, user_id)

    @pytest.mark.asyncio
    async def test_make_conversation_private_success(
        self, public_service, mock_conversation_store, sample_metadata
    ):
        """Test successfully making a conversation private."""
        # Setup
        conversation_id = 'test-conversation-123'
        user_id = 'user-123'

        # Make it public first
        sample_metadata.is_public = True
        sample_metadata.public_share_token = 'test-token'
        sample_metadata.shared_at = datetime.now(timezone.utc)

        mock_conversation_store.validate_metadata.return_value = True
        mock_conversation_store.get_metadata.return_value = sample_metadata
        mock_conversation_store.save_metadata.return_value = None

        # Execute
        await public_service.make_conversation_private(conversation_id, user_id)

        # Verify
        assert sample_metadata.is_public is False
        assert sample_metadata.public_share_token is None
        assert sample_metadata.shared_at is None

        mock_conversation_store.validate_metadata.assert_called_once_with(
            conversation_id, user_id
        )
        mock_conversation_store.get_metadata.assert_called_once_with(conversation_id)
        mock_conversation_store.save_metadata.assert_called_once_with(sample_metadata)

    @pytest.mark.asyncio
    async def test_get_public_conversation_success(
        self, public_service, mock_conversation_store, sample_metadata
    ):
        """Test getting public conversation info."""
        # Setup
        conversation_id = 'test-conversation-123'
        sample_metadata.is_public = True

        mock_conversation_store.get_metadata.return_value = sample_metadata

        # Execute
        result = await public_service.get_public_conversation(conversation_id)

        # Verify
        assert result is not None
        assert result.conversation_id == conversation_id
        assert result.title == 'Test Conversation'

        mock_conversation_store.get_metadata.assert_called_once_with(conversation_id)

    @pytest.mark.asyncio
    async def test_get_public_conversation_not_public(
        self, public_service, mock_conversation_store, sample_metadata
    ):
        """Test getting conversation that is not public."""
        # Setup
        conversation_id = 'test-conversation-123'
        sample_metadata.is_public = False

        mock_conversation_store.get_metadata.return_value = sample_metadata

        # Execute
        result = await public_service.get_public_conversation(conversation_id)

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_public_conversation_not_found(
        self, public_service, mock_conversation_store
    ):
        """Test getting conversation that doesn't exist."""
        # Setup
        conversation_id = 'nonexistent-conversation'

        mock_conversation_store.get_metadata.side_effect = FileNotFoundError()

        # Execute
        result = await public_service.get_public_conversation(conversation_id)

        # Verify
        assert result is None

    def test_filter_sensitive_content(self, public_service):
        """Test filtering of sensitive content."""
        # Test cases
        test_cases = [
            ('Hello world', 'Hello world'),  # Safe content
            (
                'My api_key is secret',
                '[Content filtered for security]',
            ),  # Contains API key
            (
                'Bearer token123',
                '[Content filtered for security]',
            ),  # Contains bearer token
            (
                'password=secret123',
                '[Content filtered for security]',
            ),  # Contains password
            ('Normal conversation', 'Normal conversation'),  # Safe content
        ]

        for input_content, expected_output in test_cases:
            result = public_service._filter_sensitive_content(input_content)
            assert result == expected_output

    def test_to_public_conversation_info(self, public_service, sample_metadata):
        """Test conversion to public conversation info."""
        # Execute
        result = public_service._to_public_conversation_info(sample_metadata)

        # Verify
        assert result.conversation_id == sample_metadata.conversation_id
        assert result.title == sample_metadata.title
        assert result.created_at == sample_metadata.created_at
        # Verify sensitive fields are not included
        assert not hasattr(result, 'user_id')
        assert not hasattr(result, 'session_api_key')
