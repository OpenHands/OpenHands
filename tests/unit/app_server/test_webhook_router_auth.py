"""Tests for webhook_router valid_sandbox and valid_conversation functions.

This module tests the webhook authentication and authorization logic.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException, status

from openhands.app_server.event_callback.webhook_router import (
    valid_conversation,
    valid_sandbox,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.user.specifiy_user_context import ADMIN


class TestValidSandbox:
    """Test suite for valid_sandbox function."""

    @pytest.mark.asyncio
    async def test_valid_sandbox_with_valid_api_key(self):
        """Test that valid API key returns sandbox info."""
        # Arrange
        session_api_key = 'valid-api-key-123'
        expected_sandbox = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            created_by_user_id='user-123',
            sandbox_spec_id='spec-123',
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=expected_sandbox
        )

        # Act
        result = await valid_sandbox(
            user_context=ADMIN,
            session_api_key=session_api_key,
            sandbox_service=mock_sandbox_service,
        )

        # Assert
        assert result == expected_sandbox
        mock_sandbox_service.get_sandbox_by_session_api_key.assert_called_once_with(
            session_api_key
        )

    @pytest.mark.asyncio
    async def test_valid_sandbox_without_api_key_raises_401(self):
        """Test that missing API key raises 401 error."""
        # Arrange
        mock_sandbox_service = AsyncMock()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await valid_sandbox(
                user_context=ADMIN,
                session_api_key=None,
                sandbox_service=mock_sandbox_service,
            )

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'X-Session-API-Key header is required' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_sandbox_with_invalid_api_key_raises_401(self):
        """Test that invalid API key raises 401 error."""
        # Arrange
        session_api_key = 'invalid-api-key'
        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=None
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await valid_sandbox(
                user_context=ADMIN,
                session_api_key=session_api_key,
                sandbox_service=mock_sandbox_service,
            )

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'Invalid session API key' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_sandbox_with_empty_api_key_raises_401(self):
        """Test that empty API key raises 401 error (same as missing key)."""
        # Arrange - empty string is falsy, so it gets rejected at the check
        session_api_key = ''
        mock_sandbox_service = AsyncMock()

        # Act & Assert - should raise 401 because empty string fails the truth check
        with pytest.raises(HTTPException) as exc_info:
            await valid_sandbox(
                user_context=ADMIN,
                session_api_key=session_api_key,
                sandbox_service=mock_sandbox_service,
            )

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'X-Session-API-Key header is required' in exc_info.value.detail
        # Verify the sandbox service was NOT called (rejected before lookup)
        mock_sandbox_service.get_sandbox_by_session_api_key.assert_not_called()


class TestValidConversation:
    """Test suite for valid_conversation function."""

    @pytest.mark.asyncio
    async def test_valid_conversation_existing_returns_info(self):
        """Test that existing conversation returns info."""
        # Arrange
        conversation_id = uuid4()
        sandbox_info = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key='api-key',
            created_by_user_id='user-123',
            sandbox_spec_id='spec-123',
        )

        expected_info = MagicMock()
        expected_info.created_by_user_id = 'user-123'

        mock_service = AsyncMock()
        mock_service.get_app_conversation_info = AsyncMock(return_value=expected_info)

        # Act
        result = await valid_conversation(
            conversation_id=conversation_id,
            sandbox_info=sandbox_info,
            app_conversation_info_service=mock_service,
        )

        # Assert
        assert result == expected_info

    @pytest.mark.asyncio
    async def test_valid_conversation_new_creates_stub(self):
        """Test that non-existing conversation creates a stub."""
        # Arrange
        conversation_id = uuid4()
        sandbox_info = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key='api-key',
            created_by_user_id='user-123',
            sandbox_spec_id='spec-123',
        )

        mock_service = AsyncMock()
        mock_service.get_app_conversation_info = AsyncMock(return_value=None)

        # Act
        result = await valid_conversation(
            conversation_id=conversation_id,
            sandbox_info=sandbox_info,
            app_conversation_info_service=mock_service,
        )

        # Assert
        assert result.id == conversation_id
        assert result.sandbox_id == sandbox_info.id
        assert result.created_by_user_id == sandbox_info.created_by_user_id

    @pytest.mark.asyncio
    async def test_valid_conversation_different_user_raises_auth_error(self):
        """Test that conversation from different user raises AuthError."""
        # Arrange
        conversation_id = uuid4()
        sandbox_info = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key='api-key',
            created_by_user_id='user-123',
            sandbox_spec_id='spec-123',
        )

        # Conversation created by different user
        different_user_info = MagicMock()
        different_user_info.created_by_user_id = 'different-user-id'

        mock_service = AsyncMock()
        mock_service.get_app_conversation_info = AsyncMock(
            return_value=different_user_info
        )

        # Act & Assert
        from openhands.app_server.errors import AuthError

        with pytest.raises(AuthError):
            await valid_conversation(
                conversation_id=conversation_id,
                sandbox_info=sandbox_info,
                app_conversation_info_service=mock_service,
            )

    @pytest.mark.asyncio
    async def test_valid_conversation_same_user_succeeds(self):
        """Test that conversation from same user succeeds."""
        # Arrange
        conversation_id = uuid4()
        user_id = 'user-123'
        sandbox_info = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key='api-key',
            created_by_user_id=user_id,
            sandbox_spec_id='spec-123',
        )

        # Conversation created by same user
        same_user_info = MagicMock()
        same_user_info.created_by_user_id = user_id

        mock_service = AsyncMock()
        mock_service.get_app_conversation_info = AsyncMock(return_value=same_user_info)

        # Act
        result = await valid_conversation(
            conversation_id=conversation_id,
            sandbox_info=sandbox_info,
            app_conversation_info_service=mock_service,
        )

        # Assert
        assert result == same_user_info


class TestWebhookAuthenticationIntegration:
    """Integration tests for webhook authentication flow."""

    @pytest.mark.asyncio
    async def test_full_auth_flow_valid_key(self):
        """Test complete auth flow with valid API key."""
        # Arrange
        session_api_key = 'valid-api-key'
        sandbox_info = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            created_by_user_id='user-123',
            sandbox_spec_id='spec-123',
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=sandbox_info
        )

        conversation_info = MagicMock()
        conversation_info.created_by_user_id = 'user-123'

        mock_conversation_service = AsyncMock()
        mock_conversation_service.get_app_conversation_info = AsyncMock(
            return_value=conversation_info
        )

        # Act - Call valid_sandbox first
        sandbox_result = await valid_sandbox(
            user_context=ADMIN,
            session_api_key=session_api_key,
            sandbox_service=mock_sandbox_service,
        )

        # Then call valid_conversation
        conversation_result = await valid_conversation(
            conversation_id=uuid4(),
            sandbox_info=sandbox_result,
            app_conversation_info_service=mock_conversation_service,
        )

        # Assert
        assert sandbox_result.id == 'sandbox-123'
        assert conversation_result.created_by_user_id == 'user-123'

    @pytest.mark.asyncio
    async def test_full_auth_flow_invalid_key_fails(self):
        """Test complete auth flow with invalid API key fails at sandbox validation."""
        # Arrange
        session_api_key = 'invalid-api-key'
        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=None
        )

        # Act & Assert - Should fail at valid_sandbox
        with pytest.raises(HTTPException) as exc_info:
            await valid_sandbox(
                user_context=ADMIN,
                session_api_key=session_api_key,
                sandbox_service=mock_sandbox_service,
            )

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_full_auth_flow_wrong_user_fails(self):
        """Test complete auth flow with valid key but wrong user fails."""
        # Arrange
        session_api_key = 'valid-api-key'
        sandbox_info = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            created_by_user_id='user-123',
            sandbox_spec_id='spec-123',
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=sandbox_info
        )

        # Conversation created by different user
        different_user_info = MagicMock()
        different_user_info.created_by_user_id = 'different-user'

        mock_conversation_service = AsyncMock()
        mock_conversation_service.get_app_conversation_info = AsyncMock(
            return_value=different_user_info
        )

        # Act - valid_sandbox succeeds
        sandbox_result = await valid_sandbox(
            user_context=ADMIN,
            session_api_key=session_api_key,
            sandbox_service=mock_sandbox_service,
        )

        # But valid_conversation fails
        from openhands.app_server.errors import AuthError

        with pytest.raises(AuthError):
            await valid_conversation(
                conversation_id=uuid4(),
                sandbox_info=sandbox_result,
                app_conversation_info_service=mock_conversation_service,
            )
