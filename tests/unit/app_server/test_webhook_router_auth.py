"""Tests for webhook_router valid_sandbox and valid_conversation functions.

This module tests the webhook authentication and authorization logic.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from openhands.app_server.event_callback.webhook_router import (
    router as webhook_router,
)
from openhands.app_server.event_callback.webhook_router import (
    valid_conversation,
    valid_sandbox,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.user.specifiy_user_context import (
    USER_CONTEXT_ATTR,
    SpecifyUserContext,
)
from openhands.server.types import AppMode


class MockRequestState:
    """A mock request state that tracks attribute assignments."""

    def __init__(self):
        self._state = {}
        self._attributes = {}

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._attributes[name] = value

    def __getattr__(self, name):
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def create_mock_request():
    """Create a mock FastAPI Request object with proper state."""
    request = MagicMock()
    request.state = MockRequestState()
    return request


def create_sandbox_service_context_manager(sandbox_service):
    """Create an async context manager that yields the given sandbox service."""

    @contextlib.asynccontextmanager
    async def _context_manager(state, request=None):
        yield sandbox_service

    return _context_manager


class TestValidSandbox:
    """Test suite for valid_sandbox function."""

    @pytest.mark.asyncio
    async def test_valid_sandbox_with_valid_api_key(self):
        """Test that valid API key returns sandbox info and sets user_context."""
        # Arrange
        session_api_key = 'valid-api-key-123'
        user_id = 'user-123'
        expected_sandbox = SandboxInfo(
            id='sandbox-123',
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            created_by_user_id=user_id,
            sandbox_spec_id='spec-123',
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=expected_sandbox
        )

        mock_request = create_mock_request()

        # Act
        with patch(
            'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
            create_sandbox_service_context_manager(mock_sandbox_service),
        ):
            result = await valid_sandbox(
                request=mock_request,
                session_api_key=session_api_key,
            )

        # Assert
        assert result == expected_sandbox
        mock_sandbox_service.get_sandbox_by_session_api_key.assert_called_once_with(
            session_api_key
        )

        # Verify user_context is set correctly on request.state
        assert USER_CONTEXT_ATTR in mock_request.state._attributes
        user_context = mock_request.state._attributes[USER_CONTEXT_ATTR]
        assert isinstance(user_context, SpecifyUserContext)
        assert user_context.user_id == user_id

    @pytest.mark.asyncio
    async def test_valid_sandbox_sets_user_context_to_sandbox_owner(self):
        """Test that user_context is set to the sandbox owner's user ID."""
        # Arrange
        session_api_key = 'valid-api-key'
        sandbox_owner_id = 'sandbox-owner-user-id'
        expected_sandbox = SandboxInfo(
            id='sandbox-456',
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            created_by_user_id=sandbox_owner_id,
            sandbox_spec_id='spec-456',
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=expected_sandbox
        )

        mock_request = create_mock_request()

        # Act
        with patch(
            'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
            create_sandbox_service_context_manager(mock_sandbox_service),
        ):
            await valid_sandbox(
                request=mock_request,
                session_api_key=session_api_key,
            )

        # Assert - user_context should be set to the sandbox owner
        assert USER_CONTEXT_ATTR in mock_request.state._attributes
        user_context = mock_request.state._attributes[USER_CONTEXT_ATTR]
        assert isinstance(user_context, SpecifyUserContext)
        assert user_context.user_id == sandbox_owner_id

    @pytest.mark.asyncio
    async def test_valid_sandbox_no_user_context_when_no_user_id(self):
        """Test that user_context is not set when sandbox has no created_by_user_id."""
        # Arrange
        session_api_key = 'valid-api-key'
        expected_sandbox = SandboxInfo(
            id='sandbox-789',
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            created_by_user_id=None,  # No user ID
            sandbox_spec_id='spec-789',
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=expected_sandbox
        )

        mock_request = create_mock_request()

        # Act
        with patch(
            'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
            create_sandbox_service_context_manager(mock_sandbox_service),
        ):
            result = await valid_sandbox(
                request=mock_request,
                session_api_key=session_api_key,
            )

        # Assert - sandbox is returned but user_context should NOT be set
        assert result == expected_sandbox

        # Verify user_context is NOT set on request.state
        assert USER_CONTEXT_ATTR not in mock_request.state._attributes

    @pytest.mark.asyncio
    async def test_valid_sandbox_no_user_context_when_no_user_id_raises_401_in_saas_mode(
        self,
    ):
        """Test that user_context is not set when sandbox has no created_by_user_id."""
        # Arrange
        session_api_key = 'valid-api-key'
        expected_sandbox = SandboxInfo(
            id='sandbox-789',
            status=SandboxStatus.RUNNING,
            session_api_key=session_api_key,
            created_by_user_id=None,  # No user ID
            sandbox_spec_id='spec-789',
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox_by_session_api_key = AsyncMock(
            return_value=expected_sandbox
        )

        mock_request = create_mock_request()

        # Act
        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
                create_sandbox_service_context_manager(mock_sandbox_service),
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.app_mode',
                AppMode.SAAS,
            ),
        ):
            with pytest.raises(HTTPException) as excinfo:
                await valid_sandbox(
                    request=mock_request,
                    session_api_key=session_api_key,
                )
            assert excinfo.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_sandbox_without_api_key_raises_401(self):
        """Test that missing API key raises 401 error."""
        # Arrange
        mock_request = create_mock_request()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await valid_sandbox(
                request=mock_request,
                session_api_key=None,
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

        mock_request = create_mock_request()

        # Act & Assert
        with patch(
            'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
            create_sandbox_service_context_manager(mock_sandbox_service),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await valid_sandbox(
                    request=mock_request,
                    session_api_key=session_api_key,
                )

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'Invalid session API key' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_sandbox_with_empty_api_key_raises_401(self):
        """Test that empty API key raises 401 error (same as missing key)."""
        # Arrange - empty string is falsy, so it gets rejected at the check
        session_api_key = ''
        mock_sandbox_service = AsyncMock()
        mock_request = create_mock_request()

        # Act & Assert - should raise 401 because empty string fails the truth check
        with pytest.raises(HTTPException) as exc_info:
            await valid_sandbox(
                request=mock_request,
                session_api_key=session_api_key,
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

        mock_request = create_mock_request()

        # Act - Call valid_sandbox first
        with patch(
            'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
            create_sandbox_service_context_manager(mock_sandbox_service),
        ):
            sandbox_result = await valid_sandbox(
                request=mock_request,
                session_api_key=session_api_key,
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

        mock_request = create_mock_request()

        # Act & Assert - Should fail at valid_sandbox
        with patch(
            'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
            create_sandbox_service_context_manager(mock_sandbox_service),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await valid_sandbox(
                    request=mock_request,
                    session_api_key=session_api_key,
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

        mock_request = create_mock_request()

        # Act - valid_sandbox succeeds
        with patch(
            'openhands.app_server.event_callback.webhook_router.get_sandbox_service',
            create_sandbox_service_context_manager(mock_sandbox_service),
        ):
            sandbox_result = await valid_sandbox(
                request=mock_request,
                session_api_key=session_api_key,
            )

        # But valid_conversation fails
        from openhands.app_server.errors import AuthError

        with pytest.raises(AuthError):
            await valid_conversation(
                conversation_id=uuid4(),
                sandbox_info=sandbox_result,
                app_conversation_info_service=mock_conversation_service,
            )


class TestWebhookRouterHTTPIntegration:
    """Integration tests for webhook router HTTP layer.

    These tests validate that FastAPI routing correctly extracts conversation_id
    from the request body rather than requiring it as a query parameter.
    """

    def test_conversation_update_endpoint_does_not_require_query_param(self):
        """Test that /webhooks/conversations endpoint accepts conversation_id in body only.

        This test validates the fix for the regression where the endpoint incorrectly
        required conversation_id as a query parameter due to using Depends(valid_conversation).

        The endpoint should:
        1. Accept POST requests without any query parameters
        2. Extract conversation_id from the request body (conversation_info.id)
        3. Return 401 (not 422) when auth fails, proving the request was parsed correctly
        """
        # Create a minimal FastAPI app with just the webhook router
        app = FastAPI()
        app.include_router(webhook_router, prefix='/api/v1')

        client = TestClient(app, raise_server_exceptions=False)

        # Create a valid request body with conversation_id in it
        conversation_id = str(uuid4())
        request_body = {
            'id': conversation_id,
            'execution_status': 'running',
            'agent': {
                'llm': {
                    'model': 'gpt-4',
                },
            },
            'stats': {
                'usage_to_metrics': {},
            },
        }

        # POST to /webhooks/conversations WITHOUT any query parameters
        # If the old bug existed (conversation_id required as query param),
        # FastAPI would return 422 Unprocessable Entity
        response = client.post(
            '/api/v1/webhooks/conversations',
            json=request_body,
            # No X-Session-API-Key header - should fail auth but NOT validation
        )

        # We expect 401 Unauthorized (missing session API key)
        # NOT 422 Unprocessable Entity (which would indicate conversation_id
        # was incorrectly required as a query parameter)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED, (
            f'Expected 401 (auth failure), got {response.status_code}. '
            f'If 422, the endpoint incorrectly requires conversation_id as query param. '
            f'Response: {response.json()}'
        )
        assert response.json()['detail'] == 'X-Session-API-Key header is required'

    def test_events_endpoint_still_requires_conversation_id_in_path(self):
        """Test that /webhooks/events/{conversation_id} correctly requires path param.

        This ensures we didn't accidentally break the events endpoint which legitimately
        requires conversation_id as a path parameter.
        """
        # Create a minimal FastAPI app with just the webhook router
        app = FastAPI()
        app.include_router(webhook_router, prefix='/api/v1')

        client = TestClient(app, raise_server_exceptions=False)

        conversation_id = str(uuid4())
        request_body = []  # Empty events list

        # POST to /webhooks/events/{conversation_id} with path parameter
        response = client.post(
            f'/api/v1/webhooks/events/{conversation_id}',
            json=request_body,
            # No X-Session-API-Key header - should fail auth but NOT validation
        )

        # We expect 401 Unauthorized (missing session API key)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()['detail'] == 'X-Session-API-Key header is required'
