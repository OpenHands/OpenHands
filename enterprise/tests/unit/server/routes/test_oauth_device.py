"""Unit tests for OAuth2 Device Flow endpoints."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from server.routes.oauth_device import (
    DeviceTokenRequest,
    device_authorization,
    device_token,
    device_verification_authenticated,
)
from storage.device_code import DeviceCode


@pytest.fixture
def mock_device_code_store():
    """Mock device code store."""
    return MagicMock()


@pytest.fixture
def mock_api_key_store():
    """Mock API key store."""
    return MagicMock()


@pytest.fixture
def mock_token_manager():
    """Mock token manager."""
    return MagicMock()


@pytest.fixture
def mock_request():
    """Mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.base_url = 'https://test.example.com/'
    return request


class TestDeviceAuthorization:
    """Test device authorization endpoint."""

    @patch('server.routes.oauth_device.device_code_store')
    async def test_device_authorization_success(self, mock_store, mock_request):
        """Test successful device authorization."""
        mock_device = DeviceCode(
            device_code='test-device-code-123',
            user_code='ABC12345',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        )
        mock_store.create_device_code.return_value = mock_device

        result = await device_authorization(mock_request)

        assert result.device_code == 'test-device-code-123'
        assert result.user_code == 'ABC12345'
        assert result.expires_in == 600
        assert result.interval == 5
        assert 'verify' in result.verification_uri
        assert 'ABC12345' in result.verification_uri_complete


class TestDeviceToken:
    """Test device token endpoint."""

    @pytest.mark.parametrize(
        'device_exists,status,expected_error',
        [
            (False, None, 'invalid_grant'),
            (True, 'expired', 'expired_token'),
            (True, 'denied', 'access_denied'),
            (True, 'pending', 'authorization_pending'),
        ],
    )
    @patch('server.routes.oauth_device.device_code_store')
    async def test_device_token_error_cases(
        self, mock_store, device_exists, status, expected_error
    ):
        """Test various error cases for device token endpoint."""
        request = DeviceTokenRequest(device_code='test-device-code')

        if device_exists:
            mock_device = MagicMock()
            mock_device.is_expired.return_value = status == 'expired'
            mock_device.status = status
            mock_store.get_by_device_code.return_value = mock_device
        else:
            mock_store.get_by_device_code.return_value = None

        result = await device_token(request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        # Check error in response content
        content = result.body.decode()
        assert expected_error in content

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_device_token_success(self, mock_store, mock_api_key_class):
        """Test successful device token retrieval."""
        request = DeviceTokenRequest(device_code='test-device-code')

        # Mock authorized device
        mock_device = MagicMock()
        mock_device.is_expired.return_value = False
        mock_device.status = 'authorized'
        mock_device.keycloak_user_id = 'user-123'
        mock_store.get_by_device_code.return_value = mock_device

        # Mock API key retrieval
        mock_api_key_store = MagicMock()
        mock_api_key_store.retrieve_api_key_by_name.return_value = 'test-api-key'
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        result = await device_token(request)

        assert result.access_token == 'test-api-key'
        assert result.token_type == 'Bearer'


class TestDeviceVerificationAuthenticated:
    """Test device verification authenticated endpoint."""

    @patch('openhands.server.user_auth.user_auth.get_user_auth')
    async def test_verification_missing_user_code(self, mock_get_user_auth):
        """Test verification with missing user code."""
        mock_request = MagicMock()
        mock_request.form = AsyncMock(return_value={'other_field': 'value'})

        with pytest.raises(HTTPException):
            await device_verification_authenticated(mock_request)

    @patch('openhands.server.user_auth.user_auth.get_user_auth')
    async def test_verification_unauthenticated_user(self, mock_get_user_auth):
        """Test verification with unauthenticated user."""
        mock_request = MagicMock()
        mock_request.form = AsyncMock(return_value={'user_code': 'ABC12345'})

        mock_user_auth = AsyncMock()
        mock_user_auth.get_user_id = AsyncMock(return_value=None)
        mock_get_user_auth.return_value = mock_user_auth

        with pytest.raises(HTTPException):
            await device_verification_authenticated(mock_request)

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    @patch('openhands.server.user_auth.user_auth.get_user_auth')
    async def test_verification_invalid_device_code(
        self, mock_get_user_auth, mock_store, mock_api_key_class
    ):
        """Test verification with invalid device code."""
        mock_request = MagicMock()
        mock_request.form = AsyncMock(return_value={'user_code': 'INVALID'})

        mock_user_auth = AsyncMock()
        mock_user_auth.get_user_id = AsyncMock(return_value='user-123')
        mock_get_user_auth.return_value = mock_user_auth

        mock_store.get_by_user_code.return_value = None

        with pytest.raises(HTTPException):
            await device_verification_authenticated(mock_request)

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    @patch('openhands.server.user_auth.user_auth.get_user_auth')
    async def test_verification_already_processed(
        self, mock_get_user_auth, mock_store, mock_api_key_class
    ):
        """Test verification with already processed device code."""
        mock_request = MagicMock()
        mock_request.form = AsyncMock(return_value={'user_code': 'ABC12345'})

        mock_user_auth = AsyncMock()
        mock_user_auth.get_user_id = AsyncMock(return_value='user-123')
        mock_get_user_auth.return_value = mock_user_auth

        mock_device = MagicMock()
        mock_device.is_pending.return_value = False
        mock_store.get_by_user_code.return_value = mock_device

        with pytest.raises(HTTPException):
            await device_verification_authenticated(mock_request)

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    @patch('openhands.server.user_auth.user_auth.get_user_auth')
    async def test_verification_success(
        self, mock_get_user_auth, mock_store, mock_api_key_class
    ):
        """Test successful device verification."""
        mock_request = MagicMock()
        mock_request.form = AsyncMock(return_value={'user_code': 'ABC12345'})

        mock_user_auth = AsyncMock()
        mock_user_auth.get_user_id = AsyncMock(return_value='user-123')
        mock_get_user_auth.return_value = mock_user_auth

        # Mock device code
        mock_device = MagicMock()
        mock_device.is_pending.return_value = True
        mock_store.get_by_user_code.return_value = mock_device
        mock_store.authorize_device_code.return_value = True

        # Mock API key store
        mock_api_key_store = MagicMock()
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        result = await device_verification_authenticated(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        mock_api_key_store.delete_api_key_by_name.assert_called_once_with(
            'user-123', 'CLI Authentication'
        )
        mock_api_key_store.create_api_key.assert_called_once()
        mock_store.authorize_device_code.assert_called_once_with(
            user_code='ABC12345', user_id='user-123'
        )
