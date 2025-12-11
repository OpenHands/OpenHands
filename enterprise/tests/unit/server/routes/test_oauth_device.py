"""Unit tests for OAuth2 Device Flow endpoints."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from server.routes.oauth_device import (
    DeviceTokenRequest,
    device_authorization,
    device_token,
    device_verification_page,
    keycloak_callback,
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
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
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


class TestDeviceVerification:
    """Test device verification endpoint."""

    async def test_verification_page_no_code(self):
        """Test verification page without user code."""
        result = await device_verification_page()

        assert isinstance(result, HTMLResponse)
        assert 'Device Authorization' in result.body.decode()
        assert 'user_code' in result.body.decode()

    @patch('server.routes.oauth_device.device_code_store')
    async def test_verification_page_invalid_code(self, mock_store):
        """Test verification page with invalid user code."""
        mock_store.get_by_user_code.return_value = None

        result = await device_verification_page(user_code='INVALID')

        assert isinstance(result, HTMLResponse)
        assert result.status_code == 400
        assert 'Invalid or expired' in result.body.decode()

    @patch('server.routes.oauth_device.config')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_verification_page_valid_code(self, mock_store, mock_config):
        """Test verification page with valid user code."""
        mock_device = MagicMock()
        mock_store.get_by_user_code.return_value = mock_device
        mock_config.jwt_secret.get_secret_value.return_value = 'test-secret'

        with patch('server.routes.oauth_device.jwt.encode', return_value='test-jwt'):
            result = await device_verification_page(user_code='ABC12345')

        assert result.status_code in (302, 307)  # Redirect
        assert 'keycloak' in result.headers['location']


class TestKeycloakCallback:
    """Test Keycloak callback endpoint."""

    @pytest.mark.parametrize(
        'code,error,expected_status',
        [
            ('', 'access_denied', 400),
            ('valid-code', 'server_error', 400),
            ('', '', 400),  # No code or error
        ],
    )
    async def test_keycloak_callback_errors(self, code, error, expected_status):
        """Test Keycloak callback error cases."""
        mock_request = MagicMock()

        result = await keycloak_callback(mock_request, code=code, error=error)

        assert isinstance(result, HTMLResponse)
        assert result.status_code == expected_status

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    @patch('server.routes.oauth_device.token_manager')
    @patch('server.routes.oauth_device.config')
    async def test_keycloak_callback_success(
        self, mock_config, mock_token_mgr, mock_store, mock_api_key_class
    ):
        """Test successful Keycloak callback."""
        mock_request = MagicMock()

        # Mock JWT decoding
        mock_config.jwt_secret.get_secret_value.return_value = 'test-secret'

        # Mock token manager
        mock_token_mgr.get_keycloak_tokens = AsyncMock(
            return_value=('access-token', 'refresh-token')
        )
        mock_token_mgr.get_user_info = AsyncMock(return_value={'sub': 'user-123'})

        # Mock device code
        mock_device = MagicMock()
        mock_device.is_pending.return_value = True
        mock_store.get_by_user_code.return_value = mock_device
        mock_store.authorize_device_code.return_value = True

        # Mock API key creation
        mock_api_key_store = MagicMock()
        mock_api_key_store.create_api_key.return_value = 'new-api-key'
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        with patch(
            'server.routes.oauth_device.jwt.decode',
            return_value={'user_code': 'ABC12345'},
        ):
            result = await keycloak_callback(
                mock_request, code='auth-code', state='jwt-state'
            )

        assert isinstance(result, HTMLResponse)
        assert result.status_code == 200
        assert 'Success!' in result.body.decode()
        mock_api_key_store.delete_api_key_by_name.assert_called_once()
        mock_api_key_store.create_api_key.assert_called_once()
