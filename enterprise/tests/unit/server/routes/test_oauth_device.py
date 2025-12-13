"""Unit tests for OAuth2 Device Flow endpoints."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

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
        mock_device.user_code = (
            'ABC12345'  # Add user_code for device-specific API key lookup
        )
        mock_store.get_by_device_code.return_value = mock_device

        # Mock API key retrieval
        mock_api_key_store = MagicMock()
        mock_api_key_store.retrieve_api_key_by_name.return_value = 'test-api-key'
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        result = await device_token(request)

        assert result.access_token == 'test-api-key'
        assert result.token_type == 'Bearer'

        # Verify that the correct device-specific API key name was used
        mock_api_key_store.retrieve_api_key_by_name.assert_called_once_with(
            'user-123', 'Device Link Access Key (ABC12345)'
        )


class TestDeviceVerificationAuthenticated:
    """Test device verification authenticated endpoint."""

    async def test_verification_unauthenticated_user(self):
        """Test verification with unauthenticated user."""
        with pytest.raises(HTTPException):
            await device_verification_authenticated(user_code='ABC12345', user_id=None)

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_verification_invalid_device_code(
        self, mock_store, mock_api_key_class
    ):
        """Test verification with invalid device code."""
        mock_store.get_by_user_code.return_value = None

        with pytest.raises(HTTPException):
            await device_verification_authenticated(
                user_code='INVALID', user_id='user-123'
            )

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_verification_already_processed(self, mock_store, mock_api_key_class):
        """Test verification with already processed device code."""
        mock_device = MagicMock()
        mock_device.is_pending.return_value = False
        mock_store.get_by_user_code.return_value = mock_device

        with pytest.raises(HTTPException):
            await device_verification_authenticated(
                user_code='ABC12345', user_id='user-123'
            )

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_verification_success(self, mock_store, mock_api_key_class):
        """Test successful device verification."""
        # Mock device code
        mock_device = MagicMock()
        mock_device.is_pending.return_value = True
        mock_store.get_by_user_code.return_value = mock_device
        mock_store.authorize_device_code.return_value = True

        # Mock API key store
        mock_api_key_store = MagicMock()
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        result = await device_verification_authenticated(
            user_code='ABC12345', user_id='user-123'
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        # Should NOT delete existing API keys (multiple devices allowed)
        mock_api_key_store.delete_api_key_by_name.assert_not_called()
        # Should create a new API key with device-specific name
        mock_api_key_store.create_api_key.assert_called_once()
        call_args = mock_api_key_store.create_api_key.call_args
        assert call_args[1]['name'] == 'Device Link Access Key (ABC12345)'
        mock_store.authorize_device_code.assert_called_once_with(
            user_code='ABC12345', user_id='user-123'
        )

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_multiple_device_authentication(self, mock_store, mock_api_key_class):
        """Test that multiple devices can authenticate simultaneously."""
        # Mock API key store
        mock_api_key_store = MagicMock()
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        # Simulate two different devices
        device1_code = 'ABC12345'
        device2_code = 'XYZ67890'
        user_id = 'user-123'

        # Mock device codes
        mock_device1 = MagicMock()
        mock_device1.is_pending.return_value = True
        mock_device2 = MagicMock()
        mock_device2.is_pending.return_value = True

        # Configure mock store to return appropriate device for each user_code
        def get_by_user_code_side_effect(user_code):
            if user_code == device1_code:
                return mock_device1
            elif user_code == device2_code:
                return mock_device2
            return None

        mock_store.get_by_user_code.side_effect = get_by_user_code_side_effect
        mock_store.authorize_device_code.return_value = True

        # Authenticate first device
        result1 = await device_verification_authenticated(
            user_code=device1_code, user_id=user_id
        )

        # Authenticate second device
        result2 = await device_verification_authenticated(
            user_code=device2_code, user_id=user_id
        )

        # Both should succeed
        assert isinstance(result1, JSONResponse)
        assert result1.status_code == 200
        assert isinstance(result2, JSONResponse)
        assert result2.status_code == 200

        # Should create two separate API keys with different names
        assert mock_api_key_store.create_api_key.call_count == 2

        # Check that each device got a unique API key name
        call_args_list = mock_api_key_store.create_api_key.call_args_list
        device1_name = call_args_list[0][1]['name']
        device2_name = call_args_list[1][1]['name']

        assert device1_name == f'Device Link Access Key ({device1_code})'
        assert device2_name == f'Device Link Access Key ({device2_code})'
        assert device1_name != device2_name  # Ensure they're different

        # Should NOT delete any existing API keys
        mock_api_key_store.delete_api_key_by_name.assert_not_called()
