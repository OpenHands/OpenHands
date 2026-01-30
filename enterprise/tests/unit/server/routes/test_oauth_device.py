"""Unit tests for OAuth2 Device Flow endpoints."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from server.routes.oauth_device import (
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
    """Mock API key store with async create_api_key."""
    mock = MagicMock()
    mock.create_api_key = AsyncMock()
    return mock


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
            current_interval=5,  # Default interval
        )
        mock_store.create_device_code.return_value = mock_device

        result = await device_authorization(mock_request)

        assert result.device_code == 'test-device-code-123'
        assert result.user_code == 'ABC12345'
        assert result.expires_in == 600
        assert result.interval == 5  # Should match device's current_interval
        assert 'verify' in result.verification_uri
        assert 'ABC12345' in result.verification_uri_complete

    @patch('server.routes.oauth_device.device_code_store')
    async def test_device_authorization_with_increased_interval(
        self, mock_store, mock_request
    ):
        """Test device authorization returns increased interval from rate limiting."""
        mock_device = DeviceCode(
            device_code='test-device-code-456',
            user_code='XYZ98765',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            current_interval=15,  # Increased interval from previous rate limiting
        )
        mock_store.create_device_code.return_value = mock_device

        result = await device_authorization(mock_request)

        assert result.device_code == 'test-device-code-456'
        assert result.user_code == 'XYZ98765'
        assert result.expires_in == 600
        assert result.interval == 15  # Should match device's increased current_interval
        assert 'verify' in result.verification_uri
        assert 'XYZ98765' in result.verification_uri_complete


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
        device_code = 'test-device-code'

        if device_exists:
            mock_device = MagicMock()
            mock_device.is_expired.return_value = status == 'expired'
            mock_device.status = status
            # Mock rate limiting - return False (not too fast) and default interval
            mock_device.check_rate_limit.return_value = (False, 5)
            mock_store.get_by_device_code.return_value = mock_device
            mock_store.update_poll_time.return_value = True
        else:
            mock_store.get_by_device_code.return_value = None

        result = await device_token(device_code=device_code)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        # Check error in response content
        content = result.body.decode()
        assert expected_error in content

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_device_token_success(self, mock_store, mock_api_key_class):
        """Test successful device token retrieval."""
        device_code = 'test-device-code'

        # Mock authorized device
        mock_device = MagicMock()
        mock_device.is_expired.return_value = False
        mock_device.status = 'authorized'
        mock_device.keycloak_user_id = 'user-123'
        mock_device.user_code = (
            'ABC12345'  # Add user_code for device-specific API key lookup
        )
        # Mock rate limiting - return False (not too fast) and default interval
        mock_device.check_rate_limit.return_value = (False, 5)
        mock_store.get_by_device_code.return_value = mock_device
        mock_store.update_poll_time.return_value = True

        # Mock API key retrieval
        mock_api_key_store = MagicMock()
        mock_api_key_store.retrieve_api_key_by_name.return_value = 'test-api-key'
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        result = await device_token(device_code=device_code)

        # Check that result is a DeviceTokenResponse
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

        # Mock API key store with async create_api_key
        mock_api_key_store = MagicMock()
        mock_api_key_store.create_api_key = AsyncMock()
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
        # Mock API key store with async create_api_key
        mock_api_key_store = MagicMock()
        mock_api_key_store.create_api_key = AsyncMock()
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


class TestDeviceTokenRateLimiting:
    """Test rate limiting for device token polling (RFC 8628 section 3.5)."""

    @patch('server.routes.oauth_device.device_code_store')
    async def test_first_poll_allowed(self, mock_store):
        """Test that the first poll is always allowed."""
        # Create a device code with no previous poll time
        mock_device = DeviceCode(
            device_code='test_device_code',
            user_code='ABC123',
            status='pending',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            last_poll_time=None,  # First poll
            current_interval=5,
        )
        mock_store.get_by_device_code.return_value = mock_device
        mock_store.update_poll_time.return_value = True

        device_code = 'test_device_code'
        result = await device_token(device_code=device_code)

        # Should return authorization_pending, not slow_down
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = result.body.decode()
        assert 'authorization_pending' in content
        assert 'slow_down' not in content

        # Should update poll time without increasing interval
        mock_store.update_poll_time.assert_called_with(
            'test_device_code', increase_interval=False
        )

    @patch('server.routes.oauth_device.device_code_store')
    async def test_normal_polling_allowed(self, mock_store):
        """Test that normal polling (respecting interval) is allowed."""
        # Create a device code with last poll time 6 seconds ago (interval is 5)
        last_poll = datetime.now(UTC) - timedelta(seconds=6)
        mock_device = DeviceCode(
            device_code='test_device_code',
            user_code='ABC123',
            status='pending',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            last_poll_time=last_poll,
            current_interval=5,
        )
        mock_store.get_by_device_code.return_value = mock_device
        mock_store.update_poll_time.return_value = True

        device_code = 'test_device_code'
        result = await device_token(device_code=device_code)

        # Should return authorization_pending, not slow_down
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = result.body.decode()
        assert 'authorization_pending' in content
        assert 'slow_down' not in content

        # Should update poll time without increasing interval
        mock_store.update_poll_time.assert_called_with(
            'test_device_code', increase_interval=False
        )

    @patch('server.routes.oauth_device.device_code_store')
    async def test_fast_polling_returns_slow_down(self, mock_store):
        """Test that polling too fast returns slow_down error."""
        # Create a device code with last poll time 2 seconds ago (interval is 5)
        last_poll = datetime.now(UTC) - timedelta(seconds=2)
        mock_device = DeviceCode(
            device_code='test_device_code',
            user_code='ABC123',
            status='pending',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            last_poll_time=last_poll,
            current_interval=5,
        )
        mock_store.get_by_device_code.return_value = mock_device
        mock_store.update_poll_time.return_value = True

        device_code = 'test_device_code'
        result = await device_token(device_code=device_code)

        # Should return slow_down error
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = result.body.decode()
        assert 'slow_down' in content
        assert 'interval' in content
        assert '10' in content  # New interval should be 5 + 5 = 10

        # Should update poll time and increase interval
        mock_store.update_poll_time.assert_called_with(
            'test_device_code', increase_interval=True
        )

    @patch('server.routes.oauth_device.device_code_store')
    async def test_interval_increases_with_repeated_fast_polling(self, mock_store):
        """Test that interval increases with repeated fast polling."""
        # Create a device code with higher current interval from previous slow_down
        last_poll = datetime.now(UTC) - timedelta(seconds=5)  # 5 seconds ago
        mock_device = DeviceCode(
            device_code='test_device_code',
            user_code='ABC123',
            status='pending',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            last_poll_time=last_poll,
            current_interval=15,  # Already increased from previous slow_down
        )
        mock_store.get_by_device_code.return_value = mock_device
        mock_store.update_poll_time.return_value = True

        device_code = 'test_device_code'
        result = await device_token(device_code=device_code)

        # Should return slow_down error with increased interval
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = result.body.decode()
        assert 'slow_down' in content
        assert '20' in content  # New interval should be 15 + 5 = 20

        # Should update poll time and increase interval
        mock_store.update_poll_time.assert_called_with(
            'test_device_code', increase_interval=True
        )

    @patch('server.routes.oauth_device.device_code_store')
    async def test_interval_caps_at_maximum(self, mock_store):
        """Test that interval is capped at maximum value."""
        # Create a device code with interval near maximum
        last_poll = datetime.now(UTC) - timedelta(seconds=30)
        mock_device = DeviceCode(
            device_code='test_device_code',
            user_code='ABC123',
            status='pending',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            last_poll_time=last_poll,
            current_interval=58,  # Near maximum of 60
        )
        mock_store.get_by_device_code.return_value = mock_device
        mock_store.update_poll_time.return_value = True

        device_code = 'test_device_code'
        result = await device_token(device_code=device_code)

        # Should return slow_down error with capped interval
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = result.body.decode()
        assert 'slow_down' in content
        assert '60' in content  # Should be capped at 60, not 63

    @patch('server.routes.oauth_device.device_code_store')
    async def test_rate_limiting_with_authorized_device(self, mock_store):
        """Test that rate limiting still applies to authorized devices."""
        # Create an authorized device code with recent poll
        last_poll = datetime.now(UTC) - timedelta(seconds=2)
        mock_device = DeviceCode(
            device_code='test_device_code',
            user_code='ABC123',
            status='authorized',  # Device is authorized
            keycloak_user_id='user123',
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            last_poll_time=last_poll,
            current_interval=5,
        )
        mock_store.get_by_device_code.return_value = mock_device
        mock_store.update_poll_time.return_value = True

        device_code = 'test_device_code'
        result = await device_token(device_code=device_code)

        # Should still return slow_down error even for authorized device
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = result.body.decode()
        assert 'slow_down' in content

        # Should update poll time and increase interval
        mock_store.update_poll_time.assert_called_with(
            'test_device_code', increase_interval=True
        )


class TestDeviceVerificationTransactionIntegrity:
    """Test transaction integrity for device verification to prevent orphaned API keys."""

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_authorization_failure_prevents_api_key_creation(
        self, mock_store, mock_api_key_class
    ):
        """Test that if device authorization fails, no API key is created."""
        # Mock device code
        mock_device = MagicMock()
        mock_device.is_pending.return_value = True
        mock_store.get_by_user_code.return_value = mock_device
        mock_store.authorize_device_code.return_value = False  # Authorization fails

        # Mock API key store with async create_api_key
        mock_api_key_store = MagicMock()
        mock_api_key_store.create_api_key = AsyncMock()
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        # Should raise HTTPException due to authorization failure
        with pytest.raises(HTTPException) as exc_info:
            await device_verification_authenticated(
                user_code='ABC12345', user_id='user-123'
            )

        assert exc_info.value.status_code == 500
        assert 'Failed to authorize the device' in exc_info.value.detail

        # API key should NOT be created since authorization failed
        mock_api_key_store.create_api_key.assert_not_called()
        mock_store.authorize_device_code.assert_called_once_with(
            user_code='ABC12345', user_id='user-123'
        )

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_api_key_creation_failure_reverts_authorization(
        self, mock_store, mock_api_key_class
    ):
        """Test that if API key creation fails after authorization, the authorization is reverted."""
        # Mock device code
        mock_device = MagicMock()
        mock_device.is_pending.return_value = True
        mock_store.get_by_user_code.return_value = mock_device
        mock_store.authorize_device_code.return_value = True  # Authorization succeeds
        mock_store.deny_device_code.return_value = True  # Cleanup succeeds

        # Mock API key store to fail on creation (async)
        mock_api_key_store = MagicMock()
        mock_api_key_store.create_api_key = AsyncMock(
            side_effect=Exception('Database error')
        )
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        # Should raise HTTPException due to API key creation failure
        with pytest.raises(HTTPException) as exc_info:
            await device_verification_authenticated(
                user_code='ABC12345', user_id='user-123'
            )

        assert exc_info.value.status_code == 500
        assert 'Failed to create API key for device access' in exc_info.value.detail

        # Authorization should have been attempted first
        mock_store.authorize_device_code.assert_called_once_with(
            user_code='ABC12345', user_id='user-123'
        )

        # API key creation should have been attempted after authorization
        mock_api_key_store.create_api_key.assert_called_once()

        # Authorization should be reverted due to API key creation failure
        mock_store.deny_device_code.assert_called_once_with('ABC12345')

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_api_key_creation_failure_cleanup_failure_logged(
        self, mock_store, mock_api_key_class
    ):
        """Test that cleanup failure is logged but doesn't prevent the main error from being raised."""
        # Mock device code
        mock_device = MagicMock()
        mock_device.is_pending.return_value = True
        mock_store.get_by_user_code.return_value = mock_device
        mock_store.authorize_device_code.return_value = True  # Authorization succeeds
        mock_store.deny_device_code.side_effect = Exception(
            'Cleanup failed'
        )  # Cleanup fails

        # Mock API key store to fail on creation (async)
        mock_api_key_store = MagicMock()
        mock_api_key_store.create_api_key = AsyncMock(
            side_effect=Exception('Database error')
        )
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        # Should still raise HTTPException for the original API key creation failure
        with pytest.raises(HTTPException) as exc_info:
            await device_verification_authenticated(
                user_code='ABC12345', user_id='user-123'
            )

        assert exc_info.value.status_code == 500
        assert 'Failed to create API key for device access' in exc_info.value.detail

        # Both operations should have been attempted
        mock_store.authorize_device_code.assert_called_once()
        mock_api_key_store.create_api_key.assert_called_once()
        mock_store.deny_device_code.assert_called_once_with('ABC12345')

    @patch('server.routes.oauth_device.ApiKeyStore')
    @patch('server.routes.oauth_device.device_code_store')
    async def test_successful_flow_creates_api_key_after_authorization(
        self, mock_store, mock_api_key_class
    ):
        """Test that in the successful flow, API key is created only after authorization."""
        # Mock device code
        mock_device = MagicMock()
        mock_device.is_pending.return_value = True
        mock_store.get_by_user_code.return_value = mock_device
        mock_store.authorize_device_code.return_value = True  # Authorization succeeds

        # Mock API key store with async create_api_key
        mock_api_key_store = MagicMock()
        mock_api_key_store.create_api_key = AsyncMock()
        mock_api_key_class.get_instance.return_value = mock_api_key_store

        result = await device_verification_authenticated(
            user_code='ABC12345', user_id='user-123'
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 200

        # Verify the order: authorization first, then API key creation
        mock_store.authorize_device_code.assert_called_once_with(
            user_code='ABC12345', user_id='user-123'
        )
        mock_api_key_store.create_api_key.assert_called_once()

        # No cleanup should be needed in successful case
        mock_store.deny_device_code.assert_not_called()
