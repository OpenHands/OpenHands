"""Unit tests for DeviceCode model."""

from datetime import datetime, timedelta, timezone

import pytest
from storage.device_code import DeviceCode, DeviceCodeStatus


class TestDeviceCode:
    """Test cases for DeviceCode model."""

    @pytest.fixture
    def device_code(self):
        """Create a test device code."""
        return DeviceCode(
            device_code='test-device-code-123',
            user_code='ABC12345',
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )

    @pytest.mark.parametrize(
        'expires_delta,expected',
        [
            (timedelta(minutes=5), False),  # Future expiry
            (timedelta(minutes=-5), True),  # Past expiry
            (timedelta(seconds=1), False),  # Just future (not expired)
        ],
    )
    def test_is_expired(self, expires_delta, expected):
        """Test expiration check with various time deltas."""
        device_code = DeviceCode(
            device_code='test-device-code',
            user_code='ABC12345',
            expires_at=datetime.now(timezone.utc) + expires_delta,
        )
        assert device_code.is_expired() == expected

    @pytest.mark.parametrize(
        'status,expired,expected',
        [
            (DeviceCodeStatus.PENDING.value, False, True),
            (DeviceCodeStatus.PENDING.value, True, False),
            (DeviceCodeStatus.AUTHORIZED.value, False, False),
            (DeviceCodeStatus.DENIED.value, False, False),
        ],
    )
    def test_is_pending(self, status, expired, expected):
        """Test pending status check."""
        expires_at = (
            datetime.now(timezone.utc) - timedelta(minutes=1)
            if expired
            else datetime.now(timezone.utc) + timedelta(minutes=10)
        )
        device_code = DeviceCode(
            device_code='test-device-code',
            user_code='ABC12345',
            status=status,
            expires_at=expires_at,
        )
        assert device_code.is_pending() == expected

    def test_authorize(self, device_code):
        """Test device authorization."""
        user_id = 'test-user-123'

        device_code.authorize(user_id)

        assert device_code.status == DeviceCodeStatus.AUTHORIZED.value
        assert device_code.keycloak_user_id == user_id
        assert device_code.authorized_at is not None
        assert isinstance(device_code.authorized_at, datetime)

    @pytest.mark.parametrize(
        'method,expected_status',
        [
            ('deny', DeviceCodeStatus.DENIED.value),
            ('expire', DeviceCodeStatus.EXPIRED.value),
        ],
    )
    def test_status_changes(self, device_code, method, expected_status):
        """Test status change methods."""
        getattr(device_code, method)()
        assert device_code.status == expected_status
