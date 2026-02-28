"""Unit tests for AuthTokenStore."""

import time
from contextlib import asynccontextmanager
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.auth.auth_error import TokenRefreshError
from sqlalchemy.exc import OperationalError
from storage.auth_token_store import (
    ACCESS_TOKEN_EXPIRY_BUFFER,
    LOCK_TIMEOUT_SECONDS,
    AuthTokenStore,
)

from openhands.integrations.service_types import ProviderType


def create_mock_session():
    """Create a mock async session with properly configured context managers."""
    session = AsyncMock()

    # Create async context manager for begin()
    @asynccontextmanager
    async def begin_context():
        yield

    session.begin = begin_context
    return session


def create_mock_session_maker(mock_session):
    """Create a mock async session maker."""

    @asynccontextmanager
    async def session_context():
        yield mock_session

    # Return a callable that returns the context manager
    return lambda: session_context()


@pytest.fixture
def mock_session():
    """Create mock async session."""
    return create_mock_session()


@pytest.fixture
def mock_session_maker(mock_session):
    """Create mock async session maker."""
    return create_mock_session_maker(mock_session)


@pytest.fixture
def auth_token_store(mock_session_maker):
    """Create AuthTokenStore instance with mocked session maker."""
    return AuthTokenStore(
        keycloak_user_id='test-user-123',
        idp=ProviderType.GITHUB,
        a_session_maker=mock_session_maker,
    )


class TestIsTokenExpired:
    """Tests for _is_token_expired method."""

    def test_both_tokens_valid(self, auth_token_store):
        """Test when both tokens are valid (not expired)."""
        current_time = int(time.time())
        access_expires = current_time + ACCESS_TOKEN_EXPIRY_BUFFER + 1000
        refresh_expires = current_time + 1000

        access_expired, refresh_expired = auth_token_store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is False
        assert refresh_expired is False

    def test_access_token_expired(self, auth_token_store):
        """Test when access token is expired but within buffer."""
        current_time = int(time.time())
        # Access token expires within buffer period
        access_expires = current_time + ACCESS_TOKEN_EXPIRY_BUFFER - 100
        refresh_expires = current_time + 10000

        access_expired, refresh_expired = auth_token_store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is True
        assert refresh_expired is False

    def test_refresh_token_expired(self, auth_token_store):
        """Test when refresh token is expired."""
        current_time = int(time.time())
        access_expires = current_time + ACCESS_TOKEN_EXPIRY_BUFFER + 1000
        refresh_expires = current_time - 100  # Already expired

        access_expired, refresh_expired = auth_token_store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is False
        assert refresh_expired is True

    def test_both_tokens_expired(self, auth_token_store):
        """Test when both tokens are expired."""
        current_time = int(time.time())
        access_expires = current_time - 100
        refresh_expires = current_time - 100

        access_expired, refresh_expired = auth_token_store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is True
        assert refresh_expired is True

    def test_zero_expiration_treated_as_never_expires(self, auth_token_store):
        """Test that 0 expiration time is treated as never expires."""
        access_expired, refresh_expired = auth_token_store._is_token_expired(0, 0)

        assert access_expired is False
        assert refresh_expired is False


class TestLoadTokensFastPath:
    """Tests for load_tokens fast path (no lock needed)."""

    @pytest.mark.asyncio
    async def test_fast_path_token_not_found(
        self, auth_token_store, mock_session_maker, mock_session
    ):
        """Test fast path returns None when no token record exists."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await auth_token_store.load_tokens()

        assert result is None

    @pytest.mark.asyncio
    async def test_fast_path_valid_token_no_refresh_needed(
        self, auth_token_store, mock_session_maker, mock_session
    ):
        """Test fast path returns tokens when they are still valid."""
        current_time = int(time.time())
        mock_token = MagicMock()
        mock_token.access_token = 'valid-access-token'
        mock_token.refresh_token = 'valid-refresh-token'
        mock_token.access_token_expires_at = (
            current_time + ACCESS_TOKEN_EXPIRY_BUFFER + 1000
        )
        mock_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = mock_token
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await auth_token_store.load_tokens()

        assert result is not None
        assert result['access_token'] == 'valid-access-token'
        assert result['refresh_token'] == 'valid-refresh-token'

    @pytest.mark.asyncio
    async def test_fast_path_no_refresh_callback_provided(
        self, auth_token_store, mock_session_maker, mock_session
    ):
        """Test fast path returns existing tokens when no refresh callback is provided."""
        current_time = int(time.time())
        mock_token = MagicMock()
        mock_token.access_token = 'expired-access-token'
        mock_token.refresh_token = 'valid-refresh-token'
        # Expired access token
        mock_token.access_token_expires_at = current_time - 100
        mock_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = mock_token
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await auth_token_store.load_tokens(check_expiration_and_refresh=None)

        assert result is not None
        assert result['access_token'] == 'expired-access-token'


class TestLoadTokensSlowPath:
    """Tests for load_tokens slow path (lock required for refresh)."""

    @pytest.mark.asyncio
    async def test_slow_path_successful_refresh(self):
        """Test slow path successfully refreshes expired tokens."""
        current_time = int(time.time())
        mock_session = create_mock_session()

        # First call (fast path) - returns expired token
        # Second call (slow path) - returns same token for update
        expired_token = MagicMock()
        expired_token.id = 1
        expired_token.access_token = 'expired-access-token'
        expired_token.refresh_token = 'valid-refresh-token'
        expired_token.access_token_expires_at = current_time - 100  # Expired
        expired_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = expired_token
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        async def mock_refresh(
            idp: ProviderType, refresh_token: str, access_exp: int, refresh_exp: int
        ) -> Dict[str, str | int]:
            return {
                'access_token': 'new-access-token',
                'refresh_token': 'new-refresh-token',
                'access_token_expires_at': current_time + 3600,
                'refresh_token_expires_at': current_time + 86400,
            }

        result = await auth_store.load_tokens(check_expiration_and_refresh=mock_refresh)

        assert result is not None
        assert result['access_token'] == 'new-access-token'
        assert result['refresh_token'] == 'new-refresh-token'

    @pytest.mark.asyncio
    async def test_slow_path_double_check_avoids_refresh(self):
        """Test double-check locking: token was refreshed by another request."""
        current_time = int(time.time())
        mock_session = create_mock_session()

        # Simulate scenario:
        # 1. Fast path sees expired token
        # 2. While waiting for lock, another request refreshes
        # 3. Slow path sees fresh token, skips refresh

        call_count = [0]

        def create_token():
            call_count[0] += 1
            token = MagicMock()
            token.id = 1
            token.access_token = 'fresh-access-token'
            token.refresh_token = 'fresh-refresh-token'
            if call_count[0] == 1:
                # First call (fast path) - expired
                token.access_token_expires_at = current_time - 100
            else:
                # Second call (slow path) - already refreshed
                token.access_token_expires_at = (
                    current_time + ACCESS_TOKEN_EXPIRY_BUFFER + 1000
                )
            token.refresh_token_expires_at = current_time + 86400
            return token

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.side_effect = (
            lambda: create_token()
        )
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        refresh_called = [False]

        async def mock_refresh(
            idp: ProviderType, refresh_token: str, access_exp: int, refresh_exp: int
        ) -> Dict[str, str | int]:
            refresh_called[0] = True
            return {
                'access_token': 'should-not-be-used',
                'refresh_token': 'should-not-be-used',
                'access_token_expires_at': current_time + 3600,
                'refresh_token_expires_at': current_time + 86400,
            }

        result = await auth_store.load_tokens(check_expiration_and_refresh=mock_refresh)

        # The refresh callback should not be called because double-check
        # found the token was already refreshed
        assert result is not None
        assert result['access_token'] == 'fresh-access-token'

    @pytest.mark.asyncio
    async def test_slow_path_token_not_found_after_lock(self):
        """Test slow path returns None if token record disappears after lock."""
        current_time = int(time.time())
        mock_session = create_mock_session()

        # First call (fast path) - token exists but expired
        # Second call (slow path with lock) - token no longer exists
        call_count = [0]

        def get_token():
            call_count[0] += 1
            if call_count[0] == 1:
                token = MagicMock()
                token.access_token_expires_at = current_time - 100  # Expired
                token.refresh_token_expires_at = current_time + 10000
                return token
            return None

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.side_effect = get_token
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        async def mock_refresh(*args) -> Dict[str, str | int]:
            return {
                'access_token': 'new-token',
                'refresh_token': 'new-refresh',
                'access_token_expires_at': current_time + 3600,
                'refresh_token_expires_at': current_time + 86400,
            }

        result = await auth_store.load_tokens(check_expiration_and_refresh=mock_refresh)

        assert result is None


class TestLoadTokensLockTimeout:
    """Tests for lock timeout handling."""

    @pytest.mark.asyncio
    async def test_lock_timeout_raises_token_refresh_error(self):
        """Test that lock timeout raises TokenRefreshError."""
        current_time = int(time.time())
        mock_session = create_mock_session()

        # First call (fast path) - returns expired token
        expired_token = MagicMock()
        expired_token.access_token_expires_at = current_time - 100
        expired_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = expired_token

        # First execute for fast path succeeds
        # Second execute (for slow path) raises OperationalError
        call_count = [0]

        async def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return mock_result
            # Simulate lock timeout
            raise OperationalError(
                'canceling statement due to lock timeout', None, None
            )

        mock_session.execute = execute_side_effect

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        async def mock_refresh(*args) -> Dict[str, str | int]:
            return {
                'access_token': 'new-token',
                'refresh_token': 'new-refresh',
                'access_token_expires_at': current_time + 3600,
                'refresh_token_expires_at': current_time + 86400,
            }

        with pytest.raises(TokenRefreshError) as exc_info:
            await auth_store.load_tokens(check_expiration_and_refresh=mock_refresh)

        assert 'lock timeout' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_lock_timeout_preserves_original_exception(self):
        """Test that TokenRefreshError preserves the original OperationalError."""
        current_time = int(time.time())
        mock_session = create_mock_session()

        expired_token = MagicMock()
        expired_token.access_token_expires_at = current_time - 100
        expired_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = expired_token

        original_error = OperationalError(
            'canceling statement due to lock timeout', None, None
        )

        call_count = [0]

        async def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return mock_result
            raise original_error

        mock_session.execute = execute_side_effect

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        async def mock_refresh(*args) -> Dict[str, str | int]:
            return {
                'access_token': 'new-token',
                'refresh_token': 'new-refresh',
                'access_token_expires_at': current_time + 3600,
                'refresh_token_expires_at': current_time + 86400,
            }

        with pytest.raises(TokenRefreshError) as exc_info:
            await auth_store.load_tokens(check_expiration_and_refresh=mock_refresh)

        # Verify the original exception is chained
        assert exc_info.value.__cause__ is original_error


class TestLoadTokensRefreshCallbackBehavior:
    """Tests for refresh callback return values."""

    @pytest.mark.asyncio
    async def test_refresh_callback_returns_none(self):
        """Test behavior when refresh callback returns None (no refresh performed)."""
        current_time = int(time.time())
        mock_session = create_mock_session()

        expired_token = MagicMock()
        expired_token.id = 1
        expired_token.access_token = 'old-access-token'
        expired_token.refresh_token = 'old-refresh-token'
        expired_token.access_token_expires_at = current_time - 100  # Expired
        expired_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = expired_token
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        async def mock_refresh_returns_none(
            idp: ProviderType, refresh_token: str, access_exp: int, refresh_exp: int
        ) -> Dict[str, str | int] | None:
            return None

        result = await auth_store.load_tokens(
            check_expiration_and_refresh=mock_refresh_returns_none
        )

        # Should return the old tokens when refresh returns None
        assert result is not None
        assert result['access_token'] == 'old-access-token'
        assert result['refresh_token'] == 'old-refresh-token'


class TestStoreTokens:
    """Tests for store_tokens method."""

    @pytest.mark.asyncio
    async def test_store_tokens_creates_new_record(self):
        """Test storing tokens when no existing record."""
        mock_session = create_mock_session()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        await auth_store.store_tokens(
            access_token='new-access-token',
            refresh_token='new-refresh-token',
            access_token_expires_at=1234567890,
            refresh_token_expires_at=1234657890,
        )

        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_tokens_updates_existing_record(self):
        """Test storing tokens updates existing record."""
        mock_session = create_mock_session()
        existing_token = MagicMock()
        existing_token.access_token = 'old-access'

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = existing_token
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_session_maker = create_mock_session_maker(mock_session)

        auth_store = AuthTokenStore(
            keycloak_user_id='test-user-123',
            idp=ProviderType.GITHUB,
            a_session_maker=mock_session_maker,
        )

        await auth_store.store_tokens(
            access_token='new-access-token',
            refresh_token='new-refresh-token',
            access_token_expires_at=1234567890,
            refresh_token_expires_at=1234657890,
        )

        assert existing_token.access_token == 'new-access-token'
        assert existing_token.refresh_token == 'new-refresh-token'


class TestIsAccessTokenValid:
    """Tests for is_access_token_valid method."""

    @pytest.mark.asyncio
    async def test_is_access_token_valid_returns_false_when_no_tokens(
        self, auth_token_store, mock_session_maker, mock_session
    ):
        """Test returns False when no tokens found."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await auth_token_store.is_access_token_valid()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_access_token_valid_returns_true_for_valid_token(
        self, auth_token_store, mock_session_maker, mock_session
    ):
        """Test returns True when token is valid."""
        current_time = int(time.time())
        mock_token = MagicMock()
        mock_token.access_token = 'valid-access'
        mock_token.refresh_token = 'valid-refresh'
        mock_token.access_token_expires_at = current_time + 1000
        mock_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = mock_token
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await auth_token_store.is_access_token_valid()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_access_token_valid_returns_false_for_expired_token(
        self, auth_token_store, mock_session_maker, mock_session
    ):
        """Test returns False when token is expired."""
        current_time = int(time.time())
        mock_token = MagicMock()
        mock_token.access_token = 'expired-access'
        mock_token.refresh_token = 'valid-refresh'
        mock_token.access_token_expires_at = current_time - 100  # Expired
        mock_token.refresh_token_expires_at = current_time + 10000

        mock_result = MagicMock()
        mock_result.scalars.return_value.one_or_none.return_value = mock_token
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await auth_token_store.is_access_token_valid()

        assert result is False


class TestGetInstance:
    """Tests for get_instance class method."""

    @pytest.mark.asyncio
    async def test_get_instance_creates_auth_token_store(self):
        """Test get_instance creates an AuthTokenStore with correct params."""
        with patch('storage.auth_token_store.a_session_maker') as mock_a_session_maker:
            store = await AuthTokenStore.get_instance(
                keycloak_user_id='user-123', idp=ProviderType.GITHUB
            )

            assert store.keycloak_user_id == 'user-123'
            assert store.idp == ProviderType.GITHUB
            assert store.a_session_maker is mock_a_session_maker


class TestIdentityProviderValue:
    """Tests for identity_provider_value property."""

    def test_identity_provider_value_returns_idp_value(self, auth_token_store):
        """Test that identity_provider_value returns the enum value."""
        assert auth_token_store.identity_provider_value == ProviderType.GITHUB.value

    def test_identity_provider_value_for_different_providers(self):
        """Test identity_provider_value for different providers."""
        for provider in [
            ProviderType.GITHUB,
            ProviderType.GITLAB,
            ProviderType.BITBUCKET,
        ]:
            store = AuthTokenStore(
                keycloak_user_id='test-user',
                idp=provider,
                a_session_maker=MagicMock(),
            )
            assert store.identity_provider_value == provider.value


class TestConstants:
    """Tests for module constants."""

    def test_access_token_expiry_buffer_value(self):
        """Test ACCESS_TOKEN_EXPIRY_BUFFER is set to 15 minutes."""
        assert ACCESS_TOKEN_EXPIRY_BUFFER == 900

    def test_lock_timeout_seconds_value(self):
        """Test LOCK_TIMEOUT_SECONDS is set to 5 seconds."""
        assert LOCK_TIMEOUT_SECONDS == 5
