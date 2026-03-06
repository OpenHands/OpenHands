"""Unit tests for AuthTokenStore using SQLite in-memory database."""

import time
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.auth_token_store import (
    ACCESS_TOKEN_EXPIRY_BUFFER,
    LOCK_TIMEOUT_SECONDS,
    AuthTokenStore,
)
from storage.auth_tokens import AuthTokens
from storage.base import Base

from openhands.integrations.service_types import ProviderType


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
    )
    return engine


@pytest.fixture
async def async_session_maker(async_engine):
    """Create an async session maker bound to the async engine."""
    async_session_maker = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    # Create all tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_session_maker


class TestIsTokenExpired:
    """Tests for _is_token_expired method."""

    def test_both_tokens_valid(self):
        """Test when both tokens are valid (not expired)."""
        store = AuthTokenStore(
            keycloak_user_id='test-user',
            idp=ProviderType.GITHUB,
        )
        current_time = int(time.time())
        access_expires = current_time + ACCESS_TOKEN_EXPIRY_BUFFER + 1000
        refresh_expires = current_time + 1000

        access_expired, refresh_expired = store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is False
        assert refresh_expired is False

    def test_access_token_expired(self):
        """Test when access token is expired but within buffer."""
        store = AuthTokenStore(
            keycloak_user_id='test-user',
            idp=ProviderType.GITHUB,
        )
        current_time = int(time.time())
        # Access token expires within buffer period
        access_expires = current_time + ACCESS_TOKEN_EXPIRY_BUFFER - 100
        refresh_expires = current_time + 10000

        access_expired, refresh_expired = store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is True
        assert refresh_expired is False

    def test_refresh_token_expired(self):
        """Test when refresh token is expired."""
        store = AuthTokenStore(
            keycloak_user_id='test-user',
            idp=ProviderType.GITHUB,
        )
        current_time = int(time.time())
        access_expires = current_time + ACCESS_TOKEN_EXPIRY_BUFFER + 1000
        refresh_expires = current_time - 100  # Already expired

        access_expired, refresh_expired = store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is False
        assert refresh_expired is True

    def test_both_tokens_expired(self):
        """Test when both tokens are expired."""
        store = AuthTokenStore(
            keycloak_user_id='test-user',
            idp=ProviderType.GITHUB,
        )
        current_time = int(time.time())
        access_expires = current_time - 100
        refresh_expires = current_time - 100

        access_expired, refresh_expired = store._is_token_expired(
            access_expires, refresh_expires
        )

        assert access_expired is True
        assert refresh_expired is True

    def test_zero_expiration_treated_as_never_expires(self):
        """Test that 0 expiration time is treated as never expires."""
        store = AuthTokenStore(
            keycloak_user_id='test-user',
            idp=ProviderType.GITHUB,
        )
        access_expired, refresh_expired = store._is_token_expired(0, 0)

        assert access_expired is False
        assert refresh_expired is False


class TestLoadTokensFastPath:
    """Tests for load_tokens fast path (no lock needed)."""

    @pytest.mark.asyncio
    async def test_fast_path_token_not_found(self, async_session_maker):
        """Test fast path returns None when no token record exists."""
        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            result = await store.load_tokens()

            assert result is None

    @pytest.mark.asyncio
    async def test_fast_path_valid_token_no_refresh_needed(self, async_session_maker):
        """Test fast path returns tokens when they are still valid."""
        current_time = int(time.time())

        # First, store a valid token in the database
        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            await store.store_tokens(
                access_token='valid-access-token',
                refresh_token='valid-refresh-token',
                access_token_expires_at=current_time
                + ACCESS_TOKEN_EXPIRY_BUFFER
                + 1000,
                refresh_token_expires_at=current_time + 10000,
            )

            # Now load tokens - should return valid tokens without refresh
            result = await store.load_tokens()

            assert result is not None
            assert result['access_token'] == 'valid-access-token'
            assert result['refresh_token'] == 'valid-refresh-token'

    @pytest.mark.asyncio
    async def test_fast_path_no_refresh_callback_provided(self, async_session_maker):
        """Test fast path returns existing tokens when no refresh callback is provided."""
        current_time = int(time.time())

        # Store expired access token
        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            await store.store_tokens(
                access_token='expired-access-token',
                refresh_token='valid-refresh-token',
                access_token_expires_at=current_time - 100,  # Expired
                refresh_token_expires_at=current_time + 10000,
            )

            # Load without refresh callback - should still return tokens
            result = await store.load_tokens(check_expiration_and_refresh=None)

            assert result is not None
            assert result['access_token'] == 'expired-access-token'


class TestLoadTokensSlowPath:
    """Tests for load_tokens slow path (lock required for refresh).

    Note: These tests require PostgreSQL's lock_timeout feature which is not
    available in SQLite. The slow path tests are skipped when using SQLite.
    """

    @pytest.mark.skip(reason='SQLite does not support PostgreSQL lock_timeout syntax')
    @pytest.mark.asyncio
    async def test_slow_path_successful_refresh(self, async_session_maker):
        """Test slow path successfully refreshes expired tokens."""
        pass

    @pytest.mark.skip(reason='SQLite does not support PostgreSQL lock_timeout syntax')
    @pytest.mark.asyncio
    async def test_refresh_callback_returns_none(self, async_session_maker):
        """Test behavior when refresh callback returns None (no refresh performed)."""
        pass

    @pytest.mark.asyncio
    async def test_slow_path_double_check_avoids_refresh(self, async_session_maker):
        """Test double-check pattern avoids unnecessary refresh."""
        current_time = int(time.time())

        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            # Store a token that will be valid when second check happens
            await store.store_tokens(
                access_token='original-access-token',
                refresh_token='valid-refresh-token',
                access_token_expires_at=current_time
                + ACCESS_TOKEN_EXPIRY_BUFFER
                + 1000,
                refresh_token_expires_at=current_time + 10000,
            )

            # Load with refresh callback - should NOT refresh since token is valid
            result = await store.load_tokens()

            assert result is not None
            assert result['access_token'] == 'original-access-token'


class TestStoreTokens:
    """Tests for store_tokens method."""

    @pytest.mark.asyncio
    async def test_store_tokens_creates_new_record(self, async_session_maker):
        """Test storing tokens when no existing record."""
        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            await store.store_tokens(
                access_token='new-access-token',
                refresh_token='new-refresh-token',
                access_token_expires_at=1234567890,
                refresh_token_expires_at=1234657890,
            )

            # Verify the token was stored
            async with async_session_maker() as session:
                result = await session.execute(
                    select(AuthTokens).where(
                        AuthTokens.keycloak_user_id == 'test-user-123',
                        AuthTokens.identity_provider == ProviderType.GITHUB.value,
                    )
                )
                token_record = result.scalars().first()
                assert token_record is not None
                assert token_record.access_token == 'new-access-token'
                assert token_record.refresh_token == 'new-refresh-token'

    @pytest.mark.asyncio
    async def test_store_tokens_updates_existing_record(self, async_session_maker):
        """Test storing tokens updates existing record."""
        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            # First, create a token record
            await store.store_tokens(
                access_token='old-access-token',
                refresh_token='old-refresh-token',
                access_token_expires_at=1234567890,
                refresh_token_expires_at=1234657890,
            )

            # Now update it
            await store.store_tokens(
                access_token='new-access-token',
                refresh_token='new-refresh-token',
                access_token_expires_at=1234567891,
                refresh_token_expires_at=1234657891,
            )

            # Verify the token was updated
            async with async_session_maker() as session:
                result = await session.execute(
                    select(AuthTokens).where(
                        AuthTokens.keycloak_user_id == 'test-user-123',
                        AuthTokens.identity_provider == ProviderType.GITHUB.value,
                    )
                )
                token_record = result.scalars().first()
                assert token_record is not None
                assert token_record.access_token == 'new-access-token'
                assert token_record.refresh_token == 'new-refresh-token'


class TestIsAccessTokenValid:
    """Tests for is_access_token_valid method."""

    @pytest.mark.asyncio
    async def test_is_access_token_valid_returns_false_when_no_tokens(
        self, async_session_maker
    ):
        """Test returns False when no tokens found."""
        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            result = await store.is_access_token_valid()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_access_token_valid_returns_true_for_valid_token(
        self, async_session_maker
    ):
        """Test returns True when token is valid."""
        current_time = int(time.time())

        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            await store.store_tokens(
                access_token='valid-access',
                refresh_token='valid-refresh',
                access_token_expires_at=current_time + 1000,
                refresh_token_expires_at=current_time + 10000,
            )

            result = await store.is_access_token_valid()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_access_token_valid_returns_false_for_expired_token(
        self, async_session_maker
    ):
        """Test returns False when token is expired."""
        current_time = int(time.time())

        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = AuthTokenStore(
                keycloak_user_id='test-user-123',
                idp=ProviderType.GITHUB,
            )

            await store.store_tokens(
                access_token='expired-access',
                refresh_token='valid-refresh',
                access_token_expires_at=current_time - 100,  # Expired
                refresh_token_expires_at=current_time + 10000,
            )

            result = await store.is_access_token_valid()

            assert result is False


class TestGetInstance:
    """Tests for get_instance class method."""

    @pytest.mark.asyncio
    async def test_get_instance_creates_auth_token_store(self, async_session_maker):
        """Test get_instance creates an AuthTokenStore with correct params."""
        with patch('storage.auth_token_store.a_session_maker', async_session_maker):
            store = await AuthTokenStore.get_instance(
                keycloak_user_id='user-123', idp=ProviderType.GITHUB
            )

            assert store.keycloak_user_id == 'user-123'
            assert store.idp == ProviderType.GITHUB


class TestIdentityProviderValue:
    """Tests for identity_provider_value property."""

    def test_identity_provider_value_returns_idp_value(self):
        """Test that identity_provider_value returns the enum value."""
        store = AuthTokenStore(
            keycloak_user_id='test-user',
            idp=ProviderType.GITHUB,
        )
        assert store.identity_provider_value == ProviderType.GITHUB.value

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
