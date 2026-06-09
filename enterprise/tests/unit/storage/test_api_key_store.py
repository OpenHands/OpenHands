"""Unit tests for ApiKeyStore system key functionality."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from storage.api_key import ApiKey
from storage.api_key_store import ApiKeyStore


@pytest.fixture
def api_key_store():
    """Create ApiKeyStore instance."""
    return ApiKeyStore()


class TestApiKeyStoreSystemKeys:
    """Test cases for system API key functionality."""

    def test_is_system_key_name_with_prefix(self, api_key_store):
        """Test that names with __SYSTEM__: prefix are identified as system keys."""
        assert api_key_store.is_system_key_name('__SYSTEM__:automation') is True
        assert api_key_store.is_system_key_name('__SYSTEM__:test-key') is True
        assert api_key_store.is_system_key_name('__SYSTEM__:') is True

    def test_is_system_key_name_without_prefix(self, api_key_store):
        """Test that names without __SYSTEM__: prefix are not system keys."""
        assert api_key_store.is_system_key_name('my-key') is False
        assert api_key_store.is_system_key_name('automation') is False
        assert api_key_store.is_system_key_name('MCP_API_KEY') is False
        assert api_key_store.is_system_key_name('') is False

    def test_is_system_key_name_none(self, api_key_store):
        """Test that None is not a system key."""
        assert api_key_store.is_system_key_name(None) is False

    def test_make_system_key_name(self, api_key_store):
        """Test system key name generation."""
        assert (
            api_key_store.make_system_key_name('automation') == '__SYSTEM__:automation'
        )
        assert api_key_store.make_system_key_name('test-key') == '__SYSTEM__:test-key'

    @pytest.mark.asyncio
    async def test_get_or_create_system_api_key_creates_new(
        self, api_key_store, async_session_maker
    ):
        """Test creating a new system API key when none exists."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')
        key_name = 'automation'

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            api_key = await api_key_store.get_or_create_system_api_key(
                user_id=user_id,
                org_id=org_id,
                name=key_name,
            )

        assert api_key.startswith('sk-oh-')
        assert len(api_key) == len('sk-oh-') + 32

        # Verify the key was created in the database
        async with async_session_maker() as session:
            result = await session.execute(select(ApiKey).filter(ApiKey.key == api_key))
            key_record = result.scalars().first()
            assert key_record is not None
            assert key_record.user_id == user_id
            assert key_record.org_id == org_id
            assert key_record.name == '__SYSTEM__:automation'
            assert key_record.expires_at is None  # System keys never expire

    @pytest.mark.asyncio
    async def test_get_or_create_system_api_key_returns_existing(
        self, api_key_store, async_session_maker
    ):
        """Test that existing valid system key is returned."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')
        key_name = 'automation'

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            # Create the first key
            first_key = await api_key_store.get_or_create_system_api_key(
                user_id=user_id,
                org_id=org_id,
                name=key_name,
            )

            # Request again - should return the same key
            second_key = await api_key_store.get_or_create_system_api_key(
                user_id=user_id,
                org_id=org_id,
                name=key_name,
            )

        assert first_key == second_key

    @pytest.mark.asyncio
    async def test_get_or_create_system_api_key_different_names(
        self, api_key_store, async_session_maker
    ):
        """Test that different names create different keys."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            key1 = await api_key_store.get_or_create_system_api_key(
                user_id=user_id,
                org_id=org_id,
                name='automation-1',
            )

            key2 = await api_key_store.get_or_create_system_api_key(
                user_id=user_id,
                org_id=org_id,
                name='automation-2',
            )

        assert key1 != key2

    @pytest.mark.asyncio
    async def test_get_or_create_system_api_key_reissues_expired(
        self, api_key_store, async_session_maker
    ):
        """Test that expired system key is replaced with a new one."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')
        key_name = 'automation'
        system_key_name = '__SYSTEM__:automation'

        # First, manually create an expired key
        expired_time = datetime.now(UTC) - timedelta(hours=1)
        async with async_session_maker() as session:
            expired_key = ApiKey(
                key='sk-oh-expired-key-12345678901234567890',
                user_id=user_id,
                org_id=org_id,
                name=system_key_name,
                expires_at=expired_time.replace(tzinfo=None),
            )
            session.add(expired_key)
            await session.commit()

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            # Request the key - should create a new one
            new_key = await api_key_store.get_or_create_system_api_key(
                user_id=user_id,
                org_id=org_id,
                name=key_name,
            )

        assert new_key != 'sk-oh-expired-key-12345678901234567890'
        assert new_key.startswith('sk-oh-')

        # Verify old key was deleted and new key exists
        async with async_session_maker() as session:
            result = await session.execute(
                select(ApiKey).filter(ApiKey.name == system_key_name)
            )
            keys = result.scalars().all()
            assert len(keys) == 1
            assert keys[0].key == new_key
            assert keys[0].expires_at is None

    @pytest.mark.asyncio
    async def test_list_api_keys_excludes_system_keys(
        self, api_key_store, async_session_maker
    ):
        """Test that list_api_keys excludes system keys."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

        # Create a user key and a system key
        async with async_session_maker() as session:
            user_key = ApiKey(
                key='sk-oh-user-key-123456789012345678901',
                user_id=user_id,
                org_id=org_id,
                name='my-user-key',
            )
            system_key = ApiKey(
                key='sk-oh-system-key-12345678901234567890',
                user_id=user_id,
                org_id=org_id,
                name='__SYSTEM__:automation',
            )
            mcp_key = ApiKey(
                key='sk-oh-mcp-key-1234567890123456789012',
                user_id=user_id,
                org_id=org_id,
                name='MCP_API_KEY',
            )
            session.add(user_key)
            session.add(system_key)
            session.add(mcp_key)
            await session.commit()

        # Mock UserStore.get_user_by_id to return a user with the correct org
        mock_user = MagicMock()
        mock_user.current_org_id = org_id

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            with patch(
                'storage.api_key_store.UserStore.get_user_by_id', new_callable=AsyncMock
            ) as mock_get_user:
                mock_get_user.return_value = mock_user
                keys = await api_key_store.list_api_keys(user_id)

        # Should only return the user key
        assert len(keys) == 1
        assert keys[0].name == 'my-user-key'

    @pytest.mark.asyncio
    async def test_delete_api_key_by_id_protects_system_keys(
        self, api_key_store, async_session_maker
    ):
        """Test that system keys cannot be deleted by users."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

        # Create a system key
        async with async_session_maker() as session:
            system_key = ApiKey(
                key='sk-oh-system-key-12345678901234567890',
                user_id=user_id,
                org_id=org_id,
                name='__SYSTEM__:automation',
            )
            session.add(system_key)
            await session.commit()
            key_id = system_key.id

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            # Attempt to delete without allow_system flag
            result = await api_key_store.delete_api_key_by_id(
                key_id, allow_system=False
            )

        assert result is False

        # Verify the key still exists
        async with async_session_maker() as session:
            result = await session.execute(select(ApiKey).filter(ApiKey.id == key_id))
            key_record = result.scalars().first()
            assert key_record is not None

    @pytest.mark.asyncio
    async def test_delete_api_key_by_id_allows_system_with_flag(
        self, api_key_store, async_session_maker
    ):
        """Test that system keys can be deleted with allow_system=True."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

        # Create a system key
        async with async_session_maker() as session:
            system_key = ApiKey(
                key='sk-oh-system-key-12345678901234567890',
                user_id=user_id,
                org_id=org_id,
                name='__SYSTEM__:automation',
            )
            session.add(system_key)
            await session.commit()
            key_id = system_key.id

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            # Delete with allow_system=True
            result = await api_key_store.delete_api_key_by_id(key_id, allow_system=True)

        assert result is True

        # Verify the key was deleted
        async with async_session_maker() as session:
            result = await session.execute(select(ApiKey).filter(ApiKey.id == key_id))
            key_record = result.scalars().first()
            assert key_record is None

    @pytest.mark.asyncio
    async def test_delete_api_key_by_id_allows_regular_keys(
        self, api_key_store, async_session_maker
    ):
        """Test that regular keys can be deleted normally."""
        user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
        org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

        # Create a regular key
        async with async_session_maker() as session:
            regular_key = ApiKey(
                key='sk-oh-regular-key-1234567890123456789',
                user_id=user_id,
                org_id=org_id,
                name='my-regular-key',
            )
            session.add(regular_key)
            await session.commit()
            key_id = regular_key.id

        with patch('storage.api_key_store.a_session_maker', async_session_maker):
            # Delete without allow_system flag - should work for regular keys
            result = await api_key_store.delete_api_key_by_id(
                key_id, allow_system=False
            )

        assert result is True

        # Verify the key was deleted
        async with async_session_maker() as session:
            result = await session.execute(select(ApiKey).filter(ApiKey.id == key_id))
            key_record = result.scalars().first()
            assert key_record is None
