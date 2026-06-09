"""Unit tests for SaasUserAuth.get_org_info() using SQLite in-memory database.

These tests exercise the real `get_org_info()` implementation with actual DB queries
to catch regressions in the SAAS org lookup logic.
"""

import uuid
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role
from storage.user import User


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
    session_maker = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return session_maker


@pytest.fixture
def user_id():
    """Generate a unique user ID for tests."""
    return str(uuid.uuid4())


@pytest.fixture
def org_id():
    """Generate a unique org ID for tests."""
    return uuid.uuid4()


async def create_role(session_maker, name: str, rank: int) -> Role:
    """Helper to create a role in the test database."""
    async with session_maker() as session:
        role = Role(name=name, rank=rank)
        session.add(role)
        await session.commit()
        await session.refresh(role)
        return role


async def create_org(session_maker, org_id: uuid.UUID, name: str) -> Org:
    """Helper to create an org in the test database."""
    async with session_maker() as session:
        org = Org(
            id=org_id,
            name=name,
            org_version=1,
            enable_proactive_conversation_starters=True,
        )
        session.add(org)
        await session.commit()
        await session.refresh(org)
        return org


async def create_user(session_maker, user_id: str, current_org_id: uuid.UUID) -> User:
    """Helper to create a user in the test database."""
    async with session_maker() as session:
        user = User(
            id=uuid.UUID(user_id),
            current_org_id=current_org_id,
            user_consents_to_analytics=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def create_org_member(
    session_maker,
    org_id: uuid.UUID,
    user_id: str,
    role_id: int,
    status: str = 'active',
    llm_api_key: str = 'test-api-key',
) -> OrgMember:
    """Helper to create an org member in the test database."""
    async with session_maker() as session:
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=role_id,
            status=status,
            llm_api_key=llm_api_key,
        )
        session.add(org_member)
        await session.commit()
        await session.refresh(org_member)
        return org_member


class TestGetOrgInfoWithRealDB:
    """Tests for get_org_info() using in-memory SQLite database."""

    @pytest.mark.asyncio
    async def test_get_org_info_returns_correct_data_for_owner(
        self, async_session_maker, user_id, org_id
    ):
        """Test that get_org_info returns correct data for an owner role."""
        # Set up test data
        owner_role = await create_role(async_session_maker, 'owner', 1)
        await create_org(async_session_maker, org_id, 'Test Organization')
        await create_user(async_session_maker, user_id, org_id)
        await create_org_member(async_session_maker, org_id, user_id, owner_role.id)

        # Create SaasUserAuth instance
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        # Patch the global a_session_maker in all stores that use it
        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            org_info = await user_auth.get_org_info()

        assert org_info is not None
        assert org_info['org_id'] == str(org_id)
        assert org_info['org_name'] == 'Test Organization'
        assert org_info['role'] == 'owner'
        assert isinstance(org_info['permissions'], list)
        # Owner should have many permissions
        assert len(org_info['permissions']) > 0
        assert 'manage_secrets' in org_info['permissions']

    @pytest.mark.asyncio
    async def test_get_org_info_returns_correct_data_for_member(
        self, async_session_maker, user_id, org_id
    ):
        """Test that get_org_info returns correct data for a member role."""
        # Set up test data
        member_role = await create_role(async_session_maker, 'member', 3)
        await create_org(async_session_maker, org_id, 'Member Org')
        await create_user(async_session_maker, user_id, org_id)
        await create_org_member(async_session_maker, org_id, user_id, member_role.id)

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            org_info = await user_auth.get_org_info()

        assert org_info is not None
        assert org_info['org_id'] == str(org_id)
        assert org_info['org_name'] == 'Member Org'
        assert org_info['role'] == 'member'
        # Member should have limited permissions
        assert isinstance(org_info['permissions'], list)

    @pytest.mark.asyncio
    async def test_get_org_info_returns_correct_data_for_admin(
        self, async_session_maker, user_id, org_id
    ):
        """Test that get_org_info returns correct data for an admin role."""
        # Set up test data
        admin_role = await create_role(async_session_maker, 'admin', 2)
        await create_org(async_session_maker, org_id, 'Admin Org')
        await create_user(async_session_maker, user_id, org_id)
        await create_org_member(async_session_maker, org_id, user_id, admin_role.id)

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            org_info = await user_auth.get_org_info()

        assert org_info is not None
        assert org_info['org_id'] == str(org_id)
        assert org_info['org_name'] == 'Admin Org'
        assert org_info['role'] == 'admin'
        assert isinstance(org_info['permissions'], list)

    @pytest.mark.asyncio
    async def test_get_org_info_returns_none_when_user_not_found(
        self, async_session_maker
    ):
        """Test that get_org_info returns None when user doesn't exist."""
        nonexistent_user_id = str(uuid.uuid4())

        user_auth = SaasUserAuth(
            user_id=nonexistent_user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            org_info = await user_auth.get_org_info()

        assert org_info is None

    @pytest.mark.asyncio
    async def test_get_org_info_returns_none_when_org_not_found(
        self, async_session_maker, user_id
    ):
        """Test that get_org_info returns None when user's org doesn't exist."""
        nonexistent_org_id = uuid.uuid4()

        # Create user pointing to nonexistent org
        async with async_session_maker() as session:
            user = User(
                id=uuid.UUID(user_id),
                current_org_id=nonexistent_org_id,
                user_consents_to_analytics=True,
            )
            session.add(user)
            await session.commit()

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            org_info = await user_auth.get_org_info()

        assert org_info is None

    @pytest.mark.asyncio
    async def test_get_org_info_caches_result(
        self, async_session_maker, user_id, org_id
    ):
        """Test that get_org_info caches the result and doesn't hit DB twice."""
        # Set up test data
        owner_role = await create_role(async_session_maker, 'owner', 1)
        await create_org(async_session_maker, org_id, 'Cached Org')
        await create_user(async_session_maker, user_id, org_id)
        await create_org_member(async_session_maker, org_id, user_id, owner_role.id)

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            # First call
            org_info1 = await user_auth.get_org_info()
            assert org_info1 is not None
            assert user_auth._org_info_loaded is True

            # Second call should return cached result
            org_info2 = await user_auth.get_org_info()
            assert org_info2 is not None
            assert org_info1 == org_info2

    @pytest.mark.asyncio
    async def test_get_org_info_caches_none_result(self, async_session_maker):
        """Test that get_org_info caches None result for nonexistent user."""
        nonexistent_user_id = str(uuid.uuid4())

        user_auth = SaasUserAuth(
            user_id=nonexistent_user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            # First call
            org_info1 = await user_auth.get_org_info()
            assert org_info1 is None
            assert user_auth._org_info_loaded is True

            # Second call should return cached None without hitting DB
            org_info2 = await user_auth.get_org_info()
            assert org_info2 is None

    @pytest.mark.asyncio
    async def test_get_org_info_with_unknown_role_returns_empty_permissions(
        self, async_session_maker, user_id, org_id
    ):
        """Test that get_org_info returns empty permissions for unknown role."""
        # Create a custom role that isn't in the ROLE_PERMISSIONS mapping
        custom_role = await create_role(async_session_maker, 'custom_role', 99)
        await create_org(async_session_maker, org_id, 'Custom Org')
        await create_user(async_session_maker, user_id, org_id)
        await create_org_member(async_session_maker, org_id, user_id, custom_role.id)

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock_refresh_token'),
        )

        with (
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
            patch('storage.org_member_store.a_session_maker', async_session_maker),
            patch('storage.role_store.a_session_maker', async_session_maker),
        ):
            org_info = await user_auth.get_org_info()

        assert org_info is not None
        assert org_info['org_id'] == str(org_id)
        assert org_info['role'] == 'custom_role'
        # Unknown roles should have empty permissions
        assert org_info['permissions'] == []
