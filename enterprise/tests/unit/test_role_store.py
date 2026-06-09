from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.role import Role
from storage.role_store import RoleStore


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_session_maker(async_engine):
    """Create an async session maker for testing."""
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_get_role_by_id_with_session(async_session_maker):
    """Test getting role by ID with an explicit session."""
    # Create a test role
    async with async_session_maker() as session:
        role = Role(name='admin', rank=1)
        session.add(role)
        await session.commit()
        await session.refresh(role)
        role_id = role.id

    # Test retrieval with explicit session
    async with async_session_maker() as session:
        retrieved_role = await RoleStore.get_role_by_id(role_id, session=session)
        assert retrieved_role is not None
        assert retrieved_role.id == role_id
        assert retrieved_role.name == 'admin'


@pytest.mark.asyncio
async def test_get_role_by_id_without_session(async_session_maker):
    """Test getting role by ID using internal session maker."""
    # Create a test role
    async with async_session_maker() as session:
        role = Role(name='admin', rank=1)
        session.add(role)
        await session.commit()
        await session.refresh(role)
        role_id = role.id

    # Test retrieval without explicit session (using patched a_session_maker)
    with patch('storage.role_store.a_session_maker', async_session_maker):
        retrieved_role = await RoleStore.get_role_by_id(role_id)
        assert retrieved_role is not None
        assert retrieved_role.id == role_id
        assert retrieved_role.name == 'admin'


@pytest.mark.asyncio
async def test_get_role_by_id_not_found(async_session_maker):
    """Test getting role by ID when it doesn't exist."""
    with patch('storage.role_store.a_session_maker', async_session_maker):
        retrieved_role = await RoleStore.get_role_by_id(99999)
        assert retrieved_role is None


@pytest.mark.asyncio
async def test_get_role_by_name_with_session(async_session_maker):
    """Test getting role by name with an explicit session."""
    # Create a test role
    async with async_session_maker() as session:
        role = Role(name='admin', rank=1)
        session.add(role)
        await session.commit()
        await session.refresh(role)
        role_id = role.id

    # Test retrieval with explicit session
    async with async_session_maker() as session:
        retrieved_role = await RoleStore.get_role_by_name('admin', session=session)
        assert retrieved_role is not None
        assert retrieved_role.id == role_id
        assert retrieved_role.name == 'admin'
        assert retrieved_role.rank == 1


@pytest.mark.asyncio
async def test_get_role_by_name_without_session(async_session_maker):
    """Test getting role by name using internal session maker."""
    # Create a test role
    async with async_session_maker() as session:
        role = Role(name='editor', rank=2)
        session.add(role)
        await session.commit()
        await session.refresh(role)
        role_id = role.id

    # Test retrieval without explicit session (using patched a_session_maker)
    with patch('storage.role_store.a_session_maker', async_session_maker):
        retrieved_role = await RoleStore.get_role_by_name('editor')
        assert retrieved_role is not None
        assert retrieved_role.id == role_id
        assert retrieved_role.name == 'editor'
        assert retrieved_role.rank == 2


@pytest.mark.asyncio
async def test_get_role_by_name_not_found_with_session(async_session_maker):
    """Test getting role by name when it doesn't exist (with explicit session)."""
    async with async_session_maker() as session:
        retrieved_role = await RoleStore.get_role_by_name(
            'nonexistent', session=session
        )
        assert retrieved_role is None


@pytest.mark.asyncio
async def test_get_role_by_name_not_found_without_session(async_session_maker):
    """Test getting role by name when it doesn't exist (without explicit session)."""
    with patch('storage.role_store.a_session_maker', async_session_maker):
        retrieved_role = await RoleStore.get_role_by_name('nonexistent')
        assert retrieved_role is None


@pytest.mark.asyncio
async def test_list_roles_with_session(async_session_maker):
    """Test listing all roles with an explicit session."""
    # Create test roles
    async with async_session_maker() as session:
        role1 = Role(name='admin', rank=1)
        role2 = Role(name='user', rank=2)
        session.add_all([role1, role2])
        await session.commit()

    # Test listing with explicit session
    async with async_session_maker() as session:
        roles = await RoleStore.list_roles(session=session)
        assert len(roles) >= 2
        role_names = [role.name for role in roles]
        assert 'admin' in role_names
        assert 'user' in role_names


@pytest.mark.asyncio
async def test_list_roles_without_session(async_session_maker):
    """Test listing all roles using internal session maker."""
    # Create test roles
    async with async_session_maker() as session:
        role1 = Role(name='admin', rank=1)
        role2 = Role(name='user', rank=2)
        session.add_all([role1, role2])
        await session.commit()

    # Test listing without explicit session (using patched a_session_maker)
    with patch('storage.role_store.a_session_maker', async_session_maker):
        roles = await RoleStore.list_roles()
        assert len(roles) >= 2
        role_names = [role.name for role in roles]
        assert 'admin' in role_names
        assert 'user' in role_names


@pytest.mark.asyncio
async def test_create_role_with_session(async_session_maker):
    """Test creating a new role with an explicit session."""
    async with async_session_maker() as session:
        role = await RoleStore.create_role(name='moderator', rank=2, session=session)
        await session.commit()

        assert role is not None
        assert role.name == 'moderator'
        assert role.rank == 2
        assert role.id is not None


@pytest.mark.asyncio
async def test_create_role_without_session(async_session_maker):
    """Test creating a new role using internal session maker."""
    with patch('storage.role_store.a_session_maker', async_session_maker):
        role = await RoleStore.create_role(name='moderator', rank=2)

        assert role is not None
        assert role.name == 'moderator'
        assert role.rank == 2
        assert role.id is not None
