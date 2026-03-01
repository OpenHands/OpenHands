import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Mock the database module before importing OrgMemberStore
with patch('storage.database.engine', create=True), patch(
    'storage.database.a_engine', create=True
):
    from storage.base import Base
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.org_member_store import OrgMemberStore
    from storage.role import Role
    from storage.user import User


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


def test_get_org_members(session_maker):
    # Test getting org_members by org ID
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user1 = User(id=uuid.uuid4(), current_org_id=org.id)
        user2 = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user1, user2, role])
        session.flush()

        org_member1 = OrgMember(
            org_id=org.id,
            user_id=user1.id,
            role_id=role.id,
            llm_api_key='test-key-1',
            status='active',
        )
        org_member2 = OrgMember(
            org_id=org.id,
            user_id=user2.id,
            role_id=role.id,
            llm_api_key='test-key-2',
            status='active',
        )
        session.add_all([org_member1, org_member2])
        session.commit()
        org_id = org.id

    # Test retrieval
    with patch('storage.org_member_store.session_maker', session_maker):
        org_members = OrgMemberStore.get_org_members(org_id)
        assert len(org_members) == 2
        api_keys = [om.llm_api_key.get_secret_value() for om in org_members]
        assert 'test-key-1' in api_keys
        assert 'test-key-2' in api_keys


def test_get_user_orgs(session_maker):
    # Test getting org_members by user ID
    with session_maker() as session:
        # Create test data
        org1 = Org(name='test-org-1')
        org2 = Org(name='test-org-2')
        session.add_all([org1, org2])
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org1.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.flush()

        org_member1 = OrgMember(
            org_id=org1.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key-1',
            status='active',
        )
        org_member2 = OrgMember(
            org_id=org2.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key-2',
            status='active',
        )
        session.add_all([org_member1, org_member2])
        session.commit()
        user_id = user.id

    # Test retrieval
    with patch('storage.org_member_store.session_maker', session_maker):
        org_members = OrgMemberStore.get_user_orgs(user_id)
        assert len(org_members) == 2
        api_keys = [ou.llm_api_key.get_secret_value() for ou in org_members]
        assert 'test-key-1' in api_keys
        assert 'test-key-2' in api_keys


def test_get_org_member(session_maker):
    # Test getting org_member by org and user ID
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        session.commit()
        org_id = org.id
        user_id = user.id

    # Test retrieval
    with patch('storage.org_member_store.session_maker', session_maker):
        retrieved_org_member = OrgMemberStore.get_org_member(org_id, user_id)
        assert retrieved_org_member is not None
        assert retrieved_org_member.org_id == org_id
        assert retrieved_org_member.user_id == user_id
        assert retrieved_org_member.llm_api_key.get_secret_value() == 'test-key'


def test_get_org_member_for_current_org(session_maker):
    # Test getting org_member for user's current organization
    with session_maker() as session:
        # Create test data - user belongs to two orgs but current_org is org1
        org1 = Org(name='test-org-1')
        org2 = Org(name='test-org-2')
        session.add_all([org1, org2])
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org1.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.flush()

        org_member1 = OrgMember(
            org_id=org1.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key-1',
            status='active',
        )
        org_member2 = OrgMember(
            org_id=org2.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key-2',
            status='active',
        )
        session.add_all([org_member1, org_member2])
        session.commit()
        user_id = user.id
        org1_id = org1.id

    # Test retrieval - should return org_member for current_org (org1)
    with patch('storage.org_member_store.session_maker', session_maker):
        retrieved_org_member = OrgMemberStore.get_org_member_for_current_org(user_id)
        assert retrieved_org_member is not None
        assert retrieved_org_member.org_id == org1_id
        assert retrieved_org_member.user_id == user_id
        assert retrieved_org_member.llm_api_key.get_secret_value() == 'test-key-1'


def test_get_org_member_for_current_org_user_not_found(session_maker):
    # Test getting org_member for non-existent user
    with patch('storage.org_member_store.session_maker', session_maker):
        retrieved_org_member = OrgMemberStore.get_org_member_for_current_org(
            uuid.uuid4()
        )
        assert retrieved_org_member is None


def test_add_user_to_org(session_maker):
    # Test adding a user to an org
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.commit()
        org_id = org.id
        user_id = user.id
        role_id = role.id

    # Test creation
    with patch('storage.org_member_store.session_maker', session_maker):
        org_member = OrgMemberStore.add_user_to_org(
            org_id=org_id,
            user_id=user_id,
            role_id=role_id,
            llm_api_key='new-test-key',
            status='active',
        )

        assert org_member is not None
        assert org_member.org_id == org_id
        assert org_member.user_id == user_id
        assert org_member.role_id == role_id
        assert org_member.llm_api_key.get_secret_value() == 'new-test-key'
        assert org_member.status == 'active'


def test_update_user_role_in_org(session_maker):
    # Test updating user role in org
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role1 = Role(name='admin', rank=1)
        role2 = Role(name='user', rank=2)
        session.add_all([user, role1, role2])
        session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role1.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        session.commit()
        org_id = org.id
        user_id = user.id
        role2_id = role2.id

    # Test update
    with patch('storage.org_member_store.session_maker', session_maker):
        updated_org_member = OrgMemberStore.update_user_role_in_org(
            org_id=org_id, user_id=user_id, role_id=role2_id, status='inactive'
        )

        assert updated_org_member is not None
        assert updated_org_member.role_id == role2_id
        assert updated_org_member.status == 'inactive'


def test_update_user_role_in_org_not_found(session_maker):
    # Test updating org_member that doesn't exist
    from uuid import uuid4

    with patch('storage.org_member_store.session_maker', session_maker):
        updated_org_member = OrgMemberStore.update_user_role_in_org(
            org_id=uuid4(), user_id=99999, role_id=1
        )
        assert updated_org_member is None


def test_remove_user_from_org(session_maker):
    # Test removing a user from an org
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        session.commit()
        org_id = org.id
        user_id = user.id

    # Test removal
    with patch('storage.org_member_store.session_maker', session_maker):
        result = OrgMemberStore.remove_user_from_org(org_id, user_id)
        assert result is True

        # Verify it's removed
        retrieved_org_member = OrgMemberStore.get_org_member(org_id, user_id)
        assert retrieved_org_member is None


def test_remove_user_from_org_not_found(session_maker):
    # Test removing user from org that doesn't exist
    from uuid import uuid4

    with patch('storage.org_member_store.session_maker', session_maker):
        result = OrgMemberStore.remove_user_from_org(uuid4(), 99999)
        assert result is False


@pytest.mark.asyncio
async def test_get_org_members_paginated_basic(async_session_maker):
    """Test basic pagination returns correct number of items."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        # Create 5 users
        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email=f'user{i}@example.com')
            for i in range(5)
        ]
        session.add_all(users)
        await session.flush()

        # Create org members
        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user in enumerate(users)
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=3
        )

        # Assert
        assert len(members) == 3
        assert has_more is True
        # Verify user and role relationships are loaded
        assert all(member.user is not None for member in members)
        assert all(member.role is not None for member in members)


@pytest.mark.asyncio
async def test_get_org_members_paginated_no_more(async_session_maker):
    """Test pagination when there are no more results."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        # Create 3 users
        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email=f'user{i}@example.com')
            for i in range(3)
        ]
        session.add_all(users)
        await session.flush()

        # Create org members
        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user in enumerate(users)
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=5
        )

        # Assert
        assert len(members) == 3
        assert has_more is False


@pytest.mark.asyncio
async def test_get_org_members_paginated_exact_limit(async_session_maker):
    """Test pagination when results exactly match limit."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        # Create exactly 5 users
        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email=f'user{i}@example.com')
            for i in range(5)
        ]
        session.add_all(users)
        await session.flush()

        # Create org members
        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user in enumerate(users)
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=5
        )

        # Assert
        assert len(members) == 5
        assert has_more is False


@pytest.mark.asyncio
async def test_get_org_members_paginated_with_offset(async_session_maker):
    """Test pagination with offset skips correct number of items."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        # Create 10 users
        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email=f'user{i}@example.com')
            for i in range(10)
        ]
        session.add_all(users)
        await session.flush()

        # Create org members
        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user in enumerate(users)
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act - Get first page
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        first_page, has_more_first = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=3
        )

        # Get second page
        second_page, has_more_second = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=3, limit=3
        )

        # Assert
        assert len(first_page) == 3
        assert has_more_first is True
        assert len(second_page) == 3
        assert has_more_second is True

        # Verify no overlap between pages
        first_user_ids = {member.user_id for member in first_page}
        second_user_ids = {member.user_id for member in second_page}
        assert first_user_ids.isdisjoint(second_user_ids)


@pytest.mark.asyncio
async def test_get_org_members_paginated_empty_org(async_session_maker):
    """Test pagination with empty organization returns empty list."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=10
        )

        # Assert
        assert len(members) == 0
        assert has_more is False


@pytest.mark.asyncio
async def test_get_org_members_paginated_ordering(async_session_maker):
    """Test that pagination orders results by user_id."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        # Create users with specific IDs to test ordering
        user_ids = [uuid.uuid4() for _ in range(5)]
        user_ids.sort()  # Sort to verify ordering

        users = [
            User(id=user_id, current_org_id=org.id, email=f'user{i}@example.com')
            for i, user_id in enumerate(user_ids)
        ]
        session.add_all(users)
        await session.flush()

        # Create org members in reverse order to test that ordering works
        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user_id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user_id in enumerate(reversed(user_ids))
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=10
        )

        # Assert
        assert len(members) == 5
        # Verify members are ordered by user_id
        member_user_ids = [member.user_id for member in members]
        assert member_user_ids == sorted(member_user_ids)


@pytest.mark.asyncio
async def test_get_org_members_paginated_eager_loading(async_session_maker):
    """Test that user and role relationships are eagerly loaded."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='owner', rank=10)
        session.add(role)
        await session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id, email='test@example.com')
        session.add(user)
        await session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=10
        )

        # Assert
        assert len(members) == 1
        member = members[0]
        # Verify relationships are loaded (not lazy)
        assert member.user is not None
        assert member.user.email == 'test@example.com'
        assert member.role is not None
        assert member.role.name == 'owner'
        assert member.role.rank == 10


@pytest.mark.asyncio
async def test_get_org_members_count_no_filter(async_session_maker):
    """Test get_org_members_count returns correct count without email filter."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email=f'user{i}@example.com')
            for i in range(5)
        ]
        session.add_all(users)
        await session.flush()

        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user in enumerate(users)
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        count = await OrgMemberStore.get_org_members_count(org_id=org_id)

    # Assert
    assert count == 5


@pytest.mark.asyncio
async def test_get_org_members_count_with_email_filter(async_session_maker):
    """Test get_org_members_count filters by email correctly."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email='alice@example.com'),
            User(id=uuid.uuid4(), current_org_id=org.id, email='bob@example.com'),
            User(
                id=uuid.uuid4(), current_org_id=org.id, email='alice.smith@example.com'
            ),
        ]
        session.add_all(users)
        await session.flush()

        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user in enumerate(users)
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        count = await OrgMemberStore.get_org_members_count(
            org_id=org_id, email_filter='alice'
        )

    # Assert
    assert count == 2


@pytest.mark.asyncio
async def test_get_org_members_paginated_with_email_filter(async_session_maker):
    """Test get_org_members_paginated filters by email correctly."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email='alice@example.com'),
            User(id=uuid.uuid4(), current_org_id=org.id, email='bob@example.com'),
            User(id=uuid.uuid4(), current_org_id=org.id, email='charlie@example.com'),
        ]
        session.add_all(users)
        await session.flush()

        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key=f'test-key-{i}',
                status='active',
            )
            for i, user in enumerate(users)
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=10, email_filter='bob'
        )

    # Assert
    assert len(members) == 1
    assert members[0].user.email == 'bob@example.com'
    assert has_more is False


@pytest.mark.asyncio
async def test_get_org_members_paginated_email_filter_case_insensitive(
    async_session_maker,
):
    """Test email filter is case-insensitive."""
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='admin', rank=1)
        session.add(role)
        await session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id, email='Alice@Example.COM')
        session.add(user)
        await session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        await session.commit()
        org_id = org.id

    # Act
    with patch('storage.org_member_store.a_session_maker', async_session_maker):
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=0, limit=10, email_filter='alice@example'
        )

    # Assert
    assert len(members) == 1
    assert members[0].user.email == 'Alice@Example.COM'


@pytest.mark.asyncio
async def test_update_all_members_llm_settings_async_with_llm_api_key(
    async_session_maker,
):
    """
    GIVEN: Organization with members and llm_api_key in member settings
    WHEN: update_all_members_llm_settings_async is called with llm_api_key
    THEN: The llm_api_key is encrypted and stored in _llm_api_key column for all members
    """
    from server.routes.org_models import OrgMemberLLMSettings
    from storage.encrypt_utils import decrypt_value

    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='member', rank=2)
        session.add(role)
        await session.flush()

        users = [
            User(id=uuid.uuid4(), current_org_id=org.id, email=f'user{i}@example.com')
            for i in range(2)
        ]
        session.add_all(users)
        await session.flush()

        org_members = [
            OrgMember(
                org_id=org.id,
                user_id=user.id,
                role_id=role.id,
                llm_api_key='old-key',
                status='active',
            )
            for user in users
        ]
        session.add_all(org_members)
        await session.commit()
        org_id = org.id

    # Act
    new_api_key = 'new-test-api-key-12345'
    member_settings = OrgMemberLLMSettings(llm_api_key=new_api_key)

    async with async_session_maker() as session:
        await OrgMemberStore.update_all_members_llm_settings_async(
            session, org_id, member_settings
        )
        await session.commit()

    # Assert
    async with async_session_maker() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(OrgMember).filter(OrgMember.org_id == org_id)
        )
        updated_members = result.scalars().all()

        assert len(updated_members) == 2
        for member in updated_members:
            # Verify the encrypted value can be decrypted to the original
            decrypted_key = decrypt_value(member._llm_api_key)
            assert decrypted_key == new_api_key


@pytest.mark.asyncio
async def test_update_all_members_llm_settings_async_with_non_encrypted_fields(
    async_session_maker,
):
    """
    GIVEN: Organization with members
    WHEN: update_all_members_llm_settings_async is called with non-encrypted fields
    THEN: The fields are updated directly without encryption
    """
    from server.routes.org_models import OrgMemberLLMSettings

    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='member', rank=2)
        session.add(role)
        await session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id, email='user@example.com')
        session.add(user)
        await session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key',
            llm_model='old-model',
            max_iterations=10,
            status='active',
        )
        session.add(org_member)
        await session.commit()
        org_id = org.id

    # Act
    member_settings = OrgMemberLLMSettings(
        llm_model='new-model',
        llm_base_url='https://new-url.com',
        max_iterations=50,
    )

    async with async_session_maker() as session:
        await OrgMemberStore.update_all_members_llm_settings_async(
            session, org_id, member_settings
        )
        await session.commit()

    # Assert
    async with async_session_maker() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(OrgMember).filter(OrgMember.org_id == org_id)
        )
        updated_member = result.scalars().first()

        assert updated_member.llm_model == 'new-model'
        assert updated_member.llm_base_url == 'https://new-url.com'
        assert updated_member.max_iterations == 50


@pytest.mark.asyncio
async def test_update_all_members_llm_settings_async_with_empty_settings(
    async_session_maker,
):
    """
    GIVEN: Organization with members and empty member settings
    WHEN: update_all_members_llm_settings_async is called with no fields set
    THEN: No database update is performed
    """
    from server.routes.org_models import OrgMemberLLMSettings

    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        role = Role(name='member', rank=2)
        session.add(role)
        await session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id, email='user@example.com')
        session.add(user)
        await session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='original-key',
            llm_model='original-model',
            status='active',
        )
        session.add(org_member)
        await session.commit()
        org_id = org.id

    # Act - Empty settings (all None)
    member_settings = OrgMemberLLMSettings()

    async with async_session_maker() as session:
        await OrgMemberStore.update_all_members_llm_settings_async(
            session, org_id, member_settings
        )
        await session.commit()

    # Assert - Original values should be unchanged
    async with async_session_maker() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(OrgMember).filter(OrgMember.org_id == org_id)
        )
        member = result.scalars().first()

        assert member.llm_model == 'original-model'
        # Original key should still be there (encrypted)
        assert member._llm_api_key is not None


# =============================================================================
# OrgMemberLLMSettings and OrgLLMSettingsUpdate Model Unit Tests
# =============================================================================


def test_org_member_llm_settings_has_updates_with_llm_api_key():
    """
    GIVEN: OrgMemberLLMSettings with only llm_api_key set
    WHEN: has_updates() is called
    THEN: Returns True
    """
    from server.routes.org_models import OrgMemberLLMSettings

    # Arrange
    settings = OrgMemberLLMSettings(llm_api_key='test-key')

    # Act
    result = settings.has_updates()

    # Assert
    assert result is True


def test_org_member_llm_settings_has_updates_empty():
    """
    GIVEN: OrgMemberLLMSettings with no fields set
    WHEN: has_updates() is called
    THEN: Returns False
    """
    from server.routes.org_models import OrgMemberLLMSettings

    # Arrange
    settings = OrgMemberLLMSettings()

    # Act
    result = settings.has_updates()

    # Assert
    assert result is False


def test_org_llm_settings_update_apply_to_org_skips_llm_api_key():
    """
    GIVEN: OrgLLMSettingsUpdate with llm_api_key and other fields set
    WHEN: apply_to_org() is called
    THEN: llm_api_key is NOT applied to org, but other fields are
    """
    from unittest.mock import MagicMock

    from server.routes.org_models import OrgLLMSettingsUpdate

    # Arrange
    settings = OrgLLMSettingsUpdate(
        default_llm_model='claude-3',
        llm_api_key='should-not-be-applied',
    )
    mock_org = MagicMock()
    mock_org.default_llm_model = None

    # Act
    settings.apply_to_org(mock_org)

    # Assert
    assert mock_org.default_llm_model == 'claude-3'
    # llm_api_key should NOT be set on org (it's member-only)
    assert (
        not hasattr(mock_org, 'llm_api_key')
        or mock_org.llm_api_key != 'should-not-be-applied'
    )


def test_org_llm_settings_update_get_member_updates_includes_llm_api_key():
    """
    GIVEN: OrgLLMSettingsUpdate with llm_api_key set
    WHEN: get_member_updates() is called
    THEN: Returns OrgMemberLLMSettings with llm_api_key included
    """
    from server.routes.org_models import OrgLLMSettingsUpdate

    # Arrange
    settings = OrgLLMSettingsUpdate(
        default_llm_model='claude-3',
        llm_api_key='new-member-key',
    )

    # Act
    member_updates = settings.get_member_updates()

    # Assert
    assert member_updates is not None
    assert member_updates.llm_api_key == 'new-member-key'
    assert member_updates.llm_model == 'claude-3'


def test_org_llm_settings_update_get_member_updates_only_llm_api_key():
    """
    GIVEN: OrgLLMSettingsUpdate with only llm_api_key set
    WHEN: get_member_updates() is called
    THEN: Returns OrgMemberLLMSettings with llm_api_key (not None)
    """
    from server.routes.org_models import OrgLLMSettingsUpdate

    # Arrange
    settings = OrgLLMSettingsUpdate(llm_api_key='member-key-only')

    # Act
    member_updates = settings.get_member_updates()

    # Assert
    assert member_updates is not None
    assert member_updates.llm_api_key == 'member-key-only'
    assert member_updates.llm_model is None


def test_org_llm_settings_update_has_updates_with_llm_api_key():
    """
    GIVEN: OrgLLMSettingsUpdate with only llm_api_key set
    WHEN: has_updates() is called
    THEN: Returns True
    """
    from server.routes.org_models import OrgLLMSettingsUpdate

    # Arrange
    settings = OrgLLMSettingsUpdate(llm_api_key='test-key')

    # Act
    result = settings.has_updates()

    # Assert
    assert result is True
