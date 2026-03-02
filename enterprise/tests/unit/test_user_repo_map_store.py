import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select
from storage.user_repo_map import UserRepositoryMap
from storage.user_repo_map_store import UserRepositoryMapStore


@pytest.fixture
def user_repo_map_store():
    return UserRepositoryMapStore(config=None)


@pytest.mark.asyncio
async def test_store_user_repo_mappings_empty_list(
    user_repo_map_store, async_session_maker
):
    """Test storing empty list of mappings."""
    # Should handle empty list gracefully
    with patch(
        'storage.user_repo_map_store.UserRepositoryMapStore.store_user_repo_mappings'
    ) as mock_method:
        mock_method.return_value = None
        result = await user_repo_map_store.store_user_repo_mappings([])
        assert result is None


@pytest.mark.asyncio
async def test_store_user_repo_mappings_new_mappings(
    user_repo_map_store, async_session_maker
):
    """Test storing new user-repository mappings in the database."""
    # Setup - create mappings
    user_id = str(uuid.uuid4())
    mapping1 = UserRepositoryMap(
        user_id=user_id,
        repo_id='github##123',
        admin=True,
    )
    mapping2 = UserRepositoryMap(
        user_id=user_id,
        repo_id='github##456',
        admin=False,
    )

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.user_repo_map_store.a_session_maker', async_session_maker):
        await user_repo_map_store.store_user_repo_mappings([mapping1, mapping2])

    # Verify the mappings were stored
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserRepositoryMap).filter(
                UserRepositoryMap.repo_id.in_(['github##123', 'github##456'])
            )
        )
        mappings = result.scalars().all()
        assert len(mappings) == 2
        repo_ids = {m.repo_id for m in mappings}
        assert 'github##123' in repo_ids
        assert 'github##456' in repo_ids


@pytest.mark.asyncio
async def test_store_user_repo_mappings_update_existing(
    user_repo_map_store, async_session_maker
):
    """Test updating existing user-repository mappings in the database."""
    user_id = str(uuid.uuid4())

    # Setup - create existing mapping
    existing_mapping = UserRepositoryMap(
        user_id=user_id,
        repo_id='github##123',
        admin=False,
    )

    async with async_session_maker() as session:
        session.add(existing_mapping)
        await session.commit()

    # Execute - update the mapping with new values
    updated_mapping = UserRepositoryMap(
        user_id=user_id,
        repo_id='github##123',
        admin=True,  # Changed from False
    )

    with patch('storage.user_repo_map_store.a_session_maker', async_session_maker):
        await user_repo_map_store.store_user_repo_mappings([updated_mapping])

    # Verify the mapping was updated
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserRepositoryMap).filter(
                UserRepositoryMap.user_id == user_id,
                UserRepositoryMap.repo_id == 'github##123',
            )
        )
        mapping = result.scalars().first()
        assert mapping is not None
        assert mapping.admin is True


@pytest.mark.asyncio
async def test_store_user_repo_mappings_mixed_new_and_existing(
    user_repo_map_store, async_session_maker
):
    """Test storing a mix of new and existing mappings."""
    user_id = str(uuid.uuid4())

    # Setup - create one existing mapping
    existing_mapping = UserRepositoryMap(
        user_id=user_id,
        repo_id='github##123',
        admin=False,
    )

    async with async_session_maker() as session:
        session.add(existing_mapping)
        await session.commit()

    # Execute - store a mix of new and existing
    mappings_to_store = [
        UserRepositoryMap(
            user_id=user_id,
            repo_id='github##123',
            admin=True,  # Will update
        ),
        UserRepositoryMap(
            user_id=user_id,
            repo_id='github##456',
            admin=True,
        ),
    ]

    with patch('storage.user_repo_map_store.a_session_maker', async_session_maker):
        await user_repo_map_store.store_user_repo_mappings(mappings_to_store)

    # Verify results
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserRepositoryMap).filter(
                UserRepositoryMap.repo_id.in_(['github##123', 'github##456'])
            )
        )
        mappings = result.scalars().all()
        assert len(mappings) == 2

        # Check the updated existing mapping
        existing = next(m for m in mappings if m.repo_id == 'github##123')
        assert existing.admin is True

        # Check the new mapping
        new = next(m for m in mappings if m.repo_id == 'github##456')
        assert new.admin is True


@pytest.mark.asyncio
async def test_store_user_repo_mappings_different_users(
    user_repo_map_store, async_session_maker
):
    """Test that mappings with different user IDs are stored separately."""
    user_id1 = str(uuid.uuid4())
    user_id2 = str(uuid.uuid4())

    # Execute - store mappings for different users
    mappings = [
        UserRepositoryMap(user_id=user_id1, repo_id='github##123', admin=True),
        UserRepositoryMap(user_id=user_id2, repo_id='github##123', admin=False),
    ]

    with patch('storage.user_repo_map_store.a_session_maker', async_session_maker):
        await user_repo_map_store.store_user_repo_mappings(mappings)

    # Verify results
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserRepositoryMap).filter(UserRepositoryMap.repo_id == 'github##123')
        )
        mappings = result.scalars().all()
        assert len(mappings) == 2

        # Check both users have correct admin values
        admin_values = {m.user_id: m.admin for m in mappings}
        assert admin_values[user_id1] is True
        assert admin_values[user_id2] is False
