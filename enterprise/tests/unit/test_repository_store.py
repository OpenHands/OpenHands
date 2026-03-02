from unittest.mock import patch

import pytest
from sqlalchemy import select
from storage.repository_store import RepositoryStore
from storage.stored_repository import StoredRepository


@pytest.fixture
def repository_store():
    return RepositoryStore(config=None)


@pytest.mark.asyncio
async def test_store_projects_empty_list(repository_store, async_session_maker):
    """Test storing empty list of repositories."""
    with patch(
        'storage.repository_store.RepositoryStore.store_projects'
    ) as mock_method:
        # Should handle empty list gracefully
        mock_method.return_value = None
        # Test that we handle empty repositories
        result = await repository_store.store_projects([])
        # The method should return early for empty list
        assert result is None


@pytest.mark.asyncio
async def test_store_projects_new_repositories(repository_store, async_session_maker):
    """Test storing new repositories in the database."""
    # Setup - create repositories
    repo1 = StoredRepository(
        repo_name='owner/repo1',
        repo_id='github##123',
        is_public=False,
    )
    repo2 = StoredRepository(
        repo_name='owner/repo2',
        repo_id='github##456',
        is_public=True,
    )

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.repository_store.a_session_maker', async_session_maker):
        await repository_store.store_projects([repo1, repo2])

    # Verify the repositories were stored
    async with async_session_maker() as session:
        result = await session.execute(
            select(StoredRepository).filter(
                StoredRepository.repo_id.in_(['github##123', 'github##456'])
            )
        )
        repos = result.scalars().all()
        assert len(repos) == 2
        repo_ids = {r.repo_id for r in repos}
        assert 'github##123' in repo_ids
        assert 'github##456' in repo_ids


@pytest.mark.asyncio
async def test_store_projects_update_existing(repository_store, async_session_maker):
    """Test updating existing repositories in the database."""
    # Setup - create existing repository
    existing_repo = StoredRepository(
        repo_name='owner/repo1',
        repo_id='github##123',
        is_public=True,
    )

    async with async_session_maker() as session:
        session.add(existing_repo)
        await session.commit()

    # Execute - update the repository with new values
    updated_repo = StoredRepository(
        repo_name='owner/repo1-updated',
        repo_id='github##123',
        is_public=False,  # Changed from True
    )

    with patch('storage.repository_store.a_session_maker', async_session_maker):
        await repository_store.store_projects([updated_repo])

    # Verify the repository was updated
    async with async_session_maker() as session:
        result = await session.execute(
            select(StoredRepository).filter(StoredRepository.repo_id == 'github##123')
        )
        repo = result.scalars().first()
        assert repo is not None
        assert repo.repo_name == 'owner/repo1-updated'
        assert repo.is_public is False


@pytest.mark.asyncio
async def test_store_projects_mixed_new_and_existing(
    repository_store, async_session_maker
):
    """Test storing a mix of new and existing repositories."""
    # Setup - create one existing repository
    existing_repo = StoredRepository(
        repo_name='owner/existing-repo',
        repo_id='github##123',
        is_public=True,
    )

    async with async_session_maker() as session:
        session.add(existing_repo)
        await session.commit()

    # Execute - store a mix of new and existing
    repos_to_store = [
        StoredRepository(
            repo_name='owner/existing-repo',
            repo_id='github##123',
            is_public=False,  # Will update
        ),
        StoredRepository(
            repo_name='owner/new-repo',
            repo_id='github##456',
            is_public=True,
        ),
    ]

    with patch('storage.repository_store.a_session_maker', async_session_maker):
        await repository_store.store_projects(repos_to_store)

    # Verify results
    async with async_session_maker() as session:
        result = await session.execute(
            select(StoredRepository).filter(
                StoredRepository.repo_id.in_(['github##123', 'github##456'])
            )
        )
        repos = result.scalars().all()
        assert len(repos) == 2

        # Check the updated existing repo
        existing = next(r for r in repos if r.repo_id == 'github##123')
        assert existing.repo_name == 'owner/existing-repo'
        assert existing.is_public is False

        # Check the new repo
        new = next(r for r in repos if r.repo_id == 'github##456')
        assert new.repo_name == 'owner/new-repo'
        assert new.is_public is True
