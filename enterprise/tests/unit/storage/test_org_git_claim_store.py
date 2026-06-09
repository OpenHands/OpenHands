"""Tests for OrgGitClaimStore with real in-memory SQLite database.

Covers CRUD operations and unique constraint enforcement.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError
from storage.org import Org
from storage.org_git_claim_store import OrgGitClaimStore
from storage.org_member import OrgMember
from storage.role import Role
from storage.user import User


@pytest.fixture
async def seed_org_and_user(async_session_maker):
    """Create a minimal org, role, user, and org_member for FK satisfaction."""
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    role_id = 1

    async with async_session_maker() as session:
        session.add(Role(id=role_id, name='owner', rank=10))
        session.add(Org(id=org_id, name='test-org'))
        session.add(User(id=user_id, current_org_id=org_id, role_id=role_id))
        session.add(
            OrgMember(
                org_id=org_id,
                user_id=user_id,
                role_id=role_id,
                status='active',
                llm_api_key='test-key',
            )
        )
        await session.commit()

    return org_id, user_id


class TestOrgGitClaimStoreCreate:
    """Tests for OrgGitClaimStore.create_claim."""

    @pytest.mark.asyncio
    async def test_create_claim_persists_and_returns(
        self, async_session_maker, seed_org_and_user
    ):
        """A new claim is persisted with correct fields and returned."""
        org_id, user_id = seed_org_and_user

        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            claim = await OrgGitClaimStore.create_claim(
                org_id=org_id,
                provider='github',
                git_organization='OpenHands',
                claimed_by=user_id,
            )

        assert claim.org_id == org_id
        assert claim.provider == 'github'
        assert claim.git_organization == 'OpenHands'
        assert claim.claimed_by == user_id
        assert claim.claimed_at is not None

    @pytest.mark.asyncio
    async def test_create_duplicate_raises_integrity_error(
        self, async_session_maker, seed_org_and_user
    ):
        """Creating a duplicate (provider, git_organization) violates the unique constraint."""
        org_id, user_id = seed_org_and_user

        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            await OrgGitClaimStore.create_claim(
                org_id=org_id,
                provider='github',
                git_organization='DuplicateOrg',
                claimed_by=user_id,
            )

            with pytest.raises(IntegrityError):
                await OrgGitClaimStore.create_claim(
                    org_id=org_id,
                    provider='github',
                    git_organization='DuplicateOrg',
                    claimed_by=user_id,
                )


class TestOrgGitClaimStoreLookup:
    """Tests for OrgGitClaimStore lookup methods."""

    @pytest.mark.asyncio
    async def test_get_claim_by_provider_and_git_org_found(
        self, async_session_maker, seed_org_and_user
    ):
        """Returns the claim when provider+git_organization exists."""
        org_id, user_id = seed_org_and_user

        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            await OrgGitClaimStore.create_claim(
                org_id=org_id,
                provider='gitlab',
                git_organization='MyGroup',
                claimed_by=user_id,
            )

            result = await OrgGitClaimStore.get_claim_by_provider_and_git_org(
                provider='gitlab', git_organization='MyGroup'
            )

        assert result is not None
        assert result.provider == 'gitlab'
        assert result.git_organization == 'MyGroup'

    @pytest.mark.asyncio
    async def test_get_claim_by_provider_and_git_org_not_found(
        self, async_session_maker
    ):
        """Returns None when no matching claim exists."""
        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            result = await OrgGitClaimStore.get_claim_by_provider_and_git_org(
                provider='github', git_organization='NonExistent'
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_claims_by_org_id(self, async_session_maker, seed_org_and_user):
        """Returns all claims belonging to the given org."""
        org_id, user_id = seed_org_and_user

        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            await OrgGitClaimStore.create_claim(
                org_id=org_id,
                provider='github',
                git_organization='Org1',
                claimed_by=user_id,
            )
            await OrgGitClaimStore.create_claim(
                org_id=org_id,
                provider='gitlab',
                git_organization='Org2',
                claimed_by=user_id,
            )

            claims = await OrgGitClaimStore.get_claims_by_org_id(org_id)

        assert len(claims) == 2


class TestOrgGitClaimStoreDelete:
    """Tests for OrgGitClaimStore.delete_claim."""

    @pytest.mark.asyncio
    async def test_delete_existing_claim_returns_true(
        self, async_session_maker, seed_org_and_user
    ):
        """Deleting an existing claim returns True and removes it from the DB."""
        org_id, user_id = seed_org_and_user

        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            claim = await OrgGitClaimStore.create_claim(
                org_id=org_id,
                provider='github',
                git_organization='ToDelete',
                claimed_by=user_id,
            )

            result = await OrgGitClaimStore.delete_claim(
                claim_id=claim.id, org_id=org_id
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_claim_returns_false(
        self, async_session_maker, seed_org_and_user
    ):
        """Deleting a claim that doesn't exist returns False."""
        org_id, _ = seed_org_and_user

        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            result = await OrgGitClaimStore.delete_claim(
                claim_id=uuid.uuid4(), org_id=org_id
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_claim_wrong_org_returns_false(
        self, async_session_maker, seed_org_and_user
    ):
        """Deleting a claim with a mismatched org_id returns False."""
        org_id, user_id = seed_org_and_user

        with patch('storage.org_git_claim_store.a_session_maker', async_session_maker):
            claim = await OrgGitClaimStore.create_claim(
                org_id=org_id,
                provider='github',
                git_organization='WrongOrg',
                claimed_by=user_id,
            )

            result = await OrgGitClaimStore.delete_claim(
                claim_id=claim.id, org_id=uuid.uuid4()
            )

        assert result is False
