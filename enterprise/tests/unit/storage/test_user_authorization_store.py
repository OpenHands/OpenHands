"""Unit tests for UserAuthorizationStore using SQLite in-memory database."""

from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.user_authorization import UserAuthorization, UserAuthorizationType
from storage.user_authorization_store import UserAuthorizationStore


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


class TestGetMatchingAuthorizations:
    """Tests for get_matching_authorizations method."""

    @pytest.mark.asyncio
    async def test_no_authorizations_returns_empty_list(self, async_session_maker):
        """Test returns empty list when no authorizations exist."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='test@example.com',
                provider_type='github',
            )
            assert result == []

    @pytest.mark.asyncio
    async def test_exact_email_pattern_match(self, async_session_maker):
        """Test matching with exact email pattern."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create a whitelist rule for exact email
            await UserAuthorizationStore.create_authorization(
                email_pattern='test@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            result = await UserAuthorizationStore.get_matching_authorizations(
                email='test@example.com',
                provider_type='github',
            )

            assert len(result) == 1
            assert result[0].email_pattern == 'test@example.com'

    @pytest.mark.asyncio
    async def test_domain_suffix_pattern_match(self, async_session_maker):
        """Test matching with domain suffix pattern (e.g., %@example.com)."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create a whitelist rule for domain
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            # Should match
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@example.com',
                provider_type='github',
            )
            assert len(result) == 1

            # Should also match different user
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='another.user@example.com',
                provider_type='github',
            )
            assert len(result) == 1

            # Should not match different domain
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@other.com',
                provider_type='github',
            )
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_null_email_pattern_matches_all_emails(self, async_session_maker):
        """Test that NULL email_pattern matches all emails."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create a rule with NULL email pattern
            await UserAuthorizationStore.create_authorization(
                email_pattern=None,
                provider_type='github',
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            # Should match any email with github provider
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='any@email.com',
                provider_type='github',
            )
            assert len(result) == 1

            # Should not match different provider
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='any@email.com',
                provider_type='gitlab',
            )
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_null_provider_type_matches_all_providers(self, async_session_maker):
        """Test that NULL provider_type matches all providers."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create a rule with NULL provider type
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@blocked.com',
                provider_type=None,
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            # Should match any provider
            for provider in ['github', 'gitlab', 'bitbucket', None]:
                result = await UserAuthorizationStore.get_matching_authorizations(
                    email='user@blocked.com',
                    provider_type=provider,
                )
                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_provider_type_filter(self, async_session_maker):
        """Test filtering by provider type."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create rules for different providers
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type='github',
                auth_type=UserAuthorizationType.WHITELIST,
            )
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type='gitlab',
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            # Check github
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@example.com',
                provider_type='github',
            )
            assert len(result) == 1
            assert result[0].type == UserAuthorizationType.WHITELIST.value

            # Check gitlab
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@example.com',
                provider_type='gitlab',
            )
            assert len(result) == 1
            assert result[0].type == UserAuthorizationType.BLACKLIST.value

    @pytest.mark.asyncio
    async def test_case_insensitive_email_matching(self, async_session_maker):
        """Test that email matching is case insensitive."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@Example.COM',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            # Should match regardless of case
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='USER@example.com',
                provider_type='github',
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_multiple_matching_rules(self, async_session_maker):
        """Test that multiple matching rules are returned."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create multiple rules that match
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )
            await UserAuthorizationStore.create_authorization(
                email_pattern=None,  # Matches all emails
                provider_type='github',
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@example.com',
                provider_type='github',
            )

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_with_provided_session(self, async_session_maker):
        """Test using a provided session instead of creating one."""
        async with async_session_maker() as session:
            # Create authorization within session
            auth = UserAuthorization(
                email_pattern='%@test.com',
                provider_type=None,
                type=UserAuthorizationType.WHITELIST.value,
            )
            session.add(auth)
            await session.flush()

            # Query within same session
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@test.com',
                provider_type='github',
                session=session,
            )

            assert len(result) == 1


class TestGetAuthorizationType:
    """Tests for get_authorization_type method."""

    @pytest.mark.asyncio
    async def test_returns_whitelist_when_whitelist_match_exists(
        self, async_session_maker
    ):
        """Test returns WHITELIST when a whitelist rule matches."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@allowed.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            result = await UserAuthorizationStore.get_authorization_type(
                email='user@allowed.com',
                provider_type='github',
            )

            assert result == UserAuthorizationType.WHITELIST

    @pytest.mark.asyncio
    async def test_returns_blacklist_when_blacklist_match_exists(
        self, async_session_maker
    ):
        """Test returns BLACKLIST when a blacklist rule matches."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@blocked.com',
                provider_type=None,
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            result = await UserAuthorizationStore.get_authorization_type(
                email='user@blocked.com',
                provider_type='github',
            )

            assert result == UserAuthorizationType.BLACKLIST

    @pytest.mark.asyncio
    async def test_returns_none_when_no_rules_exist(self, async_session_maker):
        """Test returns None when no authorization rules exist."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            result = await UserAuthorizationStore.get_authorization_type(
                email='user@example.com',
                provider_type='github',
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_only_non_matching_rules_exist(
        self, async_session_maker
    ):
        """Test returns None when rules exist but don't match."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@other.com',
                provider_type=None,
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            result = await UserAuthorizationStore.get_authorization_type(
                email='user@example.com',
                provider_type='github',
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_whitelist_takes_precedence_over_blacklist(self, async_session_maker):
        """Test whitelist takes precedence when both match."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create both whitelist and blacklist rules that match
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.BLACKLIST,
            )
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type='github',
                auth_type=UserAuthorizationType.WHITELIST,
            )

            result = await UserAuthorizationStore.get_authorization_type(
                email='user@example.com',
                provider_type='github',
            )

            assert result == UserAuthorizationType.WHITELIST

    @pytest.mark.asyncio
    async def test_returns_blacklist_for_domain_block(self, async_session_maker):
        """Test blacklist match for domain-based blocking."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@disposable-email.com',
                provider_type=None,
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            result = await UserAuthorizationStore.get_authorization_type(
                email='spammer@disposable-email.com',
                provider_type='github',
            )

            assert result == UserAuthorizationType.BLACKLIST


class TestCreateAuthorization:
    """Tests for create_authorization method."""

    @pytest.mark.asyncio
    async def test_creates_whitelist_authorization(self, async_session_maker):
        """Test creating a whitelist authorization."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            auth = await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type='github',
                auth_type=UserAuthorizationType.WHITELIST,
            )

            assert auth.id is not None
            assert auth.email_pattern == '%@example.com'
            assert auth.provider_type == 'github'
            assert auth.type == UserAuthorizationType.WHITELIST.value
            assert auth.created_at is not None
            assert auth.updated_at is not None

    @pytest.mark.asyncio
    async def test_creates_blacklist_authorization(self, async_session_maker):
        """Test creating a blacklist authorization."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            auth = await UserAuthorizationStore.create_authorization(
                email_pattern='%@blocked.com',
                provider_type=None,
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            assert auth.id is not None
            assert auth.email_pattern == '%@blocked.com'
            assert auth.provider_type is None
            assert auth.type == UserAuthorizationType.BLACKLIST.value

    @pytest.mark.asyncio
    async def test_creates_with_null_email_pattern(self, async_session_maker):
        """Test creating authorization with NULL email pattern."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            auth = await UserAuthorizationStore.create_authorization(
                email_pattern=None,
                provider_type='github',
                auth_type=UserAuthorizationType.WHITELIST,
            )

            assert auth.email_pattern is None
            assert auth.provider_type == 'github'

    @pytest.mark.asyncio
    async def test_creates_with_provided_session(self, async_session_maker):
        """Test creating authorization with a provided session."""
        async with async_session_maker() as session:
            auth = await UserAuthorizationStore.create_authorization(
                email_pattern='%@test.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
                session=session,
            )

            assert auth.id is not None

            # Verify it exists in session
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@test.com',
                provider_type='github',
                session=session,
            )
            assert len(result) == 1


class TestDeleteAuthorization:
    """Tests for delete_authorization method."""

    @pytest.mark.asyncio
    async def test_deletes_existing_authorization(self, async_session_maker):
        """Test deleting an existing authorization."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            # Create an authorization
            auth = await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            # Delete it
            deleted = await UserAuthorizationStore.delete_authorization(auth.id)

            assert deleted is True

            # Verify it's gone
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@example.com',
                provider_type='github',
            )
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_returns_false_for_nonexistent_authorization(
        self, async_session_maker
    ):
        """Test returns False when authorization doesn't exist."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            deleted = await UserAuthorizationStore.delete_authorization(99999)

            assert deleted is False

    @pytest.mark.asyncio
    async def test_deletes_with_provided_session(self, async_session_maker):
        """Test deleting authorization with a provided session."""
        async with async_session_maker() as session:
            # Create an authorization
            auth = await UserAuthorizationStore.create_authorization(
                email_pattern='%@test.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
                session=session,
            )
            auth_id = auth.id

            # Flush to persist to database before delete
            await session.flush()

            # Delete within same session
            deleted = await UserAuthorizationStore.delete_authorization(
                auth_id, session=session
            )

            assert deleted is True

            # Flush delete to database
            await session.flush()

            # Verify it's gone
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@test.com',
                provider_type='github',
                session=session,
            )
            assert len(result) == 0


class TestPatternMatchingEdgeCases:
    """Tests for edge cases in pattern matching."""

    @pytest.mark.asyncio
    async def test_wildcard_prefix_pattern(self, async_session_maker):
        """Test pattern with wildcard prefix (e.g., admin%)."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='admin%',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            # Should match
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='admin@example.com',
                provider_type='github',
            )
            assert len(result) == 1

            # Should also match
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='administrator@example.com',
                provider_type='github',
            )
            assert len(result) == 1

            # Should not match
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@admin.com',
                provider_type='github',
            )
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_single_character_wildcard(self, async_session_maker):
        """Test pattern with single character wildcard (underscore in SQL LIKE)."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='user_@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            # Should match user1@example.com
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user1@example.com',
                provider_type='github',
            )
            assert len(result) == 1

            # Should not match user12@example.com
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user12@example.com',
                provider_type='github',
            )
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_email_with_plus_sign(self, async_session_maker):
        """Test matching emails with plus signs (common for email aliases)."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.WHITELIST,
            )

            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user+alias@example.com',
                provider_type='github',
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_subdomain_email(self, async_session_maker):
        """Test that subdomain emails don't match parent domain patterns."""
        with patch(
            'storage.user_authorization_store.a_session_maker', async_session_maker
        ):
            await UserAuthorizationStore.create_authorization(
                email_pattern='%@example.com',
                provider_type=None,
                auth_type=UserAuthorizationType.BLACKLIST,
            )

            # Should match exact domain
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@example.com',
                provider_type='github',
            )
            assert len(result) == 1

            # Should NOT match subdomain
            result = await UserAuthorizationStore.get_matching_authorizations(
                email='user@sub.example.com',
                provider_type='github',
            )
            assert len(result) == 0
