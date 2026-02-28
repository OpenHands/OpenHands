import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from sqlalchemy.orm import configure_mappers

# Database connection is lazy (no module-level engines), so no patching needed
from storage.org import Org
from storage.user import User
from storage.user_store import UserStore

from openhands.storage.data_models.settings import Settings


@pytest.fixture(autouse=True, scope='session')
def load_all_models():
    configure_mappers()  # fail fast if anything’s missing
    yield


@pytest.fixture
def mock_litellm_api():
    api_key_patch = patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test_key')
    api_url_patch = patch(
        'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'
    )
    team_id_patch = patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test_team')
    client_patch = patch('httpx.AsyncClient')

    with api_key_patch, api_url_patch, team_id_patch, client_patch as mock_client:
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.json = MagicMock(return_value={'key': 'test_api_key'})
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_response
        )
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )
        yield mock_client


@pytest.fixture
def mock_stripe():
    search_patch = patch(
        'stripe.Customer.search_async',
        AsyncMock(return_value=MagicMock(id='mock-customer-id')),
    )
    payment_patch = patch(
        'stripe.Customer.list_payment_methods_async',
        AsyncMock(return_value=MagicMock(data=[{}])),
    )
    with search_patch, payment_patch:
        yield


@pytest.mark.asyncio
async def test_create_default_settings_no_org_id():
    # Test UserStore.create_default_settings with empty org_id
    settings = await UserStore.create_default_settings('', 'test-user-id')
    assert settings is None


@pytest.mark.asyncio
async def test_create_default_settings_require_org(session_maker, mock_stripe):
    # Mock stripe_service.has_payment_method to return False
    with (
        patch(
            'stripe.Customer.list_payment_methods_async',
            AsyncMock(return_value=MagicMock(data=[])),
        ),
        patch('integrations.stripe_service.session_maker', session_maker),
    ):
        settings = await UserStore.create_default_settings(
            'test-org-id', 'test-user-id'
        )
        assert settings is None


@pytest.mark.asyncio
async def test_create_default_settings_with_litellm(session_maker, mock_litellm_api):
    # Test that UserStore.create_default_settings works with LiteLLM
    with (
        patch('integrations.stripe_service.session_maker', session_maker),
        patch('storage.user_store.session_maker', session_maker),
        patch('storage.org_store.session_maker', session_maker),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'attributes': {'github_id': ['12345']}}),
        ),
    ):
        settings = await UserStore.create_default_settings(
            'test-org-id', 'test-user-id'
        )
        assert settings is not None
        assert settings.llm_api_key.get_secret_value() == 'test_api_key'
        assert settings.llm_base_url == 'http://test.url'
        assert settings.agent == 'CodeActAgent'


@pytest.mark.skip(reason='Complex integration test with session isolation issues')
@pytest.mark.asyncio
async def test_create_user(session_maker, mock_litellm_api):
    # Test creating a new user - skipped due to complex session isolation issues
    pass


def test_get_user_by_id(session_maker):
    # Test getting user by ID
    test_org_id = uuid.uuid4()
    test_user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'
    with session_maker() as session:
        # Create a test user
        user = User(id=uuid.UUID(test_user_id), current_org_id=test_org_id)
        session.add(user)
        session.commit()
        user_id = user.id

    # Test retrieval
    with patch('storage.user_store.session_maker', session_maker):
        retrieved_user = UserStore.get_user_by_id(test_user_id)
        assert retrieved_user is not None
        assert retrieved_user.id == user_id


def test_list_users(session_maker):
    # Test listing all users
    test_org_id1 = uuid.uuid4()
    test_org_id2 = uuid.uuid4()
    test_user_id1 = uuid.uuid4()
    test_user_id2 = uuid.uuid4()
    with session_maker() as session:
        # Create test users
        user1 = User(id=test_user_id1, current_org_id=test_org_id1)
        user2 = User(id=test_user_id2, current_org_id=test_org_id2)
        session.add_all([user1, user2])
        session.commit()

    # Test listing
    with patch('storage.user_store.session_maker', session_maker):
        users = UserStore.list_users()
        assert len(users) >= 2
        user_ids = [user.id for user in users]
        assert test_user_id1 in user_ids
        assert test_user_id2 in user_ids


def test_get_kwargs_from_settings():
    # Test extracting user kwargs from settings
    settings = Settings(
        language='es',
        enable_sound_notifications=True,
        llm_api_key=SecretStr('test-key'),
    )

    kwargs = UserStore.get_kwargs_from_settings(settings)

    # Should only include fields that exist in User model
    assert 'language' in kwargs
    assert 'enable_sound_notifications' in kwargs
    # Should not include fields that don't exist in User model
    assert 'llm_api_key' not in kwargs


# --- Tests for contact_name resolution in migrate_user() ---
# migrate_user() should use resolve_display_name() to populate contact_name
# from Keycloak name claims, falling back to username only when no real name
# is available. This mirrors the create_user() fix and ensures migrated Org
# records also store the user's actual display name.


class _StopAfterOrgCreation(Exception):
    """Halt migrate_user() after Org creation for contact_name inspection."""

    pass


@pytest.mark.asyncio
async def test_migrate_user_contact_name_uses_name_claim():
    """When user_info has a 'name' claim, migrate_user() should use it for contact_name."""
    user_id = str(uuid.uuid4())
    user_info = {
        'username': 'jdoe',
        'email': 'jdoe@example.com',
        'name': 'John Doe',
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    mock_user_settings = MagicMock()
    mock_user_settings.user_version = 1

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch(
            'storage.user_store.decrypt_legacy_model',
            return_value={'keycloak_user_id': user_id},
        ),
        patch('storage.user_store.UserSettings'),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.migrate_entries',
            new_callable=AsyncMock,
            side_effect=_StopAfterOrgCreation,
        ),
    ):
        with pytest.raises(_StopAfterOrgCreation):
            await UserStore.migrate_user(user_id, mock_user_settings, user_info)

    org = mock_session.add.call_args_list[0][0][0]
    assert isinstance(org, Org)
    assert org.contact_name == 'John Doe'


@pytest.mark.asyncio
async def test_migrate_user_contact_name_uses_given_family_names():
    """When only given_name and family_name are present, migrate_user() should combine them."""
    user_id = str(uuid.uuid4())
    user_info = {
        'username': 'jsmith',
        'email': 'jsmith@example.com',
        'given_name': 'Jane',
        'family_name': 'Smith',
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    mock_user_settings = MagicMock()
    mock_user_settings.user_version = 1

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch(
            'storage.user_store.decrypt_legacy_model',
            return_value={'keycloak_user_id': user_id},
        ),
        patch('storage.user_store.UserSettings'),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.migrate_entries',
            new_callable=AsyncMock,
            side_effect=_StopAfterOrgCreation,
        ),
    ):
        with pytest.raises(_StopAfterOrgCreation):
            await UserStore.migrate_user(user_id, mock_user_settings, user_info)

    org = mock_session.add.call_args_list[0][0][0]
    assert isinstance(org, Org)
    assert org.contact_name == 'Jane Smith'


@pytest.mark.asyncio
async def test_migrate_user_contact_name_falls_back_to_username():
    """When no name claims exist, migrate_user() should fall back to username."""
    user_id = str(uuid.uuid4())
    user_info = {
        'username': 'jdoe',
        'email': 'jdoe@example.com',
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    mock_user_settings = MagicMock()
    mock_user_settings.user_version = 1

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch(
            'storage.user_store.decrypt_legacy_model',
            return_value={'keycloak_user_id': user_id},
        ),
        patch('storage.user_store.UserSettings'),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.migrate_entries',
            new_callable=AsyncMock,
            side_effect=_StopAfterOrgCreation,
        ),
    ):
        with pytest.raises(_StopAfterOrgCreation):
            await UserStore.migrate_user(user_id, mock_user_settings, user_info)

    org = mock_session.add.call_args_list[0][0][0]
    assert isinstance(org, Org)
    assert org.contact_name == 'jdoe'


# --- Tests for contact_name resolution in create_user() ---
# create_user() should use resolve_display_name() to populate contact_name
# from Keycloak name claims, falling back to preferred_username only when
# no real name is available. This ensures Org records store the user's
# actual display name for use in UI and analytics.


@pytest.mark.asyncio
async def test_create_user_contact_name_uses_name_claim():
    """When user_info has a 'name' claim, create_user() should use it for contact_name."""
    user_id = str(uuid.uuid4())
    user_info = {
        'preferred_username': 'jdoe',
        'email': 'jdoe@example.com',
        'name': 'John Doe',
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch.object(
            UserStore,
            'create_default_settings',
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await UserStore.create_user(user_id, user_info)

    assert result is None  # create_default_settings returned None
    # The Org should have been added to the session with the real display name
    org = mock_session.add.call_args_list[0][0][0]
    assert isinstance(org, Org)
    assert org.contact_name == 'John Doe'


@pytest.mark.asyncio
async def test_create_user_contact_name_uses_given_family_names():
    """When only given_name and family_name are present, create_user() should combine them."""
    user_id = str(uuid.uuid4())
    user_info = {
        'preferred_username': 'jsmith',
        'email': 'jsmith@example.com',
        'given_name': 'Jane',
        'family_name': 'Smith',
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch.object(
            UserStore,
            'create_default_settings',
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await UserStore.create_user(user_id, user_info)

    assert result is None
    org = mock_session.add.call_args_list[0][0][0]
    assert isinstance(org, Org)
    assert org.contact_name == 'Jane Smith'


@pytest.mark.asyncio
async def test_create_user_contact_name_falls_back_to_username():
    """When no name claims exist, create_user() should fall back to preferred_username."""
    user_id = str(uuid.uuid4())
    user_info = {
        'preferred_username': 'jdoe',
        'email': 'jdoe@example.com',
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch.object(
            UserStore,
            'create_default_settings',
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await UserStore.create_user(user_id, user_info)

    assert result is None
    org = mock_session.add.call_args_list[0][0][0]
    assert isinstance(org, Org)
    assert org.contact_name == 'jdoe'


# --- Tests for email fields in create_user() ---
# create_user() should populate user.email and user.email_verified from the
# Keycloak user_info, ensuring the user table has the correct email data.


class _StopAfterUserCreation(Exception):
    """Halt create_user() after User creation for email field inspection."""

    pass


@pytest.mark.asyncio
async def test_create_user_sets_email_from_user_info():
    """create_user() should set user.email and user.email_verified from user_info."""
    # Arrange
    user_id = str(uuid.uuid4())
    user_info = {
        'preferred_username': 'testuser',
        'email': 'testuser@example.com',
        'email_verified': True,
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    mock_settings = Settings(language='en')
    mock_role = MagicMock()
    mock_role.id = 1

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch.object(
            UserStore,
            'create_default_settings',
            new_callable=AsyncMock,
            return_value=mock_settings,
        ),
        patch('storage.org_store.OrgStore.get_kwargs_from_settings', return_value={}),
        patch.object(UserStore, 'get_kwargs_from_settings', return_value={}),
        patch('storage.user_store.RoleStore.get_role_by_name', return_value=mock_role),
        patch(
            'storage.org_member_store.OrgMemberStore.get_kwargs_from_settings',
            return_value={'llm_model': None, 'llm_base_url': None},
        ),
        patch.object(
            mock_session,
            'commit',
            side_effect=_StopAfterUserCreation,
        ),
    ):
        # Act
        with pytest.raises(_StopAfterUserCreation):
            await UserStore.create_user(user_id, user_info)

    # Assert - User is the second object added to session (after Org)
    user = mock_session.add.call_args_list[1][0][0]
    assert isinstance(user, User)
    assert user.email == 'testuser@example.com'
    assert user.email_verified is True


@pytest.mark.asyncio
async def test_create_user_handles_missing_email_verified():
    """create_user() should handle missing email_verified in user_info gracefully."""
    # Arrange
    user_id = str(uuid.uuid4())
    user_info = {
        'preferred_username': 'testuser',
        'email': 'testuser@example.com',
        # email_verified is not present
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    mock_settings = Settings(language='en')
    mock_role = MagicMock()
    mock_role.id = 1

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch.object(
            UserStore,
            'create_default_settings',
            new_callable=AsyncMock,
            return_value=mock_settings,
        ),
        patch('storage.org_store.OrgStore.get_kwargs_from_settings', return_value={}),
        patch.object(UserStore, 'get_kwargs_from_settings', return_value={}),
        patch('storage.user_store.RoleStore.get_role_by_name', return_value=mock_role),
        patch(
            'storage.org_member_store.OrgMemberStore.get_kwargs_from_settings',
            return_value={'llm_model': None, 'llm_base_url': None},
        ),
        patch.object(
            mock_session,
            'commit',
            side_effect=_StopAfterUserCreation,
        ),
    ):
        # Act
        with pytest.raises(_StopAfterUserCreation):
            await UserStore.create_user(user_id, user_info)

    # Assert - User should have email but email_verified should be None
    user = mock_session.add.call_args_list[1][0][0]
    assert isinstance(user, User)
    assert user.email == 'testuser@example.com'
    assert user.email_verified is None


# --- Tests for backfill_contact_name on login ---
# Existing users created before the resolve_display_name fix may have
# username-style values in contact_name. The backfill updates these to
# the user's real display name when they next log in, but preserves
# custom values set via the PATCH endpoint.


def _wrap_sync_as_async_session_maker(sync_sm):
    """Wrap a sync session_maker so it can be used in place of a_session_maker."""

    @asynccontextmanager
    async def _async_sm():
        session = sync_sm()
        try:

            class _AsyncWrapper:
                async def execute(self, *args, **kwargs):
                    return session.execute(*args, **kwargs)

                async def commit(self):
                    session.commit()

            yield _AsyncWrapper()
        finally:
            session.close()

    return _async_sm


@pytest.mark.asyncio
async def test_backfill_contact_name_updates_when_matches_preferred_username(
    session_maker,
):
    """When contact_name matches preferred_username and a real name is available, update it."""
    user_id = str(uuid.uuid4())
    # Create org with username-style contact_name (as create_user used to store)
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_name='jdoe',
            contact_email='jdoe@example.com',
        )
        session.add(org)
        session.commit()

    user_info = {
        'preferred_username': 'jdoe',
        'name': 'John Doe',
    }

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.backfill_contact_name(user_id, user_info)

    with session_maker() as session:
        org = session.query(Org).filter(Org.id == uuid.UUID(user_id)).first()
        assert org.contact_name == 'John Doe'


@pytest.mark.asyncio
async def test_backfill_contact_name_updates_when_matches_username(session_maker):
    """When contact_name matches username (migrate_user legacy) and a real name is available, update it."""
    user_id = str(uuid.uuid4())
    # Create org with username-style contact_name (as migrate_user used to store)
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_name='jdoe',
            contact_email='jdoe@example.com',
        )
        session.add(org)
        session.commit()

    user_info = {
        'username': 'jdoe',
        'given_name': 'Jane',
        'family_name': 'Doe',
    }

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.backfill_contact_name(user_id, user_info)

    with session_maker() as session:
        org = session.query(Org).filter(Org.id == uuid.UUID(user_id)).first()
        assert org.contact_name == 'Jane Doe'


@pytest.mark.asyncio
async def test_backfill_contact_name_preserves_custom_value(session_maker):
    """When contact_name differs from both username fields, do not overwrite it."""
    user_id = str(uuid.uuid4())
    # Org has a custom contact_name set via PATCH endpoint
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_name='Custom Corp Name',
            contact_email='jdoe@example.com',
        )
        session.add(org)
        session.commit()

    user_info = {
        'preferred_username': 'jdoe',
        'username': 'jdoe',
        'name': 'John Doe',
    }

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.backfill_contact_name(user_id, user_info)

    with session_maker() as session:
        org = session.query(Org).filter(Org.id == uuid.UUID(user_id)).first()
        assert org.contact_name == 'Custom Corp Name'


# --- Tests for backfill_user_email on login ---
# Existing users created before the email capture fix may have NULL
# email in the User table. The backfill sets User.email from the IDP
# when the user next logs in, but preserves manual changes (non-NULL).


@pytest.mark.asyncio
async def test_backfill_user_email_sets_email_when_null(session_maker):
    """When User.email is NULL, backfill_user_email should set it from user_info."""
    user_id = str(uuid.uuid4())
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_email='jdoe@example.com',
        )
        session.add(org)
        user = User(
            id=uuid.UUID(user_id),
            current_org_id=org.id,
            email=None,
            email_verified=None,
        )
        session.add(user)
        session.commit()

    user_info = {
        'email': 'jdoe@example.com',
        'email_verified': True,
    }

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.backfill_user_email(user_id, user_info)

    with session_maker() as session:
        user = session.query(User).filter(User.id == uuid.UUID(user_id)).first()
        assert user.email == 'jdoe@example.com'
        assert user.email_verified is True


@pytest.mark.asyncio
async def test_backfill_user_email_does_not_overwrite_existing(session_maker):
    """When User.email is already set, backfill_user_email should NOT overwrite it."""
    user_id = str(uuid.uuid4())
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_email='original@example.com',
        )
        session.add(org)
        user = User(
            id=uuid.UUID(user_id),
            current_org_id=org.id,
            email='custom@example.com',
            email_verified=True,
        )
        session.add(user)
        session.commit()

    user_info = {
        'email': 'different@example.com',
        'email_verified': False,
    }

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.backfill_user_email(user_id, user_info)

    with session_maker() as session:
        user = session.query(User).filter(User.id == uuid.UUID(user_id)).first()
        assert user.email == 'custom@example.com'
        assert user.email_verified is True


@pytest.mark.asyncio
async def test_backfill_user_email_sets_verified_when_null(session_maker):
    """When User.email is set but email_verified is NULL, backfill should set email_verified."""
    user_id = str(uuid.uuid4())
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_email='jdoe@example.com',
        )
        session.add(org)
        user = User(
            id=uuid.UUID(user_id),
            current_org_id=org.id,
            email='jdoe@example.com',
            email_verified=None,
        )
        session.add(user)
        session.commit()

    user_info = {
        'email': 'different@example.com',
        'email_verified': True,
    }

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.backfill_user_email(user_id, user_info)

    with session_maker() as session:
        user = session.query(User).filter(User.id == uuid.UUID(user_id)).first()
        # email should NOT be overwritten since it's non-NULL
        assert user.email == 'jdoe@example.com'
        # email_verified should be set since it was NULL
        assert user.email_verified is True


@pytest.mark.asyncio
async def test_create_user_sets_email_verified_false_from_user_info():
    """When user_info has email_verified=False, create_user() should set User.email_verified=False."""
    user_id = str(uuid.uuid4())
    user_info = {
        'preferred_username': 'jsmith',
        'email': 'jsmith@example.com',
        'email_verified': False,
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    mock_settings = Settings(
        language='en',
        llm_api_key=SecretStr('test-key'),
        llm_base_url='http://test.url',
    )

    mock_role = MagicMock()
    mock_role.id = 1

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch.object(
            UserStore,
            'create_default_settings',
            new_callable=AsyncMock,
            return_value=mock_settings,
        ),
        patch('storage.user_store.RoleStore.get_role_by_name', return_value=mock_role),
        patch(
            'storage.org_member_store.OrgMemberStore.get_kwargs_from_settings',
            return_value={'llm_model': None, 'llm_base_url': None},
        ),
    ):
        mock_session.commit.side_effect = _StopAfterUserCreation
        with pytest.raises(_StopAfterUserCreation):
            await UserStore.create_user(user_id, user_info)

    user = mock_session.add.call_args_list[1][0][0]
    assert isinstance(user, User)
    assert user.email == 'jsmith@example.com'
    assert user.email_verified is False


@pytest.mark.asyncio
async def test_create_user_preserves_org_contact_email():
    """create_user() must still set Org.contact_email (no regression)."""
    user_id = str(uuid.uuid4())
    user_info = {
        'preferred_username': 'jdoe',
        'email': 'jdoe@example.com',
        'email_verified': True,
    }

    mock_session = MagicMock()
    mock_sm = MagicMock()
    mock_sm.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_sm.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch('storage.user_store.session_maker', mock_sm),
        patch.object(
            UserStore,
            'create_default_settings',
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        await UserStore.create_user(user_id, user_info)

    org = mock_session.add.call_args_list[0][0][0]
    assert isinstance(org, Org)
    assert org.contact_email == 'jdoe@example.com'


def test_update_current_org_success(session_maker):
    """
    GIVEN: User exists in database
    WHEN: update_current_org is called with new org_id
    THEN: User's current_org_id is updated and user is returned
    """
    # Arrange
    user_id = str(uuid.uuid4())
    initial_org_id = uuid.uuid4()
    new_org_id = uuid.uuid4()

    with session_maker() as session:
        user = User(id=uuid.UUID(user_id), current_org_id=initial_org_id)
        session.add(user)
        session.commit()

    # Act
    with patch('storage.user_store.session_maker', session_maker):
        result = UserStore.update_current_org(user_id, new_org_id)

    # Assert
    assert result is not None
    assert result.current_org_id == new_org_id


def test_update_current_org_user_not_found(session_maker):
    """
    GIVEN: User does not exist in database
    WHEN: update_current_org is called
    THEN: None is returned
    """
    # Arrange
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()

    # Act
    with patch('storage.user_store.session_maker', session_maker):
        result = UserStore.update_current_org(user_id, org_id)

    # Assert
    assert result is None


# --- Tests for update_user_email ---
# update_user_email() should unconditionally overwrite User.email and/or email_verified.
# Unlike backfill_user_email(), it does not check for NULL before writing.


@pytest.mark.asyncio
async def test_update_user_email_overwrites_existing(session_maker):
    """update_user_email() should overwrite existing email and email_verified values."""
    user_id = str(uuid.uuid4())
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_email='old@example.com',
        )
        session.add(org)
        user = User(
            id=uuid.UUID(user_id),
            current_org_id=org.id,
            email='old@example.com',
            email_verified=True,
        )
        session.add(user)
        session.commit()

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.update_user_email(
            user_id, email='new@example.com', email_verified=False
        )

    with session_maker() as session:
        user = session.query(User).filter(User.id == uuid.UUID(user_id)).first()
        assert user.email == 'new@example.com'
        assert user.email_verified is False


@pytest.mark.asyncio
async def test_update_user_email_updates_only_email_verified(session_maker):
    """update_user_email() with email=None should only update email_verified."""
    user_id = str(uuid.uuid4())
    with session_maker() as session:
        org = Org(
            id=uuid.UUID(user_id),
            name=f'user_{user_id}_org',
            contact_email='keep@example.com',
        )
        session.add(org)
        user = User(
            id=uuid.UUID(user_id),
            current_org_id=org.id,
            email='keep@example.com',
            email_verified=False,
        )
        session.add(user)
        session.commit()

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.update_user_email(user_id, email_verified=True)

    with session_maker() as session:
        user = session.query(User).filter(User.id == uuid.UUID(user_id)).first()
        assert user.email == 'keep@example.com'
        assert user.email_verified is True


@pytest.mark.asyncio
async def test_update_user_email_noop_when_both_none():
    """update_user_email() with both args None should not open a session."""
    user_id = str(uuid.uuid4())
    mock_session_maker = MagicMock()

    with patch('storage.user_store.a_session_maker', mock_session_maker):
        await UserStore.update_user_email(user_id, email=None, email_verified=None)

    mock_session_maker.assert_not_called()


@pytest.mark.asyncio
async def test_update_user_email_missing_user_returns_without_error(session_maker):
    """update_user_email() with a non-existent user_id should return without error."""
    user_id = str(uuid.uuid4())

    with patch(
        'storage.user_store.a_session_maker',
        _wrap_sync_as_async_session_maker(session_maker),
    ):
        await UserStore.update_user_email(
            user_id, email='new@example.com', email_verified=False
        )
