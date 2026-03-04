"""
Tests for UserStore following the async pattern from test_api_key_store.py.
Uses SQLite database with standard fixtures.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from storage.org import Org
from storage.user import User
from storage.user_store import UserStore

from openhands.storage.data_models.settings import Settings

# --- Fixtures ---


@pytest.fixture
def mock_litellm_api():
    """Mock LiteLLM API calls to prevent external dependencies."""
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
        mock_client.return_value.__aenter__.return_value.patch.return_value = (
            mock_response
        )
        yield mock_client


# --- Tests for get_kwargs_from_settings ---


def test_get_kwargs_from_settings():
    """Test extracting user kwargs from Settings object."""
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


# --- Tests for create_default_settings ---


@pytest.mark.asyncio
async def test_create_default_settings_no_org_id():
    """Test that create_default_settings returns None when org_id is empty."""
    settings = await UserStore.create_default_settings('', 'test-user-id')
    assert settings is None


@pytest.mark.asyncio
async def test_create_default_settings_with_litellm(mock_litellm_api):
    """Test that create_default_settings works with mocked LiteLLM."""
    org_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    # Mock LiteLlmManager.create_entries to return a Settings object
    mock_settings = Settings(
        language='en',
        llm_api_key=SecretStr('test_api_key'),
        llm_base_url='http://test.url',
        agent='CodeActAgent',
    )

    with patch(
        'storage.lite_llm_manager.LiteLlmManager.create_entries',
        new_callable=AsyncMock,
        return_value=mock_settings,
    ):
        settings = await UserStore.create_default_settings(org_id, user_id)

    # With mock, should return settings with API key from LiteLLM
    assert settings is not None
    assert settings.llm_api_key.get_secret_value() == 'test_api_key'
    assert settings.llm_base_url == 'http://test.url'


# --- Tests for get_user_by_id ---


@pytest.mark.asyncio
async def test_get_user_by_id_existing_user(async_session_maker):
    """Test retrieving an existing user by ID."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    # Create test data
    async with async_session_maker() as session:
        org = Org(id=org_id, name='test-org')
        session.add(org)
        user = User(id=user_id, current_org_id=org_id)
        session.add(user)
        await session.commit()

    # Test retrieval with patched session maker
    with patch('storage.user_store.a_session_maker', async_session_maker):
        result = await UserStore.get_user_by_id(str(user_id))

    assert result is not None
    assert result.id == user_id
    assert result.current_org_id == org_id


@pytest.mark.asyncio
async def test_get_user_by_id_user_not_found(async_session_maker):
    """Test that get_user_by_id returns None for non-existent user."""
    non_existent_id = str(uuid.uuid4())

    with patch('storage.user_store.a_session_maker', async_session_maker):
        # Mock the lock functions to avoid Redis dependency
        with (
            patch.object(UserStore, '_acquire_user_creation_lock', return_value=True),
            patch.object(UserStore, '_release_user_creation_lock', return_value=True),
        ):
            result = await UserStore.get_user_by_id(non_existent_id)

    assert result is None


# --- Tests for get_user_by_email ---


@pytest.mark.asyncio
async def test_get_user_by_email_existing_user(async_session_maker):
    """Test retrieving a user by email."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()
    email = 'test@example.com'

    # Create test data
    async with async_session_maker() as session:
        org = Org(id=org_id, name='test-org')
        session.add(org)
        user = User(id=user_id, current_org_id=org_id, email=email)
        session.add(user)
        await session.commit()

    # Test retrieval
    with patch('storage.user_store.a_session_maker', async_session_maker):
        result = await UserStore.get_user_by_email(email)

    assert result is not None
    assert result.id == user_id
    assert result.email == email


@pytest.mark.asyncio
async def test_get_user_by_email_not_found(async_session_maker):
    """Test that get_user_by_email returns None for non-existent email."""
    with patch('storage.user_store.a_session_maker', async_session_maker):
        result = await UserStore.get_user_by_email('nonexistent@example.com')

    assert result is None


@pytest.mark.asyncio
async def test_get_user_by_email_empty_email(async_session_maker):
    """Test that get_user_by_email returns None for empty email."""
    with patch('storage.user_store.a_session_maker', async_session_maker):
        result = await UserStore.get_user_by_email('')

    assert result is None


@pytest.mark.asyncio
async def test_get_user_by_email_none_email(async_session_maker):
    """Test that get_user_by_email returns None for None email."""
    with patch('storage.user_store.a_session_maker', async_session_maker):
        result = await UserStore.get_user_by_email(None)

    assert result is None


# --- Tests for update_user_email ---


@pytest.mark.asyncio
async def test_update_user_email_overwrites_existing(async_session_maker):
    """Test that update_user_email overwrites existing email and email_verified."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    # Create test data with existing email
    async with async_session_maker() as session:
        org = Org(id=org_id, name='test-org')
        session.add(org)
        user = User(
            id=user_id,
            current_org_id=org_id,
            email='old@example.com',
            email_verified=True,
        )
        session.add(user)
        await session.commit()

    # Update email
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.update_user_email(
            str(user_id), email='new@example.com', email_verified=False
        )

    # Verify update
    async with async_session_maker() as session:
        result = await session.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        assert user.email == 'new@example.com'
        assert user.email_verified is False


@pytest.mark.asyncio
async def test_update_user_email_updates_only_email(async_session_maker):
    """Test that update_user_email can update only email."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    # Create test data
    async with async_session_maker() as session:
        org = Org(id=org_id, name='test-org')
        session.add(org)
        user = User(
            id=user_id,
            current_org_id=org_id,
            email='old@example.com',
            email_verified=False,
        )
        session.add(user)
        await session.commit()

    # Update only email
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.update_user_email(str(user_id), email='new@example.com')

    # Verify update - email_verified should remain unchanged
    async with async_session_maker() as session:
        result = await session.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        assert user.email == 'new@example.com'
        assert user.email_verified is False


@pytest.mark.asyncio
async def test_update_user_email_updates_only_verified(async_session_maker):
    """Test that update_user_email can update only email_verified."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    # Create test data
    async with async_session_maker() as session:
        org = Org(id=org_id, name='test-org')
        session.add(org)
        user = User(
            id=user_id,
            current_org_id=org_id,
            email='keep@example.com',
            email_verified=False,
        )
        session.add(user)
        await session.commit()

    # Update only email_verified
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.update_user_email(str(user_id), email_verified=True)

    # Verify update - email should remain unchanged
    async with async_session_maker() as session:
        result = await session.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        assert user.email == 'keep@example.com'
        assert user.email_verified is True


@pytest.mark.asyncio
async def test_update_user_email_noop_when_both_none():
    """Test that update_user_email does nothing when both args are None."""
    user_id = str(uuid.uuid4())
    mock_session_maker = MagicMock()

    with patch('storage.user_store.a_session_maker', mock_session_maker):
        await UserStore.update_user_email(user_id, email=None, email_verified=None)

    # Session maker should not have been called
    mock_session_maker.assert_not_called()


@pytest.mark.asyncio
async def test_update_user_email_missing_user(async_session_maker):
    """Test that update_user_email handles missing user gracefully."""
    user_id = str(uuid.uuid4())

    # Should not raise exception
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.update_user_email(
            user_id, email='new@example.com', email_verified=True
        )


# --- Tests for backfill_user_email ---


@pytest.mark.asyncio
async def test_backfill_user_email_sets_email_when_null(async_session_maker):
    """Test that backfill_user_email sets email when it is NULL."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    # Create test data with NULL email
    async with async_session_maker() as session:
        org = Org(id=org_id, name='test-org')
        session.add(org)
        user = User(
            id=user_id,
            current_org_id=org_id,
            email=None,
            email_verified=None,
        )
        session.add(user)
        await session.commit()

    user_info = {'email': 'new@example.com', 'email_verified': True}

    # Backfill
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_user_email(str(user_id), user_info)

    # Verify update
    async with async_session_maker() as session:
        result = await session.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        assert user.email == 'new@example.com'
        assert user.email_verified is True


@pytest.mark.asyncio
async def test_backfill_user_email_does_not_overwrite_existing(async_session_maker):
    """Test that backfill_user_email does not overwrite existing email."""
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    # Create test data with existing email
    async with async_session_maker() as session:
        org = Org(id=org_id, name='test-org')
        session.add(org)
        user = User(
            id=user_id,
            current_org_id=org_id,
            email='existing@example.com',
            email_verified=None,
        )
        session.add(user)
        await session.commit()

    user_info = {'email': 'new@example.com', 'email_verified': True}

    # Backfill
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_user_email(str(user_id), user_info)

    # Verify email was NOT overwritten but email_verified was set
    async with async_session_maker() as session:
        result = await session.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        assert user.email == 'existing@example.com'  # Should not be overwritten
        assert user.email_verified is True  # Should be set since it was NULL


@pytest.mark.asyncio
async def test_backfill_user_email_user_not_found(async_session_maker):
    """Test that backfill_user_email handles missing user gracefully."""
    user_id = str(uuid.uuid4())
    user_info = {'email': 'new@example.com', 'email_verified': True}

    # Should not raise exception
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_user_email(user_id, user_info)


# --- Tests for backfill_contact_name ---


@pytest.mark.asyncio
async def test_backfill_contact_name_updates_when_matches_preferred_username(
    async_session_maker,
):
    """Test that backfill_contact_name updates when contact_name matches preferred_username."""
    user_id = uuid.uuid4()

    # Create test org with contact_name = preferred_username
    async with async_session_maker() as session:
        org = Org(
            id=user_id,
            name='test-org',
            contact_name='jdoe',  # This is the username-style value
        )
        session.add(org)
        await session.commit()

    user_info = {
        'preferred_username': 'jdoe',
        'name': 'John Doe',
    }

    # Backfill
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_contact_name(str(user_id), user_info)

    # Verify update
    async with async_session_maker() as session:
        result = await session.execute(select(Org).filter(Org.id == user_id))
        org = result.scalars().first()
        assert org.contact_name == 'John Doe'


@pytest.mark.asyncio
async def test_backfill_contact_name_updates_when_matches_username(
    async_session_maker,
):
    """Test that backfill_contact_name updates when contact_name matches username."""
    user_id = uuid.uuid4()

    # Create test org with contact_name = username
    async with async_session_maker() as session:
        org = Org(
            id=user_id,
            name='test-org',
            contact_name='johnsmith',
        )
        session.add(org)
        await session.commit()

    user_info = {
        'username': 'johnsmith',
        'given_name': 'John',
        'family_name': 'Smith',
    }

    # Backfill
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_contact_name(str(user_id), user_info)

    # Verify update - should combine given and family names
    async with async_session_maker() as session:
        result = await session.execute(select(Org).filter(Org.id == user_id))
        org = result.scalars().first()
        assert org.contact_name == 'John Smith'


@pytest.mark.asyncio
async def test_backfill_contact_name_preserves_custom_value(async_session_maker):
    """Test that backfill_contact_name preserves custom contact_name values."""
    user_id = uuid.uuid4()

    # Create test org with custom contact_name (not matching username)
    async with async_session_maker() as session:
        org = Org(
            id=user_id,
            name='test-org',
            contact_name='Custom Company Name',
        )
        session.add(org)
        await session.commit()

    user_info = {
        'preferred_username': 'jdoe',
        'name': 'John Doe',
    }

    # Backfill
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_contact_name(str(user_id), user_info)

    # Verify contact_name was NOT updated (preserved custom value)
    async with async_session_maker() as session:
        result = await session.execute(select(Org).filter(Org.id == user_id))
        org = result.scalars().first()
        assert org.contact_name == 'Custom Company Name'


@pytest.mark.asyncio
async def test_backfill_contact_name_org_not_found(async_session_maker):
    """Test that backfill_contact_name handles missing org gracefully."""
    user_id = str(uuid.uuid4())
    user_info = {'name': 'John Doe'}

    # Should not raise exception
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_contact_name(user_id, user_info)


@pytest.mark.asyncio
async def test_backfill_contact_name_no_real_name(async_session_maker):
    """Test that backfill_contact_name does nothing when no real name is available."""
    user_id = uuid.uuid4()

    # Create test org
    async with async_session_maker() as session:
        org = Org(
            id=user_id,
            name='test-org',
            contact_name='jdoe',
        )
        session.add(org)
        await session.commit()

    user_info = {
        'preferred_username': 'jdoe',
        # No 'name', 'given_name', or 'family_name'
    }

    # Backfill
    with patch('storage.user_store.a_session_maker', async_session_maker):
        await UserStore.backfill_contact_name(str(user_id), user_info)

    # Verify contact_name was NOT updated
    async with async_session_maker() as session:
        result = await session.execute(select(Org).filter(Org.id == user_id))
        org = result.scalars().first()
        assert org.contact_name == 'jdoe'


# --- Tests for update_current_org ---


@pytest.mark.asyncio
async def test_update_current_org_success(async_session_maker):
    """Test updating a user's current organization."""
    user_id = uuid.uuid4()
    initial_org_id = uuid.uuid4()
    new_org_id = uuid.uuid4()

    # Create test data
    async with async_session_maker() as session:
        org1 = Org(id=initial_org_id, name='org1')
        org2 = Org(id=new_org_id, name='org2')
        session.add_all([org1, org2])
        user = User(id=user_id, current_org_id=initial_org_id)
        session.add(user)
        await session.commit()

    # Update current org
    with patch('storage.user_store.a_session_maker', async_session_maker):
        result = await UserStore.update_current_org(str(user_id), new_org_id)

    assert result is not None
    assert result.current_org_id == new_org_id


@pytest.mark.asyncio
async def test_update_current_org_user_not_found(async_session_maker):
    """Test that update_current_org returns None for non-existent user."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()

    with patch('storage.user_store.a_session_maker', async_session_maker):
        result = await UserStore.update_current_org(user_id, org_id)

    assert result is None


# --- Tests for list_users ---


@pytest.mark.asyncio
async def test_list_users(async_session_maker):
    """Test listing all users."""
    user_id1 = uuid.uuid4()
    user_id2 = uuid.uuid4()
    org_id1 = uuid.uuid4()
    org_id2 = uuid.uuid4()

    # Create test data
    async with async_session_maker() as session:
        org1 = Org(id=org_id1, name='org1')
        org2 = Org(id=org_id2, name='org2')
        session.add_all([org1, org2])
        user1 = User(id=user_id1, current_org_id=org_id1)
        user2 = User(id=user_id2, current_org_id=org_id2)
        session.add_all([user1, user2])
        await session.commit()

    # List users
    with patch('storage.user_store.a_session_maker', async_session_maker):
        users = await UserStore.list_users()

    assert len(users) >= 2
    user_ids = [user.id for user in users]
    assert user_id1 in user_ids
    assert user_id2 in user_ids


# --- Tests for _has_custom_settings ---


def test_has_custom_settings_custom_base_url():
    """Test that custom base_url is detected as custom settings."""
    from storage.user_settings import UserSettings

    user_settings = UserSettings(
        keycloak_user_id='test',
        llm_base_url='https://custom.api.example.com',
        llm_model='some-model',
    )

    result = UserStore._has_custom_settings(user_settings, old_user_version=1)

    assert result is True


def test_has_custom_settings_no_model():
    """Test that no model set means using defaults."""
    from storage.user_settings import UserSettings

    user_settings = UserSettings(
        keycloak_user_id='test',
        llm_base_url=None,
        llm_model=None,
    )

    result = UserStore._has_custom_settings(user_settings, old_user_version=1)

    assert result is False


def test_has_custom_settings_empty_model():
    """Test that empty model string means using defaults."""
    from storage.user_settings import UserSettings

    user_settings = UserSettings(
        keycloak_user_id='test',
        llm_base_url=None,
        llm_model='   ',  # whitespace only
    )

    result = UserStore._has_custom_settings(user_settings, old_user_version=1)

    assert result is False


# --- Tests for _create_user_settings_from_entities ---


def test_create_user_settings_from_entities():
    """Test creating UserSettings from OrgMember, User, and Org entities."""
    user_id = str(uuid.uuid4())

    # Create mock entities
    org_member = MagicMock()
    org_member.llm_api_key = SecretStr('test-api-key')
    org_member.llm_api_key_for_byor = None
    org_member.llm_model = 'claude-3-5-sonnet'
    org_member.llm_base_url = 'https://api.example.com'
    org_member.max_iterations = 50

    user = MagicMock()
    user.accepted_tos = None
    user.enable_sound_notifications = True
    user.language = 'en'
    user.user_consents_to_analytics = True
    user.email = 'test@example.com'
    user.email_verified = True
    user.git_user_name = 'testuser'
    user.git_user_email = 'test@git.com'

    org = MagicMock()
    org.agent = 'CodeActAgent'
    org.security_analyzer = 'mock-analyzer'
    org.confirmation_mode = False
    org.remote_runtime_resource_factor = 1.0
    org.enable_default_condenser = True
    org.billing_margin = 0.0
    org.enable_proactive_conversation_starters = True
    org.sandbox_base_container_image = None
    org.sandbox_runtime_container_image = None
    org.org_version = 1
    org.mcp_config = None
    org.search_api_key = None
    org.sandbox_api_key = None
    org.max_budget_per_task = None
    org.enable_solvability_analysis = False
    org.v1_enabled = True
    org.condenser_max_size = None
    org.default_llm_model = 'default-model'
    org.default_llm_base_url = 'https://default.api.com'
    org.default_max_iterations = 100

    result = UserStore._create_user_settings_from_entities(
        user_id, org_member, user, org
    )

    assert result.keycloak_user_id == user_id
    assert result.llm_api_key == 'test-api-key'
    assert result.llm_model == 'claude-3-5-sonnet'
    assert result.language == 'en'
    assert result.email == 'test@example.com'


def test_create_user_settings_from_entities_with_org_fallback():
    """Test that _create_user_settings_from_entities falls back to org defaults."""
    user_id = str(uuid.uuid4())

    # Create mock entities with None in OrgMember
    org_member = MagicMock()
    org_member.llm_api_key = None
    org_member.llm_api_key_for_byor = None
    org_member.llm_model = None  # Should fall back to org.default_llm_model
    org_member.llm_base_url = None  # Should fall back to org.default_llm_base_url
    org_member.max_iterations = None  # Should fall back to org.default_max_iterations

    user = MagicMock()
    user.accepted_tos = None
    user.enable_sound_notifications = False
    user.language = 'es'
    user.user_consents_to_analytics = False
    user.email = None
    user.email_verified = None
    user.git_user_name = None
    user.git_user_email = None

    org = MagicMock()
    org.agent = 'CodeActAgent'
    org.security_analyzer = None
    org.confirmation_mode = True
    org.remote_runtime_resource_factor = 2.0
    org.enable_default_condenser = False
    org.billing_margin = 0.1
    org.enable_proactive_conversation_starters = False
    org.sandbox_base_container_image = 'custom-image'
    org.sandbox_runtime_container_image = None
    org.org_version = 2
    org.mcp_config = {'key': 'value'}
    org.search_api_key = SecretStr('search-key')
    org.sandbox_api_key = None
    org.max_budget_per_task = 10.0
    org.enable_solvability_analysis = True
    org.v1_enabled = False
    org.condenser_max_size = 1000
    # Org defaults
    org.default_llm_model = 'default-model'
    org.default_llm_base_url = 'https://default.api.com'
    org.default_max_iterations = 100

    result = UserStore._create_user_settings_from_entities(
        user_id, org_member, user, org
    )

    # Should have fallen back to org defaults
    assert result.llm_model == 'default-model'
    assert result.llm_base_url == 'https://default.api.com'
    assert result.max_iterations == 100
    assert result.language == 'es'
    assert result.search_api_key == 'search-key'


# --- Tests for Redis lock functions (mocked) ---


@pytest.mark.asyncio
async def test_acquire_user_creation_lock_no_redis():
    """Test that _acquire_user_creation_lock returns True when Redis is unavailable."""
    with patch.object(UserStore, '_get_redis_client', return_value=None):
        result = await UserStore._acquire_user_creation_lock('test-user-id')

    assert result is True


@pytest.mark.asyncio
async def test_acquire_user_creation_lock_acquired():
    """Test that _acquire_user_creation_lock returns True when lock is acquired."""
    mock_redis = AsyncMock()
    mock_redis.set.return_value = True

    with patch.object(UserStore, '_get_redis_client', return_value=mock_redis):
        result = await UserStore._acquire_user_creation_lock('test-user-id')

    assert result is True
    mock_redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_acquire_user_creation_lock_not_acquired():
    """Test that _acquire_user_creation_lock returns False when lock is not acquired."""
    mock_redis = AsyncMock()
    mock_redis.set.return_value = False

    with patch.object(UserStore, '_get_redis_client', return_value=mock_redis):
        result = await UserStore._acquire_user_creation_lock('test-user-id')

    assert result is False


@pytest.mark.asyncio
async def test_release_user_creation_lock_no_redis():
    """Test that _release_user_creation_lock returns True when Redis is unavailable."""
    with patch.object(UserStore, '_get_redis_client', return_value=None):
        result = await UserStore._release_user_creation_lock('test-user-id')

    assert result is True


@pytest.mark.asyncio
async def test_release_user_creation_lock_released():
    """Test that _release_user_creation_lock returns True when lock is released."""
    mock_redis = AsyncMock()
    mock_redis.delete.return_value = 1

    with patch.object(UserStore, '_get_redis_client', return_value=mock_redis):
        result = await UserStore._release_user_creation_lock('test-user-id')

    assert result is True
    mock_redis.delete.assert_called_once()
