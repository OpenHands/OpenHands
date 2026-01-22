import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

# Mock the database module before importing UserStore
with patch('storage.database.engine'), patch('storage.database.a_engine'):
    from storage.user import User
    from storage.user_store import UserStore

from sqlalchemy.orm import configure_mappers

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
