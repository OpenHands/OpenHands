from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr
from server.constants import (
    CURRENT_USER_SETTINGS_VERSION,
    LITE_LLM_API_URL,
    LITE_LLM_TEAM_ID,
    get_default_litellm_model,
)
from storage.saas_settings_store import SaasSettingsStore
from storage.user_settings import UserSettings

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.settings import Settings


@pytest.fixture
def mock_litellm_get_response():
    mock_response = AsyncMock()
    mock_response.is_success = True
    mock_response.json = MagicMock(return_value={'user_info': {}})
    return mock_response


@pytest.fixture
def mock_litellm_post_response():
    mock_response = AsyncMock()
    mock_response.is_success = True
    mock_response.json = MagicMock(return_value={'key': 'test_api_key'})
    return mock_response


@pytest.fixture
def mock_litellm_api(mock_litellm_get_response, mock_litellm_post_response):
    api_key_patch = patch('storage.saas_settings_store.LITE_LLM_API_KEY', 'test_key')
    api_url_patch = patch(
        'storage.saas_settings_store.LITE_LLM_API_URL', 'http://test.url'
    )
    team_id_patch = patch('storage.saas_settings_store.LITE_LLM_TEAM_ID', 'test_team')
    client_patch = patch('httpx.AsyncClient')

    with api_key_patch, api_url_patch, team_id_patch, client_patch as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_litellm_get_response
        )
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_litellm_post_response
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


@pytest.fixture
def mock_github_user():
    with patch(
        'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
        AsyncMock(return_value={'attributes': {'github_id': ['12345']}}),
    ) as mock_github:
        yield mock_github


@pytest.fixture
def mock_config():
    config = MagicMock(spec=OpenHandsConfig)
    config.jwt_secret = SecretStr('test_secret')
    config.file_store = 'google_cloud'
    config.file_store_path = 'bucket'
    return config


@pytest.fixture
def settings_store(session_maker, mock_config):
    store = SaasSettingsStore('user-id', session_maker, mock_config)

    # Patch the store method directly to filter out email and email_verified
    original_load = store.load
    original_create_default = store.create_default_settings
    original_update_litellm = store.update_settings_with_litellm_default

    # Patch the load method to add email and email_verified
    async def patched_load():
        settings = await original_load()
        if settings:
            # Add email and email_verified fields to mimic SaasUserAuth behavior
            settings.email = 'test@example.com'
            settings.email_verified = True
        return settings

    # Patch the create_default_settings method to add email and email_verified
    async def patched_create_default(settings):
        settings = await original_create_default(settings)
        if settings:
            # Add email and email_verified fields to mimic SaasUserAuth behavior
            settings.email = 'test@example.com'
            settings.email_verified = True
        return settings

    # Patch the update_settings_with_litellm_default method
    async def patched_update_litellm(settings):
        updated_settings = await original_update_litellm(settings)
        if updated_settings:
            # Add email and email_verified fields to mimic SaasUserAuth behavior
            updated_settings.email = 'test@example.com'
            updated_settings.email_verified = True
        return updated_settings

    # Patch the store method to filter out email and email_verified
    async def patched_store(item):
        if item:
            # Make a copy of the item without email and email_verified
            item_dict = item.model_dump(context={'expose_secrets': True})
            if 'email' in item_dict:
                del item_dict['email']
            if 'email_verified' in item_dict:
                del item_dict['email_verified']
            if 'secrets_store' in item_dict:
                del item_dict['secrets_store']

            # Continue with the original implementation
            with store.session_maker() as session:
                existing = None
                if item_dict:
                    store._encrypt_kwargs(item_dict)
                    query = session.query(UserSettings).filter(
                        UserSettings.keycloak_user_id == store.user_id
                    )

                    # First check if we have an existing entry in the new table
                    existing = query.first()

                if existing:
                    # Update existing entry
                    for key, value in item_dict.items():
                        if key in existing.__class__.__table__.columns:
                            setattr(existing, key, value)
                    existing.user_version = CURRENT_USER_SETTINGS_VERSION
                    session.merge(existing)
                else:
                    item_dict['keycloak_user_id'] = store.user_id
                    item_dict['user_version'] = CURRENT_USER_SETTINGS_VERSION
                    settings = UserSettings(**item_dict)
                    session.add(settings)
                session.commit()

    # Replace the methods with our patched versions
    store.store = patched_store
    store.load = patched_load
    store.create_default_settings = patched_create_default
    store.update_settings_with_litellm_default = patched_update_litellm
    return store


@pytest.mark.asyncio
async def test_store_and_load_keycloak_user(settings_store):
    # Set a UUID-like Keycloak user ID
    settings_store.user_id = '550e8400-e29b-41d4-a716-446655440000'
    settings = Settings(
        llm_api_key=SecretStr('secret_key'),
        llm_base_url=LITE_LLM_API_URL,
        agent='smith',
        email='test@example.com',
        email_verified=True,
    )

    await settings_store.store(settings)

    # Load and verify settings
    loaded_settings = await settings_store.load()
    assert loaded_settings is not None
    assert loaded_settings.llm_api_key.get_secret_value() == 'secret_key'
    assert loaded_settings.agent == 'smith'

    # Verify it was stored in user_settings table with keycloak_user_id
    with settings_store.session_maker() as session:
        stored = (
            session.query(UserSettings)
            .filter(
                UserSettings.keycloak_user_id == '550e8400-e29b-41d4-a716-446655440000'
            )
            .first()
        )
        assert stored is not None
        assert stored.agent == 'smith'


@pytest.mark.asyncio
async def test_load_returns_default_when_not_found(
    settings_store, mock_litellm_api, mock_stripe, mock_github_user, session_maker
):
    file_store = MagicMock()
    file_store.read.side_effect = FileNotFoundError()

    with (
        patch(
            'storage.saas_settings_store.get_file_store',
            MagicMock(return_value=file_store),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        loaded_settings = await settings_store.load()
        assert loaded_settings is not None
        assert loaded_settings.language == 'en'
        assert loaded_settings.agent == 'CodeActAgent'
        assert loaded_settings.llm_api_key.get_secret_value() == 'test_api_key'
        assert loaded_settings.llm_base_url == 'http://test.url'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default(
    settings_store, mock_litellm_api, session_maker
):
    settings = Settings()
    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'testy@tester.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        settings = await settings_store.update_settings_with_litellm_default(settings)

    assert settings.agent == 'CodeActAgent'
    assert settings.llm_api_key
    assert settings.llm_api_key.get_secret_value() == 'test_api_key'
    assert settings.llm_base_url == 'http://test.url'

    # Get the actual call arguments
    call_args = mock_litellm_api.return_value.__aenter__.return_value.post.call_args[1]

    # Check that the URL and most of the JSON payload match what we expect
    assert call_args['json']['user_email'] == 'testy@tester.com'
    assert call_args['json']['models'] == []
    assert call_args['json']['max_budget'] == 10.0
    assert call_args['json']['user_id'] == 'user-id'
    assert call_args['json']['teams'] == ['test_team']
    assert call_args['json']['auto_create_key'] is True
    assert call_args['json']['send_invite_email'] is False
    assert call_args['json']['metadata']['version'] == CURRENT_USER_SETTINGS_VERSION
    assert 'model' in call_args['json']['metadata']


@pytest.mark.asyncio
async def test_create_default_settings_no_user_id():
    store = SaasSettingsStore('', MagicMock(), MagicMock())
    settings = await store.create_default_settings(None)
    assert settings is None


@pytest.mark.asyncio
async def test_create_default_settings_require_payment_enabled(
    settings_store, mock_stripe
):
    # Mock stripe_service.has_payment_method to return False
    with (
        patch('storage.saas_settings_store.REQUIRE_PAYMENT', True),
        patch(
            'stripe.Customer.list_payment_methods_async',
            AsyncMock(return_value=MagicMock(data=[])),
        ),
        patch(
            'integrations.stripe_service.session_maker', settings_store.session_maker
        ),
    ):
        settings = await settings_store.create_default_settings(None)
        assert settings is None


@pytest.mark.asyncio
async def test_create_default_settings_require_payment_disabled(
    settings_store, mock_stripe, mock_github_user, mock_litellm_api, session_maker
):
    # Even without payment method, should get default settings when REQUIRE_PAYMENT is False
    file_store = MagicMock()
    file_store.read.side_effect = FileNotFoundError()
    with (
        patch('storage.saas_settings_store.REQUIRE_PAYMENT', False),
        patch(
            'stripe.Customer.list_payment_methods_async',
            AsyncMock(return_value=MagicMock(data=[])),
        ),
        patch(
            'storage.saas_settings_store.get_file_store',
            MagicMock(return_value=file_store),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        settings = await settings_store.create_default_settings(None)
        assert settings is not None
        assert settings.language == 'en'


@pytest.mark.asyncio
async def test_create_default_lite_llm_settings_no_api_config(settings_store):
    with (
        patch('storage.saas_settings_store.LITE_LLM_API_KEY', None),
        patch('storage.saas_settings_store.LITE_LLM_API_URL', None),
    ):
        settings = Settings()
        settings = await settings_store.update_settings_with_litellm_default(settings)


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_error(settings_store):
    with patch(
        'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
        AsyncMock(return_value={'email': 'duplicate@example.com'}),
    ):
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                AsyncMock(
                    json=MagicMock(
                        return_value={'user_info': {'max_budget': 10, 'spend': 5}}
                    )
                )
            )
            mock_client.return_value.__aenter__.return_value.post.return_value.is_success = False
            settings = Settings()
            settings = await settings_store.update_settings_with_litellm_default(
                settings
            )
            assert settings is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'status_code,user_info_response,should_succeed',
    [
        # 200 OK with user info - existing user (v1.79.x and v1.80+ behavior)
        (200, {'user_info': {'max_budget': 10, 'spend': 5}}, True),
        # 200 OK with empty user info - new user (v1.79.x behavior)
        (200, {'user_info': None}, True),
        # 404 Not Found - new user (v1.80+ behavior)
        (404, None, True),
        # 500 Internal Server Error - should fail
        (500, None, False),
    ],
)
async def test_update_settings_with_litellm_default_handles_user_info_responses(
    settings_store, session_maker, status_code, user_info_response, should_succeed
):
    """Test that various LiteLLM user/info responses are handled correctly.

    LiteLLM API behavior changed between versions:
    - v1.79.x and earlier: GET /user/info always succeeds with empty user_info
    - v1.80.x and later: GET /user/info returns 404 for non-existent users
    """
    mock_get_response = MagicMock()
    mock_get_response.status_code = status_code
    if user_info_response is not None:
        mock_get_response.json = MagicMock(return_value=user_info_response)
        mock_get_response.raise_for_status = MagicMock()
    else:
        mock_get_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                'Error', request=MagicMock(), response=mock_get_response
            )
            if status_code >= 500
            else None
        )

    # Mock successful responses for POST operations (delete and create)
    mock_post_response = MagicMock()
    mock_post_response.is_success = True
    mock_post_response.json = MagicMock(return_value={'key': 'new_user_api_key'})

    with (
        patch('storage.saas_settings_store.LITE_LLM_API_KEY', 'test_key'),
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://test.url'),
        patch('storage.saas_settings_store.LITE_LLM_TEAM_ID', 'test_team'),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'testuser@example.com'}),
        ),
        patch('httpx.AsyncClient') as mock_client,
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Set up the mock client
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_get_response
        )
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_post_response
        )

        settings = Settings()
        if should_succeed:
            settings = await settings_store.update_settings_with_litellm_default(
                settings
            )
            assert settings is not None
            assert settings.llm_api_key is not None
            assert settings.llm_api_key.get_secret_value() == 'new_user_api_key'
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await settings_store.update_settings_with_litellm_default(settings)


@pytest.mark.asyncio
async def test_update_settings_with_litellm_retry_on_duplicate_email(
    settings_store, mock_litellm_api, session_maker
):
    # First response is a delete and succeeds
    mock_delete_response = MagicMock()
    mock_delete_response.is_success = True
    mock_delete_response.status_code = 200

    # Second response fails with duplicate email error
    mock_error_response = MagicMock()
    mock_error_response.is_success = False
    mock_error_response.status_code = 400
    mock_error_response.text = 'User with this email already exists'

    # Thire response succeeds with no email
    mock_success_response = MagicMock()
    mock_success_response.is_success = True
    mock_success_response.json = MagicMock(return_value={'key': 'new_test_api_key'})

    # Set up mocks
    post_mock = AsyncMock()
    post_mock.side_effect = [
        mock_delete_response,
        mock_error_response,
        mock_success_response,
    ]
    mock_litellm_api.return_value.__aenter__.return_value.post = post_mock

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'duplicate@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        settings = Settings()
        settings = await settings_store.update_settings_with_litellm_default(settings)

    assert settings is not None
    assert settings.llm_api_key
    assert settings.llm_api_key.get_secret_value() == 'new_test_api_key'

    # Verify second call was with email
    second_call_args = post_mock.call_args_list[1][1]
    assert second_call_args['json']['user_email'] == 'duplicate@example.com'

    # Verify third call was with None for email
    third_call_args = post_mock.call_args_list[2][1]
    assert third_call_args['json']['user_email'] is None


@pytest.mark.asyncio
async def test_create_user_in_lite_llm(settings_store):
    # Test the _create_user_in_lite_llm method directly
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.is_success = True
    mock_client.post.return_value = mock_response
    test_model = 'custom-model/test-model'

    # Test with email
    await settings_store._create_user_in_lite_llm(
        mock_client, 'test@example.com', 50, 10, test_model
    )

    # Get the actual call arguments
    call_args = mock_client.post.call_args[1]

    # Check that the URL and most of the JSON payload match what we expect
    assert call_args['json']['user_email'] == 'test@example.com'
    assert call_args['json']['models'] == []
    assert call_args['json']['max_budget'] == 50
    assert call_args['json']['spend'] == 10
    assert call_args['json']['user_id'] == 'user-id'
    assert call_args['json']['teams'] == [LITE_LLM_TEAM_ID]
    assert call_args['json']['auto_create_key'] is True
    assert call_args['json']['send_invite_email'] is False
    assert call_args['json']['metadata']['version'] == CURRENT_USER_SETTINGS_VERSION
    assert call_args['json']['metadata']['model'] == test_model

    # Test with None email
    mock_client.post.reset_mock()
    await settings_store._create_user_in_lite_llm(mock_client, None, 25, 15, test_model)

    # Get the actual call arguments
    call_args = mock_client.post.call_args[1]

    # Check that the URL and most of the JSON payload match what we expect
    assert call_args['json']['user_email'] is None
    assert call_args['json']['models'] == []
    assert call_args['json']['max_budget'] == 25
    assert call_args['json']['spend'] == 15
    assert call_args['json']['user_id'] == str(settings_store.user_id)
    assert call_args['json']['teams'] == [LITE_LLM_TEAM_ID]
    assert call_args['json']['auto_create_key'] is True
    assert call_args['json']['send_invite_email'] is False
    assert call_args['json']['metadata']['version'] == CURRENT_USER_SETTINGS_VERSION
    assert call_args['json']['metadata']['model'] == test_model

    # Verify response is returned correctly
    assert (
        await settings_store._create_user_in_lite_llm(
            mock_client, 'email@test.com', 30, 7, test_model
        )
        == mock_response
    )


@pytest.mark.asyncio
async def test_encryption(settings_store):
    settings_store.user_id = 'mock-id'  # GitHub user ID
    settings = Settings(
        llm_api_key=SecretStr('secret_key'),
        agent='smith',
        llm_base_url=LITE_LLM_API_URL,
        email='test@example.com',
        email_verified=True,
    )
    await settings_store.store(settings)
    with settings_store.session_maker() as session:
        stored = (
            session.query(UserSettings)
            .filter(UserSettings.keycloak_user_id == 'mock-id')
            .first()
        )
        # The stored key should be encrypted
        assert stored.llm_api_key != 'secret_key'
        # But we should be able to decrypt it when loading
        loaded_settings = await settings_store.load()
        assert loaded_settings.llm_api_key.get_secret_value() == 'secret_key'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_preserves_custom_model(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has a custom LLM model set
    custom_model = 'anthropic/claude-3-5-sonnet-20241022'
    settings = Settings(llm_model=custom_model)

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Custom model is preserved
    assert updated_settings is not None
    assert updated_settings.llm_model == custom_model
    assert updated_settings.agent == 'CodeActAgent'
    assert updated_settings.llm_api_key is not None

    # Assert: LiteLLM metadata contains user's custom model
    call_args = mock_litellm_api.return_value.__aenter__.return_value.post.call_args[1]
    assert call_args['json']['metadata']['model'] == custom_model


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_uses_default_when_no_model(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has no model set (new user scenario)
    settings = Settings()

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'newuser@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default model is assigned
    assert updated_settings is not None
    expected_default = get_default_litellm_model()
    assert updated_settings.llm_model == expected_default
    assert updated_settings.agent == 'CodeActAgent'

    # Assert: LiteLLM metadata contains default model
    call_args = mock_litellm_api.return_value.__aenter__.return_value.post.call_args[1]
    assert call_args['json']['metadata']['model'] == expected_default


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_handles_empty_string_model(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has empty string as model (edge case)
    settings = Settings(llm_model='')

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default model is used (empty string treated as no model)
    assert updated_settings is not None
    expected_default = get_default_litellm_model()
    assert updated_settings.llm_model == expected_default


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_handles_whitespace_model(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has whitespace-only model (edge case)
    settings = Settings(llm_model='   ')

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default model is used (whitespace treated as no model)
    assert updated_settings is not None
    expected_default = get_default_litellm_model()
    assert updated_settings.llm_model == expected_default


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_preserves_custom_api_key(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has a custom API key and custom model (so has_custom=True)
    custom_api_key = 'sk-custom-user-api-key-12345'
    custom_model = 'gpt-4'
    settings = Settings(llm_model=custom_model, llm_api_key=SecretStr(custom_api_key))

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Custom API key is preserved when user has custom settings
    assert updated_settings is not None
    assert updated_settings.llm_api_key.get_secret_value() == custom_api_key
    assert updated_settings.llm_api_key.get_secret_value() != 'test_api_key'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_preserves_custom_base_url(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has a custom base URL
    custom_base_url = 'https://api.custom-llm-provider.com/v1'
    settings = Settings(llm_base_url=custom_base_url)

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Custom base URL is preserved
    assert updated_settings is not None
    assert updated_settings.llm_base_url == custom_base_url
    assert updated_settings.llm_base_url != LITE_LLM_API_URL


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_preserves_custom_api_key_and_base_url(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has both custom API key and base URL
    custom_api_key = 'sk-custom-user-api-key-67890'
    custom_base_url = 'https://api.another-llm-provider.com/v1'
    custom_model = 'openai/gpt-4'
    settings = Settings(
        llm_model=custom_model,
        llm_api_key=SecretStr(custom_api_key),
        llm_base_url=custom_base_url,
    )

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: All custom settings are preserved
    assert updated_settings is not None
    assert updated_settings.llm_model == custom_model
    assert updated_settings.llm_api_key.get_secret_value() == custom_api_key
    assert updated_settings.llm_base_url == custom_base_url


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_uses_default_api_key_when_none(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has no API key set
    settings = Settings(llm_api_key=None)

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default LiteLLM API key is assigned
    assert updated_settings is not None
    assert updated_settings.llm_api_key is not None
    assert updated_settings.llm_api_key.get_secret_value() == 'test_api_key'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_uses_default_base_url_when_none(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has no base URL set
    settings = Settings(llm_base_url=None)

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://test.url'),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default LiteLLM base URL is assigned (using mocked value)
    assert updated_settings is not None
    assert updated_settings.llm_base_url == 'http://test.url'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_handles_empty_api_key(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has empty string as API key (edge case)
    settings = Settings(llm_api_key=SecretStr(''))

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default API key is used (empty string treated as no key)
    assert updated_settings is not None
    assert updated_settings.llm_api_key.get_secret_value() == 'test_api_key'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_handles_empty_base_url(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has empty string as base URL (edge case)
    settings = Settings(llm_base_url='')

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://test.url'),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default base URL is used (empty string treated as no URL)
    assert updated_settings is not None
    assert updated_settings.llm_base_url == 'http://test.url'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_handles_whitespace_api_key(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has whitespace-only API key (edge case)
    settings = Settings(llm_api_key=SecretStr('   '))

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default API key is used (whitespace treated as no key)
    assert updated_settings is not None
    assert updated_settings.llm_api_key.get_secret_value() == 'test_api_key'


@pytest.mark.asyncio
async def test_update_settings_with_litellm_default_handles_whitespace_base_url(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User has whitespace-only base URL (edge case)
    settings = Settings(llm_base_url='   ')

    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('storage.saas_settings_store.session_maker', session_maker),
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://test.url'),
    ):
        # Act: Update settings with LiteLLM defaults
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

    # Assert: Default base URL is used (whitespace treated as no URL)
    assert updated_settings is not None
    assert updated_settings.llm_base_url == 'http://test.url'


# Tests for version migration and helper methods


@pytest.mark.asyncio
async def test_has_custom_settings_with_custom_base_url(settings_store):
    # Arrange: User with custom base URL (BYOR)
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(llm_base_url='http://custom.url')

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: Custom base URL detected
        assert has_custom is True


@pytest.mark.asyncio
async def test_has_custom_settings_with_default_base_url(settings_store):
    # Arrange: User with default base URL
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(llm_base_url='http://default.url')

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: No custom settings (no model set)
        assert has_custom is False


@pytest.mark.asyncio
async def test_has_custom_settings_with_no_model(settings_store):
    # Arrange: User with no model set
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(llm_model=None, llm_base_url='http://default.url')

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: No custom settings (using defaults)
        assert has_custom is False


@pytest.mark.asyncio
async def test_has_custom_settings_with_empty_model(settings_store):
    # Arrange: User with empty model
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(llm_model='', llm_base_url='http://default.url')

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: No custom settings (empty treated as no model)
        assert has_custom is False


@pytest.mark.asyncio
async def test_has_custom_settings_with_whitespace_model(settings_store):
    # Arrange: User with whitespace-only model
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(llm_model='   ', llm_base_url='http://default.url')

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: No custom settings (whitespace treated as no model)
        assert has_custom is False


@pytest.mark.asyncio
async def test_has_custom_settings_with_custom_model(settings_store):
    # Arrange: User with custom model
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(llm_model='gpt-4', llm_base_url='http://default.url')

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: Custom model detected
        assert has_custom is True


@pytest.mark.asyncio
async def test_has_custom_settings_matches_old_default_model(settings_store):
    # Arrange: User with old version and model matching old default
    with (
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'),
        patch('server.constants.CURRENT_USER_SETTINGS_VERSION', 5),
        patch(
            'server.constants.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022'},
        ),
    ):
        settings = Settings(
            llm_model='litellm_proxy/prod/claude-3-5-sonnet-20241022',
            llm_base_url='http://default.url',
        )

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, 1)

        # Assert: Matches old default, so not custom
        assert has_custom is False


@pytest.mark.asyncio
async def test_has_custom_settings_matches_old_default_by_base_name(settings_store):
    # Arrange: User with old version and model matching old default by base name
    with (
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'),
        patch('server.constants.CURRENT_USER_SETTINGS_VERSION', 5),
        patch(
            'server.constants.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022'},
        ),
    ):
        settings = Settings(
            llm_model='anthropic/claude-3-5-sonnet-20241022',
            llm_base_url='http://default.url',
        )

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, 1)

        # Assert: Matches old default by base name, so not custom
        assert has_custom is False


@pytest.mark.asyncio
async def test_has_custom_settings_with_old_version_but_custom_model(settings_store):
    # Arrange: User with old version but custom model
    with (
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'),
        patch('server.constants.CURRENT_USER_SETTINGS_VERSION', 5),
        patch(
            'server.constants.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022'},
        ),
    ):
        settings = Settings(llm_model='gpt-4', llm_base_url='http://default.url')

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, 1)

        # Assert: Custom model detected
        assert has_custom is True


@pytest.mark.asyncio
async def test_has_custom_settings_with_current_version(settings_store):
    # Arrange: User with current version
    with (
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'),
        patch('server.constants.CURRENT_USER_SETTINGS_VERSION', 5),
        patch(
            'server.constants.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022', 5: 'claude-opus-4-5-20251101'},
        ),
    ):
        settings = Settings(
            llm_model='claude-3-5-sonnet-20241022', llm_base_url='http://default.url'
        )

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, 5)

        # Assert: Current version, so model is custom (not old default)
        assert has_custom is True


@pytest.mark.asyncio
async def test_has_custom_settings_with_none_version(settings_store):
    # Arrange: User with no version
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(
            llm_model='claude-3-5-sonnet-20241022', llm_base_url='http://default.url'
        )

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: No version, so model is custom
        assert has_custom is True


@pytest.mark.asyncio
async def test_has_custom_settings_with_invalid_version(settings_store):
    # Arrange: User with invalid version
    with (
        patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'),
        patch('server.constants.CURRENT_USER_SETTINGS_VERSION', 5),
        patch(
            'server.constants.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022'},
        ),
    ):
        settings = Settings(
            llm_model='claude-3-5-sonnet-20241022', llm_base_url='http://default.url'
        )

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, 99)

        # Assert: Invalid version, so model is custom
        assert has_custom is True


@pytest.mark.asyncio
async def test_has_custom_settings_normalizes_whitespace(settings_store):
    # Arrange: Settings with whitespace in values
    with patch('storage.saas_settings_store.LITE_LLM_API_URL', 'http://default.url'):
        settings = Settings(
            llm_model='  claude-3-5-sonnet-20241022  ',
            llm_base_url='  http://default.url  ',
        )

        # Act: Check if has custom settings
        has_custom = settings_store._has_custom_settings(settings, None)

        # Assert: Whitespace is normalized, custom model detected
        assert has_custom is True


@pytest.mark.asyncio
async def test_update_settings_upgrades_user_from_old_defaults(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User with old version using old defaults
    old_version = 1
    old_model = 'litellm_proxy/prod/claude-3-5-sonnet-20241022'
    settings = Settings(llm_model=old_model, llm_base_url=LITE_LLM_API_URL)

    # Use a consistent test URL
    test_base_url = 'http://test.url'

    with (
        patch('storage.saas_settings_store.session_maker', session_maker),
        patch(
            'server.constants.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022', 5: 'claude-opus-4-5-20251101'},
        ),
        patch(
            'storage.saas_settings_store.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022', 5: 'claude-opus-4-5-20251101'},
        ),
        patch('server.constants.CURRENT_USER_SETTINGS_VERSION', 5),
        patch('storage.saas_settings_store.CURRENT_USER_SETTINGS_VERSION', 5),
        patch('storage.saas_settings_store.LITE_LLM_API_URL', test_base_url),
        patch(
            'storage.saas_settings_store.get_default_litellm_model',
            return_value='litellm_proxy/prod/claude-opus-4-5-20251101',
        ),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
    ):
        # Create existing user settings with old version
        with session_maker() as session:
            existing_settings = UserSettings(
                keycloak_user_id=settings_store.user_id,
                user_version=old_version,
                llm_model=old_model,
                llm_base_url=test_base_url,
            )
            session.add(existing_settings)
            session.commit()

        # Update settings to use test_base_url
        # Set user_version to match the database so _has_custom_settings can detect old defaults
        settings = Settings(
            llm_model=old_model, llm_base_url=test_base_url, user_version=old_version
        )

        # Act: Update settings
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

        # Assert: Settings upgraded to new defaults
        assert updated_settings is not None
        assert (
            updated_settings.llm_model == 'litellm_proxy/prod/claude-opus-4-5-20251101'
        )
        assert updated_settings.llm_base_url == test_base_url


@pytest.mark.asyncio
async def test_update_settings_preserves_custom_settings_during_upgrade(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User with old version but custom settings
    old_version = 1
    custom_model = 'gpt-4'
    custom_base_url = 'http://custom.url'
    settings = Settings(llm_model=custom_model, llm_base_url=custom_base_url)

    with (
        patch('storage.saas_settings_store.session_maker', session_maker),
        patch(
            'server.constants.USER_SETTINGS_VERSION_TO_MODEL',
            {1: 'claude-3-5-sonnet-20241022'},
        ),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
    ):
        # Create existing user settings with old version
        with session_maker() as session:
            existing_settings = UserSettings(
                keycloak_user_id=settings_store.user_id,
                user_version=old_version,
                llm_model=custom_model,
                llm_base_url=custom_base_url,
            )
            session.add(existing_settings)
            session.commit()

        # Act: Update settings
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

        # Assert: Custom settings preserved
        assert updated_settings is not None
        assert updated_settings.llm_model == custom_model
        assert updated_settings.llm_base_url == custom_base_url


@pytest.mark.asyncio
async def test_update_settings_migrates_billing_margin_v3_to_v4(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User with version 3 and billing margin
    old_version = 3
    billing_margin = 2.0
    max_budget = 10.0
    spend = 5.0

    settings = Settings()

    mock_get_response = AsyncMock()
    mock_get_response.is_success = True
    mock_get_response.json = MagicMock(
        return_value={'user_info': {'max_budget': max_budget, 'spend': spend}}
    )

    mock_post_response = AsyncMock()
    mock_post_response.is_success = True
    mock_post_response.json = MagicMock(return_value={'key': 'test_api_key'})

    with (
        patch('storage.saas_settings_store.session_maker', session_maker),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('httpx.AsyncClient') as mock_client,
    ):
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_get_response
        )
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_post_response
        )

        # Create existing user settings with version 3 and billing margin
        with session_maker() as session:
            existing_settings = UserSettings(
                keycloak_user_id=settings_store.user_id,
                user_version=old_version,
                billing_margin=billing_margin,
            )
            session.add(existing_settings)
            session.commit()

        # Act: Update settings
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

        # Assert: Settings updated
        assert updated_settings is not None

        # Assert: Billing margin applied to budget
        call_args = mock_client.return_value.__aenter__.return_value.post.call_args[1]
        assert call_args['json']['max_budget'] == max_budget * billing_margin
        assert call_args['json']['spend'] == spend * billing_margin

        # Assert: Billing margin reset to 1.0
        with session_maker() as session:
            updated_user_settings = (
                session.query(UserSettings)
                .filter(UserSettings.keycloak_user_id == settings_store.user_id)
                .first()
            )
            assert updated_user_settings.billing_margin == 1.0


@pytest.mark.asyncio
async def test_update_settings_skips_billing_margin_migration_when_already_v4(
    settings_store, mock_litellm_api, session_maker
):
    # Arrange: User with version 4
    version = 4
    billing_margin = 2.0
    max_budget = 10.0
    spend = 5.0

    settings = Settings()

    mock_get_response = AsyncMock()
    mock_get_response.is_success = True
    mock_get_response.json = MagicMock(
        return_value={'user_info': {'max_budget': max_budget, 'spend': spend}}
    )

    mock_post_response = AsyncMock()
    mock_post_response.is_success = True
    mock_post_response.json = MagicMock(return_value={'key': 'test_api_key'})

    with (
        patch('storage.saas_settings_store.session_maker', session_maker),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'user@example.com'}),
        ),
        patch('httpx.AsyncClient') as mock_client,
    ):
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_get_response
        )
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_post_response
        )

        # Create existing user settings with version 4
        with session_maker() as session:
            existing_settings = UserSettings(
                keycloak_user_id=settings_store.user_id,
                user_version=version,
                billing_margin=billing_margin,
            )
            session.add(existing_settings)
            session.commit()

        # Act: Update settings
        updated_settings = await settings_store.update_settings_with_litellm_default(
            settings
        )

        # Assert: Settings updated
        assert updated_settings is not None

        # Assert: Billing margin NOT applied (version >= 4)
        call_args = mock_client.return_value.__aenter__.return_value.post.call_args[1]
        assert call_args['json']['max_budget'] == max_budget
        assert call_args['json']['spend'] == spend
