import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType
from openhands.server.routes.secrets import (
    app as secrets_router,
)
from openhands.server.routes.secrets import (
    check_provider_tokens,
)
from openhands.server.routes.settings import store_llm_settings
from openhands.server.settings import POSTProviderModel
from openhands.storage import get_file_store
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.data_models.settings import Settings
from openhands.storage.secrets.file_secrets_store import FileSecretsStore


# Mock functions to simulate the actual functions in settings.py
async def get_settings_store(request):
    """Mock function to get settings store."""
    return MagicMock()


@pytest.fixture
def test_client():
    # Create a test client with a FastAPI app that includes the secrets router
    # This is necessary because TestClient with APIRouter directly doesn't set up
    # the full middleware stack in newer FastAPI versions (0.118.0+)
    test_app = FastAPI()
    test_app.include_router(secrets_router)

    with (
        patch.dict(os.environ, {'SESSION_API_KEY': ''}, clear=False),
        patch('openhands.server.dependencies._SESSION_API_KEY', None),
        patch(
            'openhands.server.routes.secrets.check_provider_tokens',
            AsyncMock(return_value=''),
        ),
    ):
        client = TestClient(test_app)
        yield client


@pytest.fixture
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('secrets_store'))


@pytest.fixture
def file_secrets_store(temp_dir):
    file_store = get_file_store('local', temp_dir)
    store = FileSecretsStore(file_store)
    with patch(
        'openhands.storage.secrets.file_secrets_store.FileSecretsStore.get_instance',
        AsyncMock(return_value=store),
    ):
        yield store


# Tests for check_provider_tokens
@pytest.mark.asyncio
async def test_check_provider_tokens_valid():
    """Test check_provider_tokens with valid tokens."""
    provider_token = ProviderToken(token=SecretStr('valid-token'))
    providers = POSTProviderModel(provider_tokens={ProviderType.GITHUB: provider_token})

    # Empty existing provider tokens
    existing_provider_tokens = {}

    # Mock the validate_provider_token function to return GITHUB for valid tokens
    with patch(
        'openhands.server.routes.secrets.validate_provider_token'
    ) as mock_validate:
        mock_validate.return_value = ProviderType.GITHUB

        result = await check_provider_tokens(providers, existing_provider_tokens)

        # Should return empty string for valid token
        assert result == ''
        mock_validate.assert_called_once()


@pytest.mark.asyncio
async def test_check_provider_tokens_invalid():
    """Test check_provider_tokens with invalid tokens."""
    provider_token = ProviderToken(token=SecretStr('invalid-token'))
    providers = POSTProviderModel(provider_tokens={ProviderType.GITHUB: provider_token})

    # Empty existing provider tokens
    existing_provider_tokens = {}

    # Mock the validate_provider_token function to return None for invalid tokens
    with patch(
        'openhands.server.routes.secrets.validate_provider_token'
    ) as mock_validate:
        mock_validate.return_value = None

        result = await check_provider_tokens(providers, existing_provider_tokens)

        # Should return error message for invalid token
        assert 'Invalid token' in result
        mock_validate.assert_called_once()


@pytest.mark.asyncio
async def test_check_provider_tokens_wrong_type():
    """Test check_provider_tokens with unsupported provider type."""
    # We can't test with an unsupported provider type directly since the model enforces valid types
    # Instead, we'll test with an empty provider_tokens dictionary
    providers = POSTProviderModel(provider_tokens={})

    # Empty existing provider tokens
    existing_provider_tokens = {}

    result = await check_provider_tokens(providers, existing_provider_tokens)

    # Should return empty string for no providers
    assert result == ''


@pytest.mark.asyncio
async def test_check_provider_tokens_no_tokens():
    """Test check_provider_tokens with no tokens."""
    providers = POSTProviderModel(provider_tokens={})

    # Empty existing provider tokens
    existing_provider_tokens = {}

    result = await check_provider_tokens(providers, existing_provider_tokens)

    # Should return empty string when no tokens provided
    assert result == ''


# Tests for store_llm_settings
@pytest.mark.asyncio
async def test_store_llm_settings_new_settings():
    """Test store_llm_settings with new settings."""
    settings = Settings(
        llm_model='gpt-4',
        llm_api_key='test-api-key',
        llm_base_url='https://api.example.com',
    )

    # No existing settings
    existing_settings = None

    result = await store_llm_settings(settings, existing_settings)

    # Should return settings with the provided values
    assert result.llm_model == 'gpt-4'
    assert result.llm_api_key.get_secret_value() == 'test-api-key'
    assert result.llm_base_url == 'https://api.example.com'


@pytest.mark.asyncio
async def test_store_llm_settings_update_existing():
    """Test store_llm_settings updates existing settings."""
    settings = Settings(
        llm_model='gpt-4',
        llm_api_key='new-api-key',
        llm_base_url='https://new.example.com',
    )

    # Create existing settings
    existing_settings = Settings(
        llm_model='gpt-3.5',
        llm_api_key=SecretStr('old-api-key'),
        llm_base_url='https://old.example.com',
    )

    result = await store_llm_settings(settings, existing_settings)

    # Should return settings with the updated values
    assert result.llm_model == 'gpt-4'
    assert result.llm_api_key.get_secret_value() == 'new-api-key'
    assert result.llm_base_url == 'https://new.example.com'


@pytest.mark.asyncio
async def test_store_llm_settings_partial_update():
    """Test store_llm_settings with partial update.

    Note: When llm_base_url is not provided in the update and the model is NOT an
    openhands model, we attempt to get the URL from litellm.get_api_base().
    For OpenAI models, this returns https://api.openai.com.
    """
    settings = Settings(
        llm_model='gpt-4'  # Only updating model (not an openhands model)
    )

    # Create existing settings
    existing_settings = Settings(
        llm_model='gpt-3.5',
        llm_api_key=SecretStr('existing-api-key'),
        llm_base_url='https://existing.example.com',
    )

    result = await store_llm_settings(settings, existing_settings)

    # Should return settings with updated model but keep API key
    assert result.llm_model == 'gpt-4'
    # For SecretStr objects, we need to compare the secret value
    assert result.llm_api_key.get_secret_value() == 'existing-api-key'
    # OpenAI models: litellm.get_api_base() returns https://api.openai.com
    assert result.llm_base_url == 'https://api.openai.com'


@pytest.mark.asyncio
async def test_store_llm_settings_anthropic_model_gets_api_base():
    """Test store_llm_settings with an Anthropic model.

    For Anthropic models, get_provider_api_base() returns the Anthropic API base URL
    via ProviderConfigManager.get_provider_model_info().
    """
    settings = Settings(
        llm_model='anthropic/claude-sonnet-4-5-20250929'  # Anthropic model
    )

    existing_settings = Settings(
        llm_model='gpt-3.5',
        llm_api_key=SecretStr('existing-api-key'),
    )

    result = await store_llm_settings(settings, existing_settings)

    assert result.llm_model == 'anthropic/claude-sonnet-4-5-20250929'
    assert result.llm_api_key.get_secret_value() == 'existing-api-key'
    # Anthropic models get https://api.anthropic.com via ProviderConfigManager
    assert result.llm_base_url == 'https://api.anthropic.com'


@pytest.mark.asyncio
async def test_store_llm_settings_litellm_error_logged():
    """Test that litellm errors are logged when getting api_base fails."""
    from unittest.mock import patch

    settings = Settings(
        llm_model='unknown-model-xyz'  # A model that litellm won't recognize
    )

    existing_settings = Settings(
        llm_model='gpt-3.5',
        llm_api_key=SecretStr('existing-api-key'),
    )

    # The function should not raise even if litellm fails
    with patch('openhands.server.routes.settings.logger') as mock_logger:
        result = await store_llm_settings(settings, existing_settings)

        # llm_base_url should remain None since litellm couldn't find the model
        assert result.llm_base_url is None
        # Either error or debug should have been logged
        assert mock_logger.error.called or mock_logger.debug.called


@pytest.mark.asyncio
async def test_store_llm_settings_openhands_model_gets_default_url():
    """Test store_llm_settings with openhands model gets LiteLLM proxy URL.

    When llm_base_url is not provided and the model is an openhands model,
    it gets set to the default LiteLLM proxy URL.
    """
    import os

    settings = Settings(
        llm_model='openhands/claude-sonnet-4-5-20250929'  # openhands model
    )

    # Create existing settings
    existing_settings = Settings(
        llm_model='gpt-3.5',
        llm_api_key=SecretStr('existing-api-key'),
    )

    result = await store_llm_settings(settings, existing_settings)

    # Should return settings with updated model
    assert result.llm_model == 'openhands/claude-sonnet-4-5-20250929'
    # For SecretStr objects, we need to compare the secret value
    assert result.llm_api_key.get_secret_value() == 'existing-api-key'
    # openhands models get the LiteLLM proxy URL
    expected_base_url = os.environ.get(
        'LITE_LLM_API_URL', 'https://llm-proxy.app.all-hands.dev'
    )
    assert result.llm_base_url == expected_base_url


# Tests for store_provider_tokens
@pytest.mark.asyncio
async def test_store_provider_tokens_new_tokens(test_client, file_secrets_store):
    """Test store_provider_tokens with new tokens."""
    provider_tokens = {'provider_tokens': {'github': {'token': 'new-token'}}}

    # Mock the settings store
    mock_store = MagicMock()
    mock_store.load = AsyncMock(return_value=None)  # No existing settings

    Secrets()

    user_secrets = await file_secrets_store.store(Secrets())

    response = test_client.post('/api/add-git-providers', json=provider_tokens)
    assert response.status_code == 200

    user_secrets = await file_secrets_store.load()

    assert (
        user_secrets.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'new-token'
    )


@pytest.mark.asyncio
async def test_store_provider_tokens_update_existing(test_client, file_secrets_store):
    """Test store_provider_tokens updates existing tokens."""
    # Create existing settings with a GitHub token
    github_token = ProviderToken(token=SecretStr('old-token'))
    provider_tokens = {ProviderType.GITHUB: github_token}

    # Create a Secrets with the provider tokens
    user_secrets = Secrets(provider_tokens=provider_tokens)

    await file_secrets_store.store(user_secrets)

    response = test_client.post(
        '/api/add-git-providers',
        json={'provider_tokens': {'github': {'token': 'updated-token'}}},
    )

    assert response.status_code == 200

    user_secrets = await file_secrets_store.load()

    assert (
        user_secrets.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'updated-token'
    )


@pytest.mark.asyncio
async def test_store_provider_tokens_keep_existing(test_client, file_secrets_store):
    """Test store_provider_tokens keeps existing tokens when empty string provided."""
    # Create existing secrets with a GitHub token
    github_token = ProviderToken(token=SecretStr('existing-token'))
    provider_tokens = {ProviderType.GITHUB: github_token}
    user_secrets = Secrets(provider_tokens=provider_tokens)

    await file_secrets_store.store(user_secrets)

    response = test_client.post(
        '/api/add-git-providers',
        json={'provider_tokens': {'github': {'token': ''}}},
    )
    assert response.status_code == 200

    user_secrets = await file_secrets_store.load()

    assert (
        user_secrets.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'existing-token'
    )
