import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.errors import AuthError
from openhands.app_server.secrets.secrets_router import (
    check_provider_tokens,
)
from openhands.app_server.settings.settings_router import store_llm_settings
from openhands.core.config.mcp_config import MCPConfig, MCPStdioServerConfig
from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType
from openhands.server.routes.secrets import (
    app as secrets_router,
)
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
        patch('openhands.app_server.utils.dependencies._SESSION_API_KEY', None),
        patch(
            'openhands.app_server.secrets.secrets_router.check_provider_tokens',
            AsyncMock(return_value=None),
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
        'openhands.app_server.secrets.secrets_router.validate_provider_token'
    ) as mock_validate:
        mock_validate.return_value = ProviderType.GITHUB
        await check_provider_tokens(providers, existing_provider_tokens)
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
        'openhands.app_server.secrets.secrets_router.validate_provider_token'
    ) as mock_validate:
        mock_validate.return_value = None

        # Should raise error for invalid token
        with pytest.raises(AuthError):
            await check_provider_tokens(providers, existing_provider_tokens)

        mock_validate.assert_called_once()


@pytest.mark.asyncio
async def test_check_provider_tokens_wrong_type():
    """Test check_provider_tokens with unsupported provider type."""
    # We can't test with an unsupported provider type directly since the model enforces valid types
    # Instead, we'll test with an empty provider_tokens dictionary
    providers = POSTProviderModel(provider_tokens={})

    # Empty existing provider tokens
    existing_provider_tokens = {}

    await check_provider_tokens(providers, existing_provider_tokens)


@pytest.mark.asyncio
async def test_check_provider_tokens_no_tokens():
    """Test check_provider_tokens with no tokens."""
    providers = POSTProviderModel(provider_tokens={})

    # Empty existing provider tokens
    existing_provider_tokens = {}

    await check_provider_tokens(providers, existing_provider_tokens)


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

    When llm_base_url="" (explicitly cleared), it must be stored as None
    regardless of the model type. Auto-detection is disabled for all models
    except openhands/ models (which use the LiteLLM proxy).
    """
    settings = Settings(
        llm_model='gpt-4',  # Only updating model (not an openhands model)
        llm_base_url='',  # Explicitly cleared (e.g. basic mode save)
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
    # llm_base_url="" is an explicit clear — must not be repopulated via auto-detection
    assert result.llm_base_url is None


@pytest.mark.asyncio
async def test_store_llm_settings_advanced_view_clear_removes_base_url():
    """Regression test for #13420: clearing Base URL in Advanced view must persist.

    Before the fix, llm_base_url="" was treated identically to llm_base_url=None,
    causing the backend to re-run auto-detection and overwrite the user's intent.
    """
    settings = Settings(
        llm_model='gpt-4',
        llm_base_url='',  # User deleted the field in Advanced view
    )

    existing_settings = Settings(
        llm_model='gpt-4',
        llm_api_key=SecretStr('my-api-key'),
        llm_base_url='https://my-custom-proxy.example.com',
    )

    result = await store_llm_settings(settings, existing_settings)

    # The old custom URL must not come back
    assert result.llm_base_url is None


@pytest.mark.asyncio
async def test_store_llm_settings_mcp_update_preserves_base_url():
    """Test that saving MCP config (without LLM fields) preserves existing base URL.

    Regression test: When adding an MCP server, the frontend sends only mcp_config
    and v1_enabled. This should not wipe out the existing llm_base_url.
    """
    # Simulate what the MCP add/update/delete mutations send: mcp_config but no LLM fields
    settings = Settings(
        mcp_config=MCPConfig(
            stdio_servers=[
                MCPStdioServerConfig(
                    name='my-server',
                    command='npx',
                    args=['-y', '@my/mcp-server'],
                    env={'API_TOKEN': 'secret123', 'ENDPOINT': 'https://example.com'},
                )
            ],
        ),
    )

    # Create existing settings with a custom base URL
    existing_settings = Settings(
        llm_model='anthropic/claude-sonnet-4-5-20250929',
        llm_api_key=SecretStr('existing-api-key'),
        llm_base_url='https://my-custom-proxy.example.com',
    )

    result = await store_llm_settings(settings, existing_settings)

    # All existing LLM settings should be preserved
    assert result.llm_model == 'anthropic/claude-sonnet-4-5-20250929'
    assert result.llm_api_key.get_secret_value() == 'existing-api-key'
    assert result.llm_base_url == 'https://my-custom-proxy.example.com'


@pytest.mark.asyncio
async def test_store_llm_settings_no_existing_base_url_stays_none():
    """Test that base URL stays None when no existing base URL is present.

    Auto-detection is disabled for non-openhands models to avoid setting wrong
    URLs (e.g. litellm returns localhost:11434 for ollama, which is wrong in Docker).
    Users must provide base_url explicitly for non-openhands models.
    """
    settings = Settings(
        llm_model='gpt-4'  # Not an openhands model
    )

    # Existing settings without a base URL
    existing_settings = Settings(
        llm_model='gpt-3.5',
        llm_api_key=SecretStr('existing-api-key'),
    )

    result = await store_llm_settings(settings, existing_settings)

    assert result.llm_model == 'gpt-4'
    assert result.llm_api_key.get_secret_value() == 'existing-api-key'
    # No auto-detection: base_url stays None, user must provide it explicitly
    assert result.llm_base_url is None


@pytest.mark.asyncio
async def test_store_llm_settings_anthropic_model_no_auto_detect():
    """Test store_llm_settings with an Anthropic model.

    Auto-detection is disabled for non-openhands models. Users must provide
    base_url explicitly. Litellm handles the default endpoints without needing
    an explicit base_url stored in settings.
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
    # No auto-detection: base_url stays None, litellm uses its own defaults
    assert result.llm_base_url is None


@pytest.mark.asyncio
async def test_store_llm_settings_unknown_model_base_url_stays_none():
    """Test that unknown models don't raise and base_url stays None.

    With auto-detection disabled, unknown models simply leave base_url as None.
    No litellm lookups are attempted, so no errors are logged.
    """
    settings = Settings(
        llm_model='unknown-model-xyz'  # A model that litellm won't recognize
    )

    existing_settings = Settings(
        llm_model='gpt-3.5',
        llm_api_key=SecretStr('existing-api-key'),
    )

    result = await store_llm_settings(settings, existing_settings)

    # llm_base_url should remain None — no auto-detection, no errors
    assert result.llm_base_url is None
    assert result.llm_model == 'unknown-model-xyz'


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
