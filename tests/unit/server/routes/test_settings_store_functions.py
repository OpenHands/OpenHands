import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.errors import AuthError
from openhands.app_server.secrets.secrets_router import check_provider_tokens
from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType
from openhands.sdk.llm import LLM
from openhands.sdk.settings import (
    AGENT_SETTINGS_SCHEMA_VERSION,
    AgentSettings,
    ConversationSettings,
)
from openhands.server.routes.secrets import (
    app as secrets_router,
)
from openhands.server.settings import POSTProviderModel
from openhands.storage import get_file_store
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.data_models.settings import Settings
from openhands.storage.secrets.file_secrets_store import FileSecretsStore

_EXPOSE = {'expose_secrets': True}


def _apply_settings_payload(
    payload: dict, existing_settings: Settings | None
) -> Settings:
    """Test helper — mirrors the inlined logic in the settings route."""
    settings = existing_settings.model_copy() if existing_settings else Settings()
    settings.update(payload)
    return settings


_DEFAULT_LLM = LLM(model='test-model')


def _make_settings(llm: LLM | None = None) -> Settings:
    """Helper to create Settings with an AgentSettings object."""
    return Settings(agent_settings=AgentSettings(llm=llm or _DEFAULT_LLM))


def _secret(s: SecretStr | None) -> str | None:
    """Unwrap a SecretStr to its plain value."""
    return s.get_secret_value() if s is not None else None


def _persisted(settings: Settings) -> dict:
    """Dump agent_settings with secrets exposed for persistence assertions."""
    return settings.agent_settings.model_dump(mode='json', context=_EXPOSE)


@pytest.fixture(autouse=True)
def allow_short_context_windows():
    with patch.dict(os.environ, {'ALLOW_SHORT_CONTEXT_WINDOWS': 'true'}, clear=False):
        yield


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

        with pytest.raises(AuthError):
            await check_provider_tokens(providers, existing_provider_tokens)

        mock_validate.assert_called_once()


@pytest.mark.asyncio
async def test_check_provider_tokens_wrong_type():
    """Test check_provider_tokens with unsupported provider type."""
    providers = POSTProviderModel(provider_tokens={})
    existing_provider_tokens = {}

    await check_provider_tokens(providers, existing_provider_tokens)


@pytest.mark.asyncio
async def test_check_provider_tokens_no_tokens():
    """Test check_provider_tokens with no tokens."""
    providers = POSTProviderModel(provider_tokens={})
    existing_provider_tokens = {}

    await check_provider_tokens(providers, existing_provider_tokens)


# Tests for _apply_settings_payload (SDK-first settings)
def test_apply_payload_sdk_keys_stored_and_readable():
    """Nested SDK keys should be stored in agent_settings and readable via properties."""
    payload = {
        'agent_settings': {
            'llm': {
                'model': 'gpt-4',
                'api_key': 'test-api-key',
                'base_url': 'https://api.example.com',
            }
        },
    }

    result = _apply_settings_payload(payload, None)

    assert _persisted(result)['llm']['model'] == 'gpt-4'
    assert _persisted(result)['llm']['api_key'] == 'test-api-key'
    assert _persisted(result)['llm']['base_url'] == 'https://api.example.com'
    # Properties read from agent_settings
    assert result.agent_settings.llm.model == 'gpt-4'
    assert _secret(result.agent_settings.llm.api_key) == 'test-api-key'
    assert result.agent_settings.llm.base_url == 'https://api.example.com'


def test_apply_payload_updates_existing():
    """Nested SDK keys should update existing settings."""
    existing = _make_settings(
        LLM(
            model='gpt-3.5',
            api_key=SecretStr('old-api-key'),
            base_url='https://old.example.com',
        )
    )

    payload = {
        'agent_settings': {
            'llm': {
                'model': 'gpt-4',
                'api_key': 'new-api-key',
                'base_url': 'https://new.example.com',
            }
        },
    }

    result = _apply_settings_payload(payload, existing)

    assert result.agent_settings.llm.model == 'gpt-4'
    assert _secret(result.agent_settings.llm.api_key) == 'new-api-key'
    assert result.agent_settings.llm.base_url == 'https://new.example.com'


def test_apply_payload_preserves_secrets_when_not_provided():
    """When the API key is not in the payload, the existing value is preserved."""
    existing = _make_settings(
        LLM(model='gpt-3.5', api_key=SecretStr('existing-api-key'))
    )

    payload = {'agent_settings': {'llm': {'model': 'gpt-4'}}}

    result = _apply_settings_payload(payload, existing)

    assert result.agent_settings.llm.model == 'gpt-4'
    assert _secret(result.agent_settings.llm.api_key) == 'existing-api-key'
    assert result.agent_settings.llm.base_url is None


def test_apply_payload_mcp_update_preserves_existing_llm_settings():
    existing_settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='anthropic/claude-sonnet-4-5-20250929',
                api_key=SecretStr('existing-api-key'),
                base_url='https://my-custom-proxy.example.com',
            )
        ),
    )

    result = _apply_settings_payload(
        {
            'agent_settings': {
                'mcp_config': {
                    'stdio_servers': [
                        {
                            'name': 'my-server',
                            'command': 'npx',
                            'args': ['-y', '@my/mcp-server'],
                            'env': {
                                'API_TOKEN': 'secret123',
                                'ENDPOINT': 'https://example.com',
                            },
                        }
                    ]
                }
            }
        },
        existing_settings,
    )

    assert result.agent_settings.llm.model == 'anthropic/claude-sonnet-4-5-20250929'
    assert _secret(result.agent_settings.llm.api_key) == 'existing-api-key'
    assert result.agent_settings.llm.base_url == 'https://my-custom-proxy.example.com'


def test_apply_payload_clears_secrets_when_explicitly_null_or_empty():
    """Explicit null/empty secret values should clear existing SDK secrets."""
    existing = _make_settings(
        LLM(
            model='anthropic/claude-sonnet-4-5-20250929',
            api_key=SecretStr('existing-api-key'),
        )
    )

    payload = {'agent_settings': {'llm': {'api_key': None}}}
    result = _apply_settings_payload(payload, existing)
    assert result.agent_settings.llm.api_key is None

    payload = {'agent_settings': {'llm': {'api_key': ''}}}
    result = _apply_settings_payload(payload, existing)
    assert result.agent_settings.llm.api_key is None


def test_apply_payload_preserves_explicit_null_non_secret_sdk_resets():
    """Explicit null non-secret SDK values should survive for inherited-settings clearing."""
    existing = _make_settings(
        LLM(model='openai/gpt-4o', base_url='https://custom.example/v1')
    )

    result = _apply_settings_payload(
        {'agent_settings': {'llm': {'base_url': None}}}, existing
    )

    assert result.agent_settings.llm.base_url is None
    assert _persisted(result)['llm']['base_url'] is None


def test_apply_payload_mcp_preserves_llm_settings():
    """Non-LLM payloads (e.g. MCP config) should not affect existing LLM settings."""
    existing = _make_settings(
        LLM(
            model='anthropic/claude-sonnet-4-5-20250929',
            api_key=SecretStr('existing-api-key'),
            base_url='https://my-custom-proxy.example.com',
        )
    )

    payload = {
        'agent_settings': {
            'mcp_config': {
                'stdio_servers': [
                    {
                        'name': 'my-server',
                        'command': 'npx',
                        'args': ['-y', '@my/mcp-server'],
                    }
                ],
            },
        },
    }

    result = _apply_settings_payload(payload, existing)

    assert result.agent_settings.llm.model == 'anthropic/claude-sonnet-4-5-20250929'
    assert _secret(result.agent_settings.llm.api_key) == 'existing-api-key'
    assert result.agent_settings.llm.base_url == 'https://my-custom-proxy.example.com'


def test_apply_payload_non_sdk_flat_keys_applied():
    """Non-SDK flat keys (language, git, etc.) should still be applied normally."""
    payload = {
        'language': 'ja',
        'git_user_name': 'test-user',
    }

    result = _apply_settings_payload(payload, None)

    assert result.language == 'ja'
    assert result.git_user_name == 'test-user'


def test_apply_payload_conversation_settings_stored_top_level():
    """Conversation security settings should be applied as top-level Settings fields."""
    payload = {
        'conversation_settings': {
            'confirmation_mode': True,
            'security_analyzer': 'llm',
        },
    }

    result = _apply_settings_payload(payload, None)

    assert result.conversation_settings.confirmation_mode is True
    assert result.conversation_settings.security_analyzer == 'llm'
    assert not hasattr(result.agent_settings, 'confirmation_mode')


def test_agent_settings_construction():
    """Settings constructed with proper objects should be accessible."""
    s = Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='gpt-4',
                api_key=SecretStr('my-key'),
                base_url='https://example.com',
            ),
            agent='CodeActAgent',
        ),
        conversation_settings=ConversationSettings(confirmation_mode=True),
    )

    assert _persisted(s)['llm']['model'] == 'gpt-4'
    assert _persisted(s)['llm']['api_key'] == 'my-key'
    assert _persisted(s)['llm']['base_url'] == 'https://example.com'
    assert _persisted(s)['agent'] == 'CodeActAgent'
    assert s.conversation_settings.confirmation_mode is True
    assert s.agent_settings.llm.model == 'gpt-4'
    assert s.agent_settings.agent == 'CodeActAgent'


def test_agent_settings_normalized_with_schema_version_and_extras():
    s = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='anthropic/claude-sonnet-4-5-20250929'),
        ),
        conversation_settings=ConversationSettings(
            max_iterations=64,
            confirmation_mode=True,
        ),
    )

    dump = s.agent_settings.model_dump(mode='json', context={'expose_secrets': True})
    assert dump['schema_version'] == AGENT_SETTINGS_SCHEMA_VERSION
    assert _persisted(s)['llm']['model'] == 'anthropic/claude-sonnet-4-5-20250929'
    assert s.conversation_settings.confirmation_mode is True
    assert s.conversation_settings.max_iterations == 64


def test_agent_settings_persistence_strips_secret_values():
    s = Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='anthropic/claude-sonnet-4-5-20250929',
                api_key=SecretStr('super-secret'),
            ),
        ),
        conversation_settings=ConversationSettings(max_iterations=64),
    )

    persisted = s.agent_settings.model_dump(mode='json')
    # Secrets are redacted by default (not exposed)
    assert persisted['schema_version'] == AGENT_SETTINGS_SCHEMA_VERSION
    assert persisted['llm']['model'] == 'anthropic/claude-sonnet-4-5-20250929'
    assert 'max_iterations' not in persisted
    assert persisted['llm']['api_key'] == '**********'
    assert s.conversation_settings.max_iterations == 64


def test_openhands_model_settings_remain_user_facing():
    s = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='openhands/claude-opus-4-5-20251101')
        )
    )

    assert _persisted(s)['llm']['model'] == 'litellm_proxy/claude-opus-4-5-20251101'
    api_data = s.get_agent_settings_display()
    assert api_data['llm']['model'] == 'openhands/claude-opus-4-5-20251101'


def test_litellm_proxy_model_settings_migrate_back_to_openhands_prefix():
    s = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='litellm_proxy/claude-opus-4-5-20251101')
        )
    )

    assert _persisted(s)['llm']['model'] == 'litellm_proxy/claude-opus-4-5-20251101'
    api_data = s.get_agent_settings_display()
    assert api_data['llm']['model'] == 'openhands/claude-opus-4-5-20251101'


# ──────────────────────────────────────────────────────────────────
# Regression tests for openhands/ → litellm_proxy/ round-trip bug
# ──────────────────────────────────────────────────────────────────
# The SDK's AgentSettings.model_validate automatically transforms
# "openhands/X" → "litellm_proxy/X" with a proxy base_url.  The
# settings router must convert it back for the frontend and handle
# the internal model name in is_openhands_model-style checks.


def test_post_merge_llm_fixups_handles_openhands_model_after_sdk_transform():
    """After SDK transforms openhands/X → litellm_proxy/X with a base_url,
    _post_merge_llm_fixups should recognise it as a managed model and not
    clobber the base_url with a provider lookup."""
    from openhands.app_server.settings.settings_router import _post_merge_llm_fixups

    # Simulate: user sends openhands/claude-opus-4-5, sdk transforms it
    settings = Settings()
    settings.update({'agent_settings': {'llm': {'model': 'openhands/claude-opus-4-5'}}})

    # After SDK transform:
    assert settings.agent_settings.llm.model == 'litellm_proxy/claude-opus-4-5'
    assert settings.agent_settings.llm.base_url is not None  # SDK set it

    # _post_merge_llm_fixups should NOT break the base_url
    _post_merge_llm_fixups(settings)
    assert settings.agent_settings.llm.base_url is not None


def test_post_merge_llm_fixups_sets_proxy_url_for_openhands_model_with_null_base_url():
    """When an openhands model ends up with a null base_url (edge case from
    older storage), _post_merge_llm_fixups should set the proxy URL."""
    from openhands.app_server.settings.settings_router import (
        LITE_LLM_API_URL,
        _post_merge_llm_fixups,
    )

    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='litellm_proxy/claude-opus-4-5', base_url=None)
        ),
    )
    # Force base_url to None (bypass SDK validator for this edge case)
    settings.agent_settings.llm.base_url = None

    _post_merge_llm_fixups(settings)

    # Should recognise litellm_proxy/ as a managed model and set the URL
    assert settings.agent_settings.llm.base_url == LITE_LLM_API_URL


def test_get_agent_settings_display_clears_proxy_base_url():
    """get_agent_settings_display should clear the LiteLLM proxy base_url
    for openhands models so the frontend sees null (enabling basic mode)."""
    s = Settings(
        agent_settings=AgentSettings(llm=LLM(model='openhands/claude-opus-4-5'))
    )

    # SDK sets the proxy URL internally
    assert s.agent_settings.llm.base_url is not None

    display = s.get_agent_settings_display()
    # Model name should be converted back
    assert display['llm']['model'] == 'openhands/claude-opus-4-5'
    # Proxy base_url should be cleared for display
    assert display['llm']['base_url'] is None


def test_save_then_display_roundtrip_openhands_model():
    """Full round-trip: save openhands model via update(), then display it.
    The frontend should see openhands/X with null base_url (basic mode)."""
    # Simulate save
    settings = Settings()
    settings.update(
        {
            'agent_settings': {
                'llm': {
                    'model': 'openhands/claude-opus-4-5',
                    'api_key': '',
                    'base_url': None,
                }
            }
        }
    )

    # Verify internal representation
    assert settings.agent_settings.llm.model == 'litellm_proxy/claude-opus-4-5'

    # Simulate load response
    display = settings.get_agent_settings_display()
    assert display['llm']['model'] == 'openhands/claude-opus-4-5'
    assert display['llm']['base_url'] is None


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
