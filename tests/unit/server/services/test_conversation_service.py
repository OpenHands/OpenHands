"""Unit tests for conversation_service.py - specifically testing provider token handling.

These tests verify that setup_init_conversation_settings correctly handles provider tokens
in different scenarios (provided tokens vs scaffold creation).
"""

from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.server.services.conversation_service import (
    setup_init_conversation_settings,
)
from openhands.server.types import AppMode
from openhands.storage.data_models.settings import Settings


@pytest.fixture
def mock_settings():
    """Create a real Settings object with minimal required fields."""
    return Settings(
        language='en',
        agent='CodeActAgent',
        max_iterations=100,
        llm_model='anthropic/claude-3-5-sonnet-20241022',
        llm_api_key=SecretStr('test_api_key_12345'),
        llm_base_url=None,
    )


@pytest.fixture
def mock_provider_tokens():
    """Create real provider tokens to test with."""
    return MappingProxyType(
        {
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('ghp_real_token_test123'), user_id='test_user_456'
            )
        }
    )


@pytest.mark.asyncio
async def test_setup_with_provided_tokens_uses_real_tokens(
    mock_settings, mock_provider_tokens
):
    """Test that real tokens are used when provided in SAAS mode.

    Verifies provider tokens passed in are used in ConversationInitData.
    """
    user_id = 'test_user_456'
    conversation_id = 'test_conv_123'
    providers_set = [ProviderType.GITHUB]

    # Mock the stores to return our test settings
    with patch(
        'openhands.server.services.conversation_service.SettingsStoreImpl.get_instance'
    ) as mock_settings_store_cls:
        with patch(
            'openhands.server.services.conversation_service.SecretsStoreImpl.get_instance'
        ) as mock_secrets_store_cls:
            with patch(
                'openhands.server.services.conversation_service.server_config'
            ) as mock_server_config:
                # Setup mocks
                mock_settings_store = AsyncMock()
                mock_settings_store.load = AsyncMock(return_value=mock_settings)
                mock_settings_store_cls.return_value = mock_settings_store

                mock_secrets_store = AsyncMock()
                mock_secrets_store.load = AsyncMock(return_value=None)
                mock_secrets_store_cls.return_value = mock_secrets_store

                mock_server_config.app_mode = AppMode.SAAS

                # Call with real tokens
                result = await setup_init_conversation_settings(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    providers_set=providers_set,
                    provider_tokens=mock_provider_tokens,
                )

                # Verify real tokens are used
                assert result.git_provider_tokens is not None
                assert result.git_provider_tokens == mock_provider_tokens
                assert ProviderType.GITHUB in result.git_provider_tokens, (
                    'GitHub provider should be in tokens'
                )

                github_token = result.git_provider_tokens[ProviderType.GITHUB]
                assert (
                    github_token.token.get_secret_value() == 'ghp_real_token_test123'
                ), 'Should use real token, not None'
                assert github_token.user_id == 'test_user_456', (
                    'Should preserve user_id from real token'
                )


@pytest.mark.asyncio
async def test_setup_without_tokens_non_saas_uses_user_secrets(mock_settings):
    """Test that OpenHands (OSS mode) uses user_secrets.provider_tokens when no tokens provided.

    This test verifies OpenHands (OSS mode) backward compatibility - tokens come from local config, not endpoint.
    """
    user_id = 'test_user_456'
    conversation_id = 'test_conv_123'
    providers_set = [ProviderType.GITHUB]

    # Create user_secrets with real tokens
    mock_user_secrets = MagicMock()
    mock_user_secrets.provider_tokens = MappingProxyType(
        {
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('ghp_local_token_from_config'),
                user_id='local_user_123',
            )
        }
    )
    mock_user_secrets.custom_secrets = MappingProxyType({})  # Empty dict is fine

    with patch(
        'openhands.server.services.conversation_service.SettingsStoreImpl.get_instance'
    ) as mock_settings_store_cls:
        with patch(
            'openhands.server.services.conversation_service.SecretsStoreImpl.get_instance'
        ) as mock_secrets_store_cls:
            with patch(
                'openhands.server.services.conversation_service.server_config'
            ) as mock_server_config:
                # Setup mocks
                mock_settings_store = AsyncMock()
                mock_settings_store.load = AsyncMock(return_value=mock_settings)
                mock_settings_store_cls.return_value = mock_settings_store

                mock_secrets_store = AsyncMock()
                mock_secrets_store.load = AsyncMock(return_value=mock_user_secrets)
                mock_secrets_store_cls.return_value = mock_secrets_store

                mock_server_config.app_mode = AppMode.OPENHANDS

                # Call without endpoint tokens
                result = await setup_init_conversation_settings(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    providers_set=providers_set,
                    provider_tokens=None,
                )

                # Verify user_secrets tokens are used
                assert result.git_provider_tokens is not None
                assert ProviderType.GITHUB in result.git_provider_tokens

                github_token = result.git_provider_tokens[ProviderType.GITHUB]
                assert (
                    github_token.token.get_secret_value()
                    == 'ghp_local_token_from_config'
                )
                assert github_token.user_id == 'local_user_123'
