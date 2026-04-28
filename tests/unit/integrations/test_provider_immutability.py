from types import MappingProxyType
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr, ValidationError

from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.settings.settings_models import Settings
from openhands.integrations.provider import (
    ProviderHandler,
    ProviderToken,
    ProviderType,
)


def test_provider_token_immutability():
    """Test that ProviderToken is immutable"""
    token = ProviderToken(token=SecretStr('test'), user_id='user1')

    # Test direct attribute modification
    with pytest.raises(ValidationError):
        token.token = SecretStr('new')

    with pytest.raises(ValidationError):
        token.user_id = 'new_user'

    # Test that __setattr__ is blocked
    with pytest.raises(ValidationError):
        setattr(token, 'token', SecretStr('new'))

    # Verify original values are unchanged
    assert token.token.get_secret_value() == 'test'
    assert token.user_id == 'user1'


def test_secret_store_immutability():
    """Test that Secrets is immutable"""
    store = Secrets(
        provider_tokens={ProviderType.GITHUB: ProviderToken(token=SecretStr('test'))}
    )

    # Test direct attribute modification
    with pytest.raises(ValidationError):
        store.provider_tokens = {}

    # Test dictionary mutation attempts
    with pytest.raises((TypeError, AttributeError)):
        store.provider_tokens[ProviderType.GITHUB] = ProviderToken(
            token=SecretStr('new')
        )

    with pytest.raises((TypeError, AttributeError)):
        store.provider_tokens.clear()

    with pytest.raises((TypeError, AttributeError)):
        store.provider_tokens.update(
            {ProviderType.GITLAB: ProviderToken(token=SecretStr('test'))}
        )

    # Test nested immutability
    github_token = store.provider_tokens[ProviderType.GITHUB]
    with pytest.raises(ValidationError):
        github_token.token = SecretStr('new')

    # Verify original values are unchanged
    assert store.provider_tokens[ProviderType.GITHUB].token.get_secret_value() == 'test'


def test_settings_immutability():
    """Test that Settings secrets_store is immutable"""
    settings = Settings(
        secrets_store=Secrets(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(token=SecretStr('test'))
            }
        )
    )

    # Test direct modification of secrets_store
    with pytest.raises(ValidationError):
        settings.secrets_store = Secrets()

    # Test nested modification attempts
    with pytest.raises((TypeError, AttributeError)):
        settings.secrets_store.provider_tokens[ProviderType.GITHUB] = ProviderToken(
            token=SecretStr('new')
        )

    # Test model_copy creates new instance
    new_store = Secrets(
        provider_tokens={
            ProviderType.GITHUB: ProviderToken(token=SecretStr('new_token'))
        }
    )
    new_settings = settings.model_copy(update={'secrets_store': new_store})

    # Verify original is unchanged and new has updated values
    assert (
        settings.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token.get_secret_value()
        == 'test'
    )
    assert (
        new_settings.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token.get_secret_value()
        == 'new_token'
    )

    with pytest.raises(ValidationError):
        new_settings.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token = SecretStr('')


def test_provider_handler_immutability():
    """Test that ProviderHandler maintains token immutability"""
    # Create initial tokens
    tokens = MappingProxyType(
        {ProviderType.GITHUB: ProviderToken(token=SecretStr('test'))}
    )

    handler = ProviderHandler(provider_tokens=tokens)

    # Try to modify tokens (should raise TypeError due to frozen dict)
    with pytest.raises((TypeError, AttributeError)):
        handler.provider_tokens[ProviderType.GITHUB] = ProviderToken(
            token=SecretStr('new')
        )

    # Try to modify the handler's tokens property
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        handler.provider_tokens = {}

    # Original token should be unchanged
    assert (
        handler.provider_tokens[ProviderType.GITHUB].token.get_secret_value() == 'test'
    )


def test_token_conversion():
    """Test token conversion in Secrets.create"""
    # Test with string token
    store1 = Settings(
        secrets_store=Secrets(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(token=SecretStr('test_token'))
            }
        )
    )

    assert (
        store1.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token.get_secret_value()
        == 'test_token'
    )
    assert store1.secrets_store.provider_tokens[ProviderType.GITHUB].user_id is None

    # Test with dict token
    store2 = Secrets(
        provider_tokens={'github': {'token': 'test_token', 'user_id': 'user1'}}
    )
    assert (
        store2.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'test_token'
    )
    assert store2.provider_tokens[ProviderType.GITHUB].user_id == 'user1'

    # Test with ProviderToken
    token = ProviderToken(token=SecretStr('test_token'), user_id='user2')
    store3 = Secrets(provider_tokens={ProviderType.GITHUB: token})
    assert (
        store3.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'test_token'
    )
    assert store3.provider_tokens[ProviderType.GITHUB].user_id == 'user2'

    store4 = Secrets(
        provider_tokens={
            ProviderType.GITHUB: 123  # Invalid type
        }
    )

    assert ProviderType.GITHUB not in store4.provider_tokens

    # Test with empty/None token
    store5 = Secrets(provider_tokens={ProviderType.GITHUB: None})
    assert ProviderType.GITHUB not in store5.provider_tokens

    store6 = Secrets(
        provider_tokens={
            'invalid_provider': 'test_token'  # Invalid provider type
        }
    )

    assert len(store6.provider_tokens.keys()) == 0


def test_provider_handler_type_enforcement():
    with pytest.raises((TypeError)):
        ProviderHandler(provider_tokens={'a': 'b'})


def test_get_provider_env_key():
    """Test provider environment key generation"""
    assert ProviderHandler.get_provider_env_key(ProviderType.GITHUB) == 'github_token'
    assert ProviderHandler.get_provider_env_key(ProviderType.GITLAB) == 'gitlab_token'


@pytest.mark.asyncio
async def test_get_github_organizations_delegates_to_service():
    """Test that get_github_organizations calls get_organizations_from_installations on the GitHub service."""
    tokens = MappingProxyType(
        {ProviderType.GITHUB: ProviderToken(token=SecretStr('gh-token'))}
    )
    handler = ProviderHandler(provider_tokens=tokens)

    with patch.object(handler, 'get_service') as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.get_organizations_from_installations = AsyncMock(
            return_value=['org1', 'org2']
        )

        result = await handler.get_github_organizations()

        assert result == ['org1', 'org2']
        mock_get_service.assert_called_once_with(ProviderType.GITHUB)


@pytest.mark.asyncio
async def test_get_github_organizations_returns_empty_on_error():
    """Test that get_github_organizations returns empty list when the service call fails."""
    tokens = MappingProxyType(
        {ProviderType.GITHUB: ProviderToken(token=SecretStr('gh-token'))}
    )
    handler = ProviderHandler(provider_tokens=tokens)

    with patch.object(handler, 'get_service') as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.get_organizations_from_installations = AsyncMock(
            side_effect=Exception('API error')
        )

        result = await handler.get_github_organizations()

        assert result == []


@pytest.mark.asyncio
async def test_get_gitlab_groups_delegates_to_service():
    """Test that get_gitlab_groups calls get_user_groups on the GitLab service."""
    tokens = MappingProxyType(
        {ProviderType.GITLAB: ProviderToken(token=SecretStr('gl-token'))}
    )
    handler = ProviderHandler(provider_tokens=tokens)

    with patch.object(handler, 'get_service') as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.get_user_groups = AsyncMock(return_value=['group-a', 'group-b'])

        result = await handler.get_gitlab_groups()

        assert result == ['group-a', 'group-b']
        mock_get_service.assert_called_once_with(ProviderType.GITLAB)
