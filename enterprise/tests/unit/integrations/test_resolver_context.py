"""Test for ResolverUserContext get_secrets and get_latest_token logic.

This test focuses on testing the actual ResolverUserContext implementation.
"""

from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from enterprise.integrations.resolver_context import ResolverUserContext

# Import the real classes we want to test
from openhands.integrations.provider import CustomSecret, ProviderToken
from openhands.integrations.service_types import ProviderType

# Import the SDK types we need for testing
from openhands.sdk.secret import SecretSource, StaticSecret
from openhands.storage.data_models.secrets import Secrets


@pytest.fixture
def mock_saas_user_auth():
    """Mock SaasUserAuth for testing."""
    return AsyncMock()


@pytest.fixture
def resolver_context(mock_saas_user_auth):
    """Create a ResolverUserContext instance for testing."""
    return ResolverUserContext(saas_user_auth=mock_saas_user_auth)


def create_custom_secret(value: str, description: str = 'Test secret') -> CustomSecret:
    """Helper to create CustomSecret instances."""
    return CustomSecret(secret=SecretStr(value), description=description)


def create_secrets(custom_secrets_dict: dict[str, CustomSecret]) -> Secrets:
    """Helper to create Secrets instances."""
    return Secrets(custom_secrets=MappingProxyType(custom_secrets_dict))


@pytest.mark.asyncio
async def test_get_secrets_converts_custom_to_static(
    resolver_context, mock_saas_user_auth
):
    """Test that get_secrets correctly converts CustomSecret objects to StaticSecret objects."""
    # Arrange
    secrets = create_secrets(
        {
            'TEST_SECRET_1': create_custom_secret('secret_value_1'),
            'TEST_SECRET_2': create_custom_secret('secret_value_2'),
        }
    )
    mock_saas_user_auth.get_secrets.return_value = secrets

    # Act
    result = await resolver_context.get_secrets()

    # Assert
    assert len(result) == 2
    assert all(isinstance(secret, StaticSecret) for secret in result.values())
    assert result['TEST_SECRET_1'].value.get_secret_value() == 'secret_value_1'
    assert result['TEST_SECRET_2'].value.get_secret_value() == 'secret_value_2'


@pytest.mark.asyncio
async def test_get_secrets_with_special_characters(
    resolver_context, mock_saas_user_auth
):
    """Test that secret values with special characters are preserved during conversion."""
    # Arrange
    special_value = 'very_secret_password_123!@#$%^&*()'
    secrets = create_secrets({'SPECIAL_SECRET': create_custom_secret(special_value)})
    mock_saas_user_auth.get_secrets.return_value = secrets

    # Act
    result = await resolver_context.get_secrets()

    # Assert
    assert len(result) == 1
    assert isinstance(result['SPECIAL_SECRET'], StaticSecret)
    assert result['SPECIAL_SECRET'].value.get_secret_value() == special_value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'secrets_input,expected_result',
    [
        (None, {}),  # No secrets available
        (create_secrets({}), {}),  # Empty custom secrets
    ],
)
async def test_get_secrets_empty_cases(
    resolver_context, mock_saas_user_auth, secrets_input, expected_result
):
    """Test that get_secrets handles empty cases correctly."""
    # Arrange
    mock_saas_user_auth.get_secrets.return_value = secrets_input

    # Act
    result = await resolver_context.get_secrets()

    # Assert
    assert result == expected_result


def test_static_secret_is_valid_secret_source():
    """Test that StaticSecret is a valid SecretSource for SDK validation."""
    # Arrange & Act
    static_secret = StaticSecret(value='test_secret_123')

    # Assert
    assert isinstance(static_secret, StaticSecret)
    assert isinstance(static_secret, SecretSource)
    assert static_secret.value.get_secret_value() == 'test_secret_123'


def test_custom_to_static_conversion():
    """Test the complete conversion flow from CustomSecret to StaticSecret."""
    # Arrange
    secret_value = 'conversion_test_secret'
    custom_secret = create_custom_secret(secret_value, 'Conversion test')

    # Act - simulate the conversion logic from the actual method
    extracted_value = custom_secret.secret.get_secret_value()
    static_secret = StaticSecret(value=extracted_value)

    # Assert
    assert isinstance(static_secret, StaticSecret)
    assert isinstance(static_secret, SecretSource)
    assert static_secret.value.get_secret_value() == secret_value


# ---------------------------------------------------------------------------
# Tests for get_latest_token - ensuring string values are returned
# ---------------------------------------------------------------------------


def create_provider_tokens(
    tokens_dict: dict[ProviderType, str],
) -> dict[ProviderType, ProviderToken]:
    """Helper to create provider tokens dictionary."""
    return {
        provider_type: ProviderToken(token=SecretStr(token_value))
        for provider_type, token_value in tokens_dict.items()
    }


@pytest.mark.asyncio
async def test_get_latest_token_returns_string(resolver_context, mock_saas_user_auth):
    """Test that get_latest_token returns a string, not a ProviderToken object."""
    # Arrange
    token_value = 'ghp_test_github_token_123'
    provider_tokens = create_provider_tokens({ProviderType.GITHUB: token_value})
    mock_saas_user_auth.get_provider_tokens = AsyncMock(return_value=provider_tokens)

    # Act
    result = await resolver_context.get_latest_token(ProviderType.GITHUB)

    # Assert
    assert result is not None
    assert isinstance(result, str), (
        f'Expected str, got {type(result).__name__}. '
        'get_latest_token must return a string for StaticSecret compatibility.'
    )
    assert result == token_value


@pytest.mark.asyncio
async def test_get_latest_token_returns_string_for_multiple_providers(
    resolver_context, mock_saas_user_auth
):
    """Test that get_latest_token returns strings for all provider types."""
    # Arrange
    provider_tokens = create_provider_tokens(
        {
            ProviderType.GITHUB: 'ghp_github_token',
            ProviderType.GITLAB: 'glpat_gitlab_token',
            ProviderType.BITBUCKET: 'bitbucket_token',
        }
    )
    mock_saas_user_auth.get_provider_tokens = AsyncMock(return_value=provider_tokens)

    # Act & Assert - verify each provider returns a string
    for provider_type, expected_token in [
        (ProviderType.GITHUB, 'ghp_github_token'),
        (ProviderType.GITLAB, 'glpat_gitlab_token'),
        (ProviderType.BITBUCKET, 'bitbucket_token'),
    ]:
        result = await resolver_context.get_latest_token(provider_type)
        assert isinstance(
            result, str
        ), f'Expected str for {provider_type.name}, got {type(result).__name__}'
        assert result == expected_token


@pytest.mark.asyncio
async def test_get_latest_token_returns_none_for_missing_provider(
    resolver_context, mock_saas_user_auth
):
    """Test that get_latest_token returns None when provider is not in tokens."""
    # Arrange - only GitHub token available
    provider_tokens = create_provider_tokens({ProviderType.GITHUB: 'ghp_token'})
    mock_saas_user_auth.get_provider_tokens = AsyncMock(return_value=provider_tokens)

    # Act - request GitLab token which doesn't exist
    result = await resolver_context.get_latest_token(ProviderType.GITLAB)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_token_returns_none_when_no_provider_tokens(
    resolver_context, mock_saas_user_auth
):
    """Test that get_latest_token returns None when no provider tokens exist."""
    # Arrange
    mock_saas_user_auth.get_provider_tokens = AsyncMock(return_value=None)

    # Act
    result = await resolver_context.get_latest_token(ProviderType.GITHUB)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_token_returns_none_for_empty_token(
    resolver_context, mock_saas_user_auth
):
    """Test that get_latest_token returns None when provider token has no value."""
    # Arrange - provider exists but token is None
    provider_tokens = {ProviderType.GITHUB: ProviderToken(token=None)}
    mock_saas_user_auth.get_provider_tokens = AsyncMock(return_value=provider_tokens)

    # Act
    result = await resolver_context.get_latest_token(ProviderType.GITHUB)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_token_can_be_used_with_static_secret(
    resolver_context, mock_saas_user_auth
):
    """Test that get_latest_token result can be used directly with StaticSecret.

    This is a critical integration test to ensure the return value is compatible
    with how it's used in _setup_secrets_for_git_providers.
    """
    # Arrange
    token_value = 'ghp_integration_test_token'
    provider_tokens = create_provider_tokens({ProviderType.GITHUB: token_value})
    mock_saas_user_auth.get_provider_tokens = AsyncMock(return_value=provider_tokens)

    # Act
    token = await resolver_context.get_latest_token(ProviderType.GITHUB)

    # Assert - this should NOT raise a ValidationError
    static_secret = StaticSecret(value=token, description='GITHUB authentication token')
    assert static_secret.get_value() == token_value
