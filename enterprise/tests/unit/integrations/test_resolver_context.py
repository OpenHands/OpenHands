"""Test for ResolverUserContext get_secrets conversion logic.

This test focuses on testing the actual ResolverUserContext implementation.
"""

from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from enterprise.integrations.resolver_context import ResolverUserContext

# Import the real classes we want to test
from openhands.integrations.provider import CustomSecret

# Import the SDK types we need for testing
from openhands.sdk.conversation.secret_source import SecretSource, StaticSecret
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
