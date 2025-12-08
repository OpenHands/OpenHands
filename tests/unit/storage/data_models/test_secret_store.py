from __future__ import annotations

from types import MappingProxyType
from typing import Any

from pydantic import SecretStr

from openhands.integrations.provider import (
    CustomSecret,
    ProviderToken,
    ProviderType,
)
from openhands.storage.data_models.secrets import Secrets


class TestSecrets:
    def test_adding_only_provider_tokens(self):
        """Test adding only provider tokens to the Secrets."""
        # Create provider tokens
        github_token = ProviderToken(
            token=SecretStr('github-token-123'), user_id='user1'
        )
        gitlab_token = ProviderToken(
            token=SecretStr('gitlab-token-456'), user_id='user2'
        )

        # Create a store with only provider tokens
        provider_tokens = {
            ProviderType.GITHUB: github_token,
            ProviderType.GITLAB: gitlab_token,
        }

        # Initialize the store with a dict that will be converted to MappingProxyType
        store = Secrets(provider_tokens=provider_tokens)

        # Verify the tokens were added correctly
        assert isinstance(store.provider_tokens, MappingProxyType)
        assert len(store.provider_tokens) == 2
        assert (
            store.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'github-token-123'
        )
        assert store.provider_tokens[ProviderType.GITHUB].user_id == 'user1'
        assert (
            store.provider_tokens[ProviderType.GITLAB].token.get_secret_value()
            == 'gitlab-token-456'
        )
        assert store.provider_tokens[ProviderType.GITLAB].user_id == 'user2'

        # Verify custom_secrets is empty
        assert isinstance(store.custom_secrets, MappingProxyType)
        assert len(store.custom_secrets) == 0

    def test_adding_only_custom_secrets(self):
        """Test adding only custom secrets to the Secrets."""
        # Create custom secrets
        custom_secrets = {
            'API_KEY': CustomSecret(
                secret=SecretStr('api-key-123'), description='API key'
            ),
            'DATABASE_PASSWORD': CustomSecret(
                secret=SecretStr('db-pass-456'), description='Database password'
            ),
        }

        # Initialize the store with custom secrets
        store = Secrets(custom_secrets=custom_secrets)

        # Verify the custom secrets were added correctly
        assert isinstance(store.custom_secrets, MappingProxyType)
        assert len(store.custom_secrets) == 2
        assert (
            store.custom_secrets['API_KEY'].secret.get_secret_value() == 'api-key-123'
        )
        assert (
            store.custom_secrets['DATABASE_PASSWORD'].secret.get_secret_value()
            == 'db-pass-456'
        )

        # Verify provider_tokens is empty
        assert isinstance(store.provider_tokens, MappingProxyType)
        assert len(store.provider_tokens) == 0

    def test_initializing_with_mixed_types(self):
        """Test initializing the store with mixed types (dict and MappingProxyType)."""
        # Create provider tokens as a dict
        provider_tokens_dict = {
            ProviderType.GITHUB: {'token': 'github-token-123', 'user_id': 'user1'},
        }

        # Create custom secrets as a MappingProxyType
        custom_secret = CustomSecret(
            secret=SecretStr('api-key-123'), description='API key'
        )
        custom_secrets_proxy = MappingProxyType({'API_KEY': custom_secret})

        # Test with dict for provider_tokens and MappingProxyType for custom_secrets
        store1 = Secrets(
            provider_tokens=provider_tokens_dict, custom_secrets=custom_secrets_proxy
        )

        assert isinstance(store1.provider_tokens, MappingProxyType)
        assert isinstance(store1.custom_secrets, MappingProxyType)
        assert (
            store1.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'github-token-123'
        )
        assert (
            store1.custom_secrets['API_KEY'].secret.get_secret_value() == 'api-key-123'
        )

        # Test with MappingProxyType for provider_tokens and dict for custom_secrets
        provider_token = ProviderToken(
            token=SecretStr('gitlab-token-456'), user_id='user2'
        )
        provider_tokens_proxy = MappingProxyType({ProviderType.GITLAB: provider_token})

        # Create custom secrets as a dict
        custom_secrets_dict = {
            'API_KEY': {'secret': 'api-key-123', 'description': 'API key'}
        }

        store2 = Secrets(
            provider_tokens=provider_tokens_proxy, custom_secrets=custom_secrets_dict
        )

        assert isinstance(store2.provider_tokens, MappingProxyType)
        assert isinstance(store2.custom_secrets, MappingProxyType)
        assert (
            store2.provider_tokens[ProviderType.GITLAB].token.get_secret_value()
            == 'gitlab-token-456'
        )
        assert (
            store2.custom_secrets['API_KEY'].secret.get_secret_value() == 'api-key-123'
        )

    def test_model_copy_update_fields(self):
        """Test using model_copy to update fields without affecting other fields."""
        # Create initial store
        github_token = ProviderToken(
            token=SecretStr('github-token-123'), user_id='user1'
        )
        custom_secret = {
            'API_KEY': CustomSecret(
                secret=SecretStr('api-key-123'), description='API key'
            )
        }

        initial_store = Secrets(
            provider_tokens=MappingProxyType({ProviderType.GITHUB: github_token}),
            custom_secrets=MappingProxyType(custom_secret),
        )

        # Update only provider_tokens
        gitlab_token = ProviderToken(
            token=SecretStr('gitlab-token-456'), user_id='user2'
        )
        updated_provider_tokens = MappingProxyType(
            {ProviderType.GITHUB: github_token, ProviderType.GITLAB: gitlab_token}
        )

        updated_store1 = initial_store.model_copy(
            update={'provider_tokens': updated_provider_tokens}
        )

        # Verify provider_tokens was updated but custom_secrets remains the same
        assert len(updated_store1.provider_tokens) == 2
        assert (
            updated_store1.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'github-token-123'
        )
        assert (
            updated_store1.provider_tokens[ProviderType.GITLAB].token.get_secret_value()
            == 'gitlab-token-456'
        )
        assert len(updated_store1.custom_secrets) == 1
        assert (
            updated_store1.custom_secrets['API_KEY'].secret.get_secret_value()
            == 'api-key-123'
        )

        # Update only custom_secrets
        updated_custom_secrets = MappingProxyType(
            {
                'API_KEY': CustomSecret(
                    secret=SecretStr('api-key-123'), description='API key'
                ),
                'DATABASE_PASSWORD': CustomSecret(
                    secret=SecretStr('db-pass-456'), description='DB password'
                ),
            }
        )

        updated_store2 = initial_store.model_copy(
            update={'custom_secrets': updated_custom_secrets}
        )

        # Verify custom_secrets was updated but provider_tokens remains the same
        assert len(updated_store2.provider_tokens) == 1
        assert (
            updated_store2.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'github-token-123'
        )
        assert len(updated_store2.custom_secrets) == 2
        assert (
            updated_store2.custom_secrets['API_KEY'].secret.get_secret_value()
            == 'api-key-123'
        )
        assert (
            updated_store2.custom_secrets['DATABASE_PASSWORD'].secret.get_secret_value()
            == 'db-pass-456'
        )

    def test_serialization_with_expose_secrets(self):
        """Test serializing the Secrets with expose_secrets=True."""
        # Create a store with both provider tokens and custom secrets
        github_token = ProviderToken(
            token=SecretStr('github-token-123'), user_id='user1'
        )
        custom_secrets = {
            'API_KEY': CustomSecret(
                secret=SecretStr('api-key-123'), description='API key'
            )
        }

        store = Secrets(
            provider_tokens=MappingProxyType({ProviderType.GITHUB: github_token}),
            custom_secrets=MappingProxyType(custom_secrets),
        )

        # Test serialization with expose_secrets=True
        serialized_provider_tokens = store.provider_tokens_serializer(
            store.provider_tokens, SerializationInfo(context={'expose_secrets': True})
        )

        serialized_custom_secrets = store.custom_secrets_serializer(
            store.custom_secrets, SerializationInfo(context={'expose_secrets': True})
        )

        # Verify provider tokens are exposed
        assert serialized_provider_tokens['github']['token'] == 'github-token-123'
        assert serialized_provider_tokens['github']['user_id'] == 'user1'

        # Verify custom secrets are exposed
        assert serialized_custom_secrets['API_KEY']['secret'] == 'api-key-123'
        assert serialized_custom_secrets['API_KEY']['description'] == 'API key'

        # Test serialization with expose_secrets=False (default)
        hidden_provider_tokens = store.provider_tokens_serializer(
            store.provider_tokens, SerializationInfo(context={'expose_secrets': False})
        )

        hidden_custom_secrets = store.custom_secrets_serializer(
            store.custom_secrets, SerializationInfo(context={'expose_secrets': False})
        )

        # Verify provider tokens are hidden
        assert hidden_provider_tokens['github']['token'] != 'github-token-123'
        assert '**' in hidden_provider_tokens['github']['token']

        # Verify custom secrets are hidden
        assert hidden_custom_secrets['API_KEY']['secret'] != 'api-key-123'
        assert '**' in hidden_custom_secrets['API_KEY']['secret']

    def test_initializing_provider_tokens_with_mixed_value_types(self):
        """Test initializing provider tokens with both plain strings and SecretStr objects."""
        # Create provider tokens with mixed value types
        # Note: The ProviderToken.from_value method only accepts plain strings in the token field
        # when passed as a dictionary, not SecretStr objects
        provider_tokens_dict = {
            ProviderType.GITHUB: {
                'token': 'github-token-123',  # Plain string
                'user_id': 'user1',
            },
            ProviderType.GITLAB: {
                'token': 'gitlab-token-456',  # Also using plain string
                'user_id': 'user2',
            },
        }

        # For the second provider, create a ProviderToken directly
        gitlab_token = ProviderToken(
            token=SecretStr('gitlab-token-456'), user_id='user2'
        )

        # Create a mixed dictionary with both a dict and a ProviderToken object
        mixed_provider_tokens = {
            ProviderType.GITHUB: provider_tokens_dict[ProviderType.GITHUB],  # Dict
            ProviderType.GITLAB: gitlab_token,  # ProviderToken object
        }

        # Initialize the store
        store = Secrets(provider_tokens=mixed_provider_tokens)

        # Verify all tokens are converted to SecretStr
        assert isinstance(store.provider_tokens, MappingProxyType)
        assert len(store.provider_tokens) == 2

        # Check GitHub token (was plain string in a dict)
        github_token = store.provider_tokens[ProviderType.GITHUB]
        assert isinstance(github_token.token, SecretStr)
        assert github_token.token.get_secret_value() == 'github-token-123'
        assert github_token.user_id == 'user1'

        # Check GitLab token (was a ProviderToken object)
        gitlab_token_result = store.provider_tokens[ProviderType.GITLAB]
        assert isinstance(gitlab_token_result.token, SecretStr)
        assert gitlab_token_result.token.get_secret_value() == 'gitlab-token-456'
        assert gitlab_token_result.user_id == 'user2'

    def test_initializing_custom_secrets_with_mixed_value_types(self):
        """Test initializing custom secrets with both plain strings and SecretStr objects."""
        # Create custom secrets with mixed value types
        custom_secrets_dict = {
            'API_KEY': {
                'secret': 'api-key-123',
                'description': 'API key',
            },  # Dict format
            'DATABASE_PASSWORD': CustomSecret(
                secret=SecretStr('db-pass-456'), description='DB password'
            ),  # CustomSecret object
        }

        # Initialize the store
        store = Secrets(custom_secrets=custom_secrets_dict)

        # Verify all secrets are converted to CustomSecret objects
        assert isinstance(store.custom_secrets, MappingProxyType)
        assert len(store.custom_secrets) == 2

        # Check API_KEY (was dict)
        assert isinstance(store.custom_secrets['API_KEY'], CustomSecret)
        assert (
            store.custom_secrets['API_KEY'].secret.get_secret_value() == 'api-key-123'
        )
        assert store.custom_secrets['API_KEY'].description == 'API key'

        # Check DATABASE_PASSWORD (was CustomSecret)
        assert isinstance(store.custom_secrets['DATABASE_PASSWORD'], CustomSecret)
        assert (
            store.custom_secrets['DATABASE_PASSWORD'].secret.get_secret_value()
            == 'db-pass-456'
        )
        assert store.custom_secrets['DATABASE_PASSWORD'].description == 'DB password'


# Mock class for SerializationInfo since it's not directly importable
class SerializationInfo:
    def __init__(self, context: dict[str, Any] | None = None):
        self.context = context or {}
