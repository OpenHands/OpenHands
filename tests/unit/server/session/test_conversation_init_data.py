"""Unit tests for ConversationInitData - specifically testing the field validator.

These tests verify that the immutable_validator correctly converts dict to MappingProxyType
for git_provider_tokens and custom_secrets fields, ensuring type safety.
"""

import warnings
from types import MappingProxyType

import pytest
from pydantic import SecretStr

from openhands.integrations.provider import CustomSecret, ProviderToken, ProviderType
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.storage.data_models.settings import Settings


@pytest.fixture
def base_settings():
    """Create a base Settings object with minimal required fields."""
    return Settings(
        language='en',
        agent='CodeActAgent',
        max_iterations=100,
        llm_model='anthropic/claude-3-5-sonnet-20241022',
        llm_api_key=SecretStr('test_api_key_12345'),
        llm_base_url=None,
    )


class TestConversationInitDataValidator:
    """Test suite for ConversationInitData field validator."""

    def test_git_provider_tokens_dict_converted_to_mappingproxy(self, base_settings):
        """Test that dict passed as git_provider_tokens is converted to MappingProxyType."""
        # Create provider tokens as a regular dict
        provider_tokens_dict = {
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('ghp_test_token_123'), user_id='test_user'
            ),
            ProviderType.GITLAB: ProviderToken(
                token=SecretStr('glpat_test_token_456'), user_id='test_user_2'
            ),
        }

        # Create ConversationInitData with dict
        init_data = ConversationInitData(
            **base_settings.__dict__,
            git_provider_tokens=provider_tokens_dict,
        )

        # Verify it's now a MappingProxyType
        assert isinstance(init_data.git_provider_tokens, MappingProxyType)
        assert ProviderType.GITHUB in init_data.git_provider_tokens
        assert ProviderType.GITLAB in init_data.git_provider_tokens
        assert (
            init_data.git_provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'ghp_test_token_123'
        )

    def test_git_provider_tokens_mappingproxy_preserved(self, base_settings):
        """Test that MappingProxyType passed as git_provider_tokens is converted to MappingProxyType."""
        # Create provider tokens as MappingProxyType
        provider_token = ProviderToken(
            token=SecretStr('ghp_test_token_789'), user_id='test_user_3'
        )
        provider_tokens_proxy = MappingProxyType({ProviderType.GITHUB: provider_token})

        # Create ConversationInitData with MappingProxyType
        init_data = ConversationInitData(
            **base_settings.__dict__,
            git_provider_tokens=provider_tokens_proxy,
        )

        # Verify it's a MappingProxyType (Pydantic may create a new one, but type is preserved)
        assert isinstance(init_data.git_provider_tokens, MappingProxyType)
        assert (
            init_data.git_provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'ghp_test_token_789'
        )
        assert (
            init_data.git_provider_tokens[ProviderType.GITHUB].user_id == 'test_user_3'
        )

    def test_git_provider_tokens_none_preserved(self, base_settings):
        """Test that None passed as git_provider_tokens is preserved."""
        # Create ConversationInitData with None
        init_data = ConversationInitData(
            **base_settings.__dict__,
            git_provider_tokens=None,
        )

        # Verify it's still None
        assert init_data.git_provider_tokens is None

    def test_custom_secrets_dict_converted_to_mappingproxy(self, base_settings):
        """Test that dict passed as custom_secrets is converted to MappingProxyType."""
        # Create custom secrets as a regular dict
        custom_secrets_dict = {
            'API_KEY': CustomSecret(
                secret=SecretStr('api_key_123'), description='API key for service'
            ),
            'DATABASE_URL': CustomSecret(
                secret=SecretStr('postgres://localhost'), description='Database URL'
            ),
        }

        # Create ConversationInitData with dict
        init_data = ConversationInitData(
            **base_settings.__dict__,
            custom_secrets=custom_secrets_dict,
        )

        # Verify it's now a MappingProxyType
        assert isinstance(init_data.custom_secrets, MappingProxyType)
        assert 'API_KEY' in init_data.custom_secrets
        assert 'DATABASE_URL' in init_data.custom_secrets
        assert (
            init_data.custom_secrets['API_KEY'].secret.get_secret_value()
            == 'api_key_123'
        )

    def test_custom_secrets_mappingproxy_preserved(self, base_settings):
        """Test that MappingProxyType passed as custom_secrets is converted to MappingProxyType."""
        # Create custom secrets as MappingProxyType
        custom_secret = CustomSecret(
            secret=SecretStr('api_key_456'), description='API key'
        )
        custom_secrets_proxy = MappingProxyType({'API_KEY': custom_secret})

        # Create ConversationInitData with MappingProxyType
        init_data = ConversationInitData(
            **base_settings.__dict__,
            custom_secrets=custom_secrets_proxy,
        )

        # Verify it's a MappingProxyType (Pydantic may create a new one, but type is preserved)
        assert isinstance(init_data.custom_secrets, MappingProxyType)
        assert (
            init_data.custom_secrets['API_KEY'].secret.get_secret_value()
            == 'api_key_456'
        )
        assert init_data.custom_secrets['API_KEY'].description == 'API key'

    def test_custom_secrets_none_preserved(self, base_settings):
        """Test that None passed as custom_secrets is preserved."""
        # Create ConversationInitData with None
        init_data = ConversationInitData(
            **base_settings.__dict__,
            custom_secrets=None,
        )

        # Verify it's still None
        assert init_data.custom_secrets is None

    def test_both_fields_dict_converted(self, base_settings):
        """Test that both fields are converted when passed as dicts."""
        provider_tokens_dict = {
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('ghp_token'), user_id='user1'
            )
        }
        custom_secrets_dict = {
            'SECRET': CustomSecret(
                secret=SecretStr('secret_value'), description='A secret'
            )
        }

        init_data = ConversationInitData(
            **base_settings.__dict__,
            git_provider_tokens=provider_tokens_dict,
            custom_secrets=custom_secrets_dict,
        )

        # Both should be MappingProxyType
        assert isinstance(init_data.git_provider_tokens, MappingProxyType)
        assert isinstance(init_data.custom_secrets, MappingProxyType)

    def test_empty_dict_converted_to_mappingproxy(self, base_settings):
        """Test that empty dict is converted to empty MappingProxyType."""
        # Create ConversationInitData with empty dicts
        init_data = ConversationInitData(
            **base_settings.__dict__,
            git_provider_tokens={},
            custom_secrets={},
        )

        # Both should be MappingProxyType (even if empty)
        assert isinstance(init_data.git_provider_tokens, MappingProxyType)
        assert isinstance(init_data.custom_secrets, MappingProxyType)
        assert len(init_data.git_provider_tokens) == 0
        assert len(init_data.custom_secrets) == 0

    def test_validator_prevents_mutation(self, base_settings):
        """Test that MappingProxyType prevents mutation of the underlying data."""
        provider_tokens_dict = {
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('ghp_token'), user_id='user1'
            )
        }

        init_data = ConversationInitData(
            **base_settings.__dict__,
            git_provider_tokens=provider_tokens_dict,
        )

        # Verify it's a MappingProxyType (which is immutable)
        assert isinstance(init_data.git_provider_tokens, MappingProxyType)

        # Verify that attempting to modify would raise (MappingProxyType is read-only)
        with pytest.raises(TypeError):
            # MappingProxyType doesn't support item assignment
            init_data.git_provider_tokens[ProviderType.GITLAB] = ProviderToken(
                token=SecretStr('new_token')
            )

    def test_validator_with_settings_dict_unpacking(self, base_settings):
        """Test validator works when creating from unpacked settings dict.

        This simulates the real-world usage in conversation_service.py where
        session_init_args is created from settings.__dict__.
        """
        # Simulate the pattern used in conversation_service.py
        session_init_args = {**base_settings.__dict__}
        session_init_args['git_provider_tokens'] = {
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('ghp_from_dict'), user_id='user_from_dict'
            )
        }

        # Create ConversationInitData from unpacked dict
        init_data = ConversationInitData(**session_init_args)

        # Verify it's converted to MappingProxyType
        assert isinstance(init_data.git_provider_tokens, MappingProxyType)
        assert (
            init_data.git_provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'ghp_from_dict'
        )

    def test_validator_with_mixed_types(self, base_settings):
        """Test validator with one field as dict and one as MappingProxyType."""
        # git_provider_tokens as dict
        provider_tokens_dict = {
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('ghp_dict_token'), user_id='user_dict'
            )
        }

        # custom_secrets as MappingProxyType
        custom_secret = CustomSecret(
            secret=SecretStr('secret_proxy'), description='From proxy'
        )
        custom_secrets_proxy = MappingProxyType({'SECRET': custom_secret})

        init_data = ConversationInitData(
            **base_settings.__dict__,
            git_provider_tokens=provider_tokens_dict,
            custom_secrets=custom_secrets_proxy,
        )

        # Both should be MappingProxyType
        assert isinstance(init_data.git_provider_tokens, MappingProxyType)
        assert isinstance(init_data.custom_secrets, MappingProxyType)
        # Verify the content is preserved (Pydantic may create new MappingProxyType instances)
        assert (
            init_data.git_provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'ghp_dict_token'
        )
        assert (
            init_data.custom_secrets['SECRET'].secret.get_secret_value()
            == 'secret_proxy'
        )


def test_conversation_init_data_no_pydantic_frozen_field_warning():
    """Test that ConversationInitData model does not trigger Pydantic UnsupportedFieldAttributeWarning.

    This test ensures that the 'frozen' parameter is not incorrectly used in Field()
    definitions, which would cause warnings in Pydantic v2 for union types.
    See: https://github.com/All-Hands-AI/infra/issues/860
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        # Re-import to trigger any warnings during model definition
        import importlib

        import openhands.server.session.conversation_init_data

        importlib.reload(openhands.server.session.conversation_init_data)

        # Check for warnings containing 'frozen' which would indicate
        # incorrect usage of frozen=True in Field()
        frozen_warnings = [
            warning for warning in w if 'frozen' in str(warning.message).lower()
        ]

        assert len(frozen_warnings) == 0, (
            f'Pydantic frozen field warnings found: {[str(w.message) for w in frozen_warnings]}'
        )
