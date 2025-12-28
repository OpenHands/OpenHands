from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from storage.saas_secrets_store import SaasSecretsStore
from storage.stored_custom_secrets import StoredCustomSecrets

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.integrations.provider import CustomSecret
from openhands.storage.data_models.secrets import Secrets


@pytest.fixture
def mock_config():
    config = MagicMock(spec=OpenHandsConfig)
    config.jwt_secret = SecretStr('test_secret')
    return config


@pytest.fixture
def secrets_store(session_maker, mock_config):
    return SaasSecretsStore('user-id', session_maker, mock_config)


class TestSaasSecretsStore:
    @pytest.mark.asyncio
    async def test_store_and_load(self, secrets_store):
        # Create a Secrets object with some test data
        user_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'secret_api_token', 'description': ''}
                    ),
                    'db_password': CustomSecret.from_value(
                        {'secret': 'my_password', 'description': ''}
                    ),
                }
            )
        )

        # Store the secrets
        await secrets_store.store(user_secrets)

        # Load the secrets back
        loaded_secrets = await secrets_store.load()

        # Verify the loaded secrets match the original
        assert loaded_secrets is not None
        assert (
            loaded_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'secret_api_token'
        )
        assert (
            loaded_secrets.custom_secrets['db_password'].secret.get_secret_value()
            == 'my_password'
        )

    @pytest.mark.asyncio
    async def test_encryption_decryption(self, secrets_store):
        # Create a Secrets object with sensitive data
        user_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'sensitive_token', 'description': ''}
                    ),
                    'secret_key': CustomSecret.from_value(
                        {'secret': 'sensitive_secret', 'description': ''}
                    ),
                    'normal_data': CustomSecret.from_value(
                        {'secret': 'not_sensitive', 'description': ''}
                    ),
                }
            )
        )

        assert (
            user_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'sensitive_token'
        )
        # Store the secrets
        await secrets_store.store(user_secrets)

        # Verify the data is encrypted in the database
        with secrets_store.session_maker() as session:
            stored = (
                session.query(StoredCustomSecrets)
                .filter(StoredCustomSecrets.keycloak_user_id == 'user-id')
                .first()
            )

            # The sensitive data should be encrypted
            assert stored.secret_value != 'sensitive_token'
            assert stored.secret_value != 'sensitive_secret'
            assert stored.secret_value != 'not_sensitive'

        # Load the secrets and verify decryption works
        loaded_secrets = await secrets_store.load()
        assert (
            loaded_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'sensitive_token'
        )
        assert (
            loaded_secrets.custom_secrets['secret_key'].secret.get_secret_value()
            == 'sensitive_secret'
        )
        assert (
            loaded_secrets.custom_secrets['normal_data'].secret.get_secret_value()
            == 'not_sensitive'
        )

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_kwargs(self, secrets_store):
        # Test encryption and decryption directly
        test_data: dict[str, Any] = {
            'api_token': 'test_token',
            'client_secret': 'test_secret',
            'normal_data': 'not_sensitive',
            'nested': {
                'nested_token': 'nested_secret_value',
                'nested_normal': 'nested_normal_value',
            },
        }

        # Encrypt the data
        secrets_store._encrypt_kwargs(test_data)

        # Sensitive data is encrypted
        assert test_data['api_token'] != 'test_token'
        assert test_data['client_secret'] != 'test_secret'
        assert test_data['normal_data'] != 'not_sensitive'
        assert test_data['nested']['nested_token'] != 'nested_secret_value'
        assert test_data['nested']['nested_normal'] != 'nested_normal_value'

        # Decrypt the data
        secrets_store._decrypt_kwargs(test_data)

        # Verify sensitive data is properly decrypted
        assert test_data['api_token'] == 'test_token'
        assert test_data['client_secret'] == 'test_secret'
        assert test_data['normal_data'] == 'not_sensitive'
        assert test_data['nested']['nested_token'] == 'nested_secret_value'
        assert test_data['nested']['nested_normal'] == 'nested_normal_value'

    @pytest.mark.asyncio
    async def test_empty_user_id(self, secrets_store):
        # Test that load returns None when user_id is empty
        secrets_store.user_id = ''
        assert await secrets_store.load() is None

    @pytest.mark.asyncio
    async def test_update_existing_secrets(self, secrets_store):
        # Create and store initial secrets
        initial_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'initial_token', 'description': ''}
                    ),
                    'other_value': CustomSecret.from_value(
                        {'secret': 'initial_value', 'description': ''}
                    ),
                }
            )
        )
        await secrets_store.store(initial_secrets)

        # Create and store updated secrets
        updated_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'updated_token', 'description': ''}
                    ),
                    'new_value': CustomSecret.from_value(
                        {'secret': 'new_value', 'description': ''}
                    ),
                }
            )
        )
        await secrets_store.store(updated_secrets)

        # Load the secrets and verify they were updated
        loaded_secrets = await secrets_store.load()
        assert (
            loaded_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'updated_token'
        )
        assert 'new_value' in loaded_secrets.custom_secrets
        assert (
            loaded_secrets.custom_secrets['new_value'].secret.get_secret_value()
            == 'new_value'
        )

        # The other_value should not still be present
        assert 'other_value' not in loaded_secrets.custom_secrets

    @pytest.mark.asyncio
    async def test_get_instance(self, mock_config):
        # Test the get_instance class method
        store = await SaasSecretsStore.get_instance(mock_config, 'test-user-id')
        assert isinstance(store, SaasSecretsStore)
        assert store.user_id == 'test-user-id'
        assert store.config == mock_config
