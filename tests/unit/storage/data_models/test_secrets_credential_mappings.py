"""Tests for Secrets model with credential mappings."""

from types import MappingProxyType

import pytest
from pydantic import SecretStr

from openhands.integrations.provider import CustomSecret, ProviderToken, ProviderType
from openhands.storage.data_models.credential_mapping import CredentialMapping
from openhands.storage.data_models.secrets import Secrets


def test_secrets_with_credential_mappings():
    """Test creating Secrets with credential mappings."""
    custom_secrets = {
        'NPM_TOKEN': CustomSecret(
            secret=SecretStr('npm_token_value'), description='NPM token'
        )
    }

    credential_mappings = {
        'mapping1': CredentialMapping(
            resource_pattern='npm.example.com',
            credential_name='NPM_TOKEN',
            auth_method='bearer_token',
        )
    }

    secrets = Secrets(
        custom_secrets=MappingProxyType(custom_secrets),
        provider_tokens=MappingProxyType({}),
        credential_mappings=MappingProxyType(credential_mappings),
    )

    assert len(secrets.credential_mappings) == 1
    assert 'mapping1' in secrets.credential_mappings
    assert (
        secrets.credential_mappings['mapping1'].resource_pattern
        == 'npm.example.com'
    )


def test_secrets_serialization_with_credential_mappings():
    """Test serializing Secrets with credential mappings."""
    custom_secrets = {
        'TOKEN': CustomSecret(
            secret=SecretStr('secret_value'), description='Token'
        )
    }

    credential_mappings = {
        'map1': CredentialMapping(
            resource_pattern='test.com',
            credential_name='TOKEN',
            auth_method='bearer_token',
            resource_type='api',
            description='Test mapping',
        )
    }

    secrets = Secrets(
        custom_secrets=MappingProxyType(custom_secrets),
        provider_tokens=MappingProxyType({}),
        credential_mappings=MappingProxyType(credential_mappings),
    )

    # Serialize without exposing secrets
    serialized = secrets.model_dump()
    assert 'credential_mappings' in serialized
    assert 'map1' in serialized['credential_mappings']
    mapping_data = serialized['credential_mappings']['map1']
    assert mapping_data['resource_pattern'] == 'test.com'
    assert mapping_data['credential_name'] == 'TOKEN'
    assert mapping_data['auth_method'] == 'bearer_token'
    assert mapping_data['resource_type'] == 'api'
    assert mapping_data['description'] == 'Test mapping'


def test_secrets_deserialization_with_credential_mappings():
    """Test deserializing Secrets with credential mappings."""
    data = {
        'custom_secrets': {
            'TOKEN': {
                'secret': 'secret_value',
                'description': 'Token',
            }
        },
        'provider_tokens': {},
        'credential_mappings': {
            'map1': {
                'resource_pattern': 'test.com',
                'credential_name': 'TOKEN',
                'auth_method': 'bearer_token',
                'auth_header': None,
                'resource_type': 'api',
                'description': 'Test mapping',
            }
        },
    }

    secrets = Secrets(**data)

    assert len(secrets.credential_mappings) == 1
    assert 'map1' in secrets.credential_mappings
    mapping = secrets.credential_mappings['map1']
    assert mapping.resource_pattern == 'test.com'
    assert mapping.credential_name == 'TOKEN'
    assert mapping.auth_method == 'bearer_token'


def test_secrets_backward_compatibility_no_mappings():
    """Test that Secrets without credential_mappings still works (backward compatibility)."""
    data = {
        'custom_secrets': {
            'TOKEN': {
                'secret': 'secret_value',
                'description': 'Token',
            }
        },
        'provider_tokens': {},
        # No credential_mappings field
    }

    secrets = Secrets(**data)

    # Should have empty credential_mappings
    assert len(secrets.credential_mappings) == 0
    assert isinstance(secrets.credential_mappings, MappingProxyType)


def test_secrets_json_serialization():
    """Test JSON serialization of Secrets with credential mappings."""
    custom_secrets = {
        'TOKEN': CustomSecret(
            secret=SecretStr('secret_value'), description='Token'
        )
    }

    credential_mappings = {
        'map1': CredentialMapping(
            resource_pattern='test.com',
            credential_name='TOKEN',
            auth_method='bearer_token',
        )
    }

    secrets = Secrets(
        custom_secrets=MappingProxyType(custom_secrets),
        provider_tokens=MappingProxyType({}),
        credential_mappings=MappingProxyType(credential_mappings),
    )

    # Test JSON serialization
    json_str = secrets.model_dump_json()
    assert 'credential_mappings' in json_str
    assert 'map1' in json_str
    assert 'test.com' in json_str

