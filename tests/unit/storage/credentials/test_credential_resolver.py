"""Tests for CredentialResolver."""

from types import MappingProxyType

import pytest
from pydantic import SecretStr

from openhands.integrations.provider import CustomSecret
from openhands.storage.credentials.resolver import CredentialResolver
from openhands.storage.data_models.credential_mapping import CredentialMapping
from openhands.storage.data_models.secrets import Secrets


@pytest.fixture
def sample_secrets():
    """Create sample secrets with credential mappings."""
    custom_secrets = {
        'NPM_TOKEN': CustomSecret(
            secret=SecretStr('npm_test_token_12345'),
            description='NPM registry token',
        ),
        'API_KEY': CustomSecret(
            secret=SecretStr('api_key_secret_abc123'),
            description='API key',
        ),
        'BASIC_AUTH_PASS': CustomSecret(
            secret=SecretStr('password123'),
            description='Basic auth password',
        ),
    }

    credential_mappings = {
        'mapping1': CredentialMapping(
            resource_pattern='npm.example.com',
            credential_name='NPM_TOKEN',
            auth_method='bearer_token',
            resource_type='npm',
        ),
        'mapping2': CredentialMapping(
            resource_pattern='api.example.com',
            credential_name='API_KEY',
            auth_method='api_key',
            auth_header='X-API-Key',
        ),
        'mapping3': CredentialMapping(
            resource_pattern='https://api.test.com/*',
            credential_name='API_KEY',
            auth_method='bearer_token',
        ),
        'mapping4': CredentialMapping(
            resource_pattern='auth.example.com',
            credential_name='BASIC_AUTH_PASS',
            auth_method='basic_auth',
        ),
    }

    secrets = Secrets(
        custom_secrets=MappingProxyType(custom_secrets),
        provider_tokens=MappingProxyType({}),
        credential_mappings=MappingProxyType(credential_mappings),
    )

    return secrets


def test_resolver_exact_match(sample_secrets):
    """Test resolving credentials with exact URL match."""
    resolver = CredentialResolver(sample_secrets)

    result = resolver.resolve_credential('https://npm.example.com/package')
    assert result is not None

    credential_value, auth_headers = result
    assert credential_value == 'npm_test_token_12345'
    assert 'Authorization' in auth_headers
    assert auth_headers['Authorization'] == 'Bearer npm_test_token_12345'


def test_resolver_domain_match(sample_secrets):
    """Test resolving credentials with domain match."""
    resolver = CredentialResolver(sample_secrets)

    result = resolver.resolve_credential('npm.example.com')
    assert result is not None

    credential_value, auth_headers = result
    assert 'Authorization' in auth_headers
    assert auth_headers['Authorization'] == 'Bearer npm_test_token_12345'


def test_resolver_api_key_method(sample_secrets):
    """Test resolving credentials with API key auth method."""
    resolver = CredentialResolver(sample_secrets)

    result = resolver.resolve_credential('api.example.com')
    assert result is not None

    credential_value, auth_headers = result
    assert 'X-API-Key' in auth_headers
    assert auth_headers['X-API-Key'] == 'api_key_secret_abc123'


def test_resolver_wildcard_pattern(sample_secrets):
    """Test resolving credentials with wildcard pattern."""
    resolver = CredentialResolver(sample_secrets)

    result = resolver.resolve_credential('https://api.test.com/v1/resource')
    assert result is not None

    credential_value, auth_headers = result
    assert 'Authorization' in auth_headers
    assert auth_headers['Authorization'] == 'Bearer api_key_secret_abc123'


def test_resolver_basic_auth(sample_secrets):
    """Test resolving credentials with basic auth method."""
    resolver = CredentialResolver(sample_secrets)

    result = resolver.resolve_credential('auth.example.com')
    assert result is not None

    credential_value, auth_headers = result
    assert 'Authorization' in auth_headers
    assert auth_headers['Authorization'].startswith('Basic ')


def test_resolver_no_match(sample_secrets):
    """Test resolver when no matching credential is found."""
    resolver = CredentialResolver(sample_secrets)

    result = resolver.resolve_credential('unknown.example.com')
    assert result is None


def test_resolver_no_secrets():
    """Test resolver with no secrets."""
    resolver = CredentialResolver(None)
    result = resolver.resolve_credential('any.url.com')
    assert result is None


def test_resolver_missing_credential(sample_secrets):
    """Test resolver when credential mapping references non-existent secret."""
    # Create a mapping that references a non-existent credential
    bad_mapping = {
        'bad_mapping': CredentialMapping(
            resource_pattern='test.com',
            credential_name='NON_EXISTENT',
            auth_method='bearer_token',
        )
    }
    bad_secrets = Secrets(
        custom_secrets=sample_secrets.custom_secrets,
        provider_tokens=sample_secrets.provider_tokens,
        credential_mappings=MappingProxyType(bad_mapping),
    )

    resolver = CredentialResolver(bad_secrets)
    result = resolver.resolve_credential('test.com')
    assert result is None


def test_resolver_list_mappings(sample_secrets):
    """Test listing all credential mappings."""
    resolver = CredentialResolver(sample_secrets)
    mappings = resolver.list_mappings()

    assert len(mappings) == 4
    mapping_ids = {m.resource_pattern for m in mappings}
    assert 'npm.example.com' in mapping_ids
    assert 'api.example.com' in mapping_ids


def test_resolver_get_mapping(sample_secrets):
    """Test getting a specific mapping by ID."""
    resolver = CredentialResolver(sample_secrets)
    mapping = resolver.get_mapping('mapping1')

    assert mapping is not None
    assert mapping.resource_pattern == 'npm.example.com'
    assert mapping.credential_name == 'NPM_TOKEN'


def test_resolver_get_nonexistent_mapping(sample_secrets):
    """Test getting a non-existent mapping."""
    resolver = CredentialResolver(sample_secrets)
    mapping = resolver.get_mapping('nonexistent')
    assert mapping is None


def test_resolver_pattern_matching_edge_cases(sample_secrets):
    """Test resolver with various URL formats."""
    resolver = CredentialResolver(sample_secrets)

    # Test with protocol
    result1 = resolver.resolve_credential('https://npm.example.com')
    assert result1 is not None

    # Test without protocol
    result2 = resolver.resolve_credential('npm.example.com')
    assert result2 is not None

    # Test with path
    result3 = resolver.resolve_credential('npm.example.com/packages')
    assert result3 is not None
