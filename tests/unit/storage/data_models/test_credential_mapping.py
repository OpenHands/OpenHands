"""Tests for CredentialMapping model."""

import pytest

from openhands.storage.data_models.credential_mapping import CredentialMapping


def test_credential_mapping_creation():
    """Test creating a basic credential mapping."""
    mapping = CredentialMapping(
        resource_pattern='npm.example.com',
        credential_name='NPM_TOKEN',
        auth_method='bearer_token',
        resource_type='npm',
        description='NPM registry token',
    )

    assert mapping.resource_pattern == 'npm.example.com'
    assert mapping.credential_name == 'NPM_TOKEN'
    assert mapping.auth_method == 'bearer_token'
    assert mapping.resource_type == 'npm'
    assert mapping.description == 'NPM registry token'
    assert mapping.auth_header is None


def test_credential_mapping_with_header():
    """Test creating a credential mapping with custom header."""
    mapping = CredentialMapping(
        resource_pattern='api.example.com',
        credential_name='API_KEY',
        auth_method='header',
        auth_header='X-API-Key',
        description='Custom API key',
    )

    assert mapping.auth_method == 'header'
    assert mapping.auth_header == 'X-API-Key'


def test_credential_mapping_immutability():
    """Test that CredentialMapping is immutable."""
    mapping = CredentialMapping(
        resource_pattern='test.com',
        credential_name='TOKEN',
        auth_method='api_key',
    )

    # Attempting to modify should raise an error
    with pytest.raises(Exception):  # pydantic will raise ValidationError
        mapping.resource_pattern = 'other.com'


def test_credential_mapping_auth_methods():
    """Test all supported auth methods."""
    auth_methods = ['api_key', 'bearer_token', 'basic_auth', 'header']

    for method in auth_methods:
        kwargs = {
            'resource_pattern': 'test.com',
            'credential_name': 'TOKEN',
            'auth_method': method,
        }
        if method == 'header':
            kwargs['auth_header'] = 'X-Custom-Header'

        mapping = CredentialMapping(**kwargs)
        assert mapping.auth_method == method


@pytest.mark.parametrize(
    'resource_pattern,credential_name,auth_method,auth_header,should_validate',
    [
        ('test.com', 'TOKEN', 'bearer_token', None, True),
        ('test.com', 'TOKEN', 'header', 'X-API-Key', True),
        ('test.com', 'TOKEN', 'header', None, False),  # Missing auth_header
    ],
)
def test_credential_mapping_validation(
    resource_pattern, credential_name, auth_method, auth_header, should_validate
):
    """Test credential mapping validation."""
    kwargs = {
        'resource_pattern': resource_pattern,
        'credential_name': credential_name,
        'auth_method': auth_method,
    }
    if auth_header:
        kwargs['auth_header'] = auth_header

    if should_validate:
        mapping = CredentialMapping(**kwargs)
        assert mapping.resource_pattern == resource_pattern
    else:
        with pytest.raises(Exception):
            CredentialMapping(**kwargs)

