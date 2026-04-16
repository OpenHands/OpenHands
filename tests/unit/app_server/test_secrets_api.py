"""Tests for the custom secrets API endpoints."""
# flake8: noqa: E501

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.secrets.secrets_router import (
    router as secrets_router,
)
from openhands.integrations.provider import (
    CustomSecret,
    ProviderToken,
    ProviderType,
)
from openhands.storage import get_file_store
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.secrets.file_secrets_store import FileSecretsStore


@pytest.fixture
def test_client():
    """Create a test client for the settings API."""
    app = FastAPI()
    app.include_router(secrets_router)

    # Mock SESSION_API_KEY to None to disable authentication in tests
    with patch.dict(os.environ, {'SESSION_API_KEY': ''}, clear=False):
        # Clear the SESSION_API_KEY to disable auth dependency
        with patch('openhands.app_server.utils.dependencies._SESSION_API_KEY', None):
            yield TestClient(app)


@pytest.fixture
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('secrets_store'))


@pytest.fixture
def file_secrets_store(temp_dir):
    file_store = get_file_store('local', temp_dir)
    store = FileSecretsStore(file_store)
    with patch(
        'openhands.storage.secrets.file_secrets_store.FileSecretsStore.get_instance',
        AsyncMock(return_value=store),
    ):
        yield store


@pytest.mark.asyncio
async def test_search_custom_secrets(test_client, file_secrets_store):
    """Test searching custom secrets."""
    # Create initial settings with custom secrets
    custom_secrets = {
        'API_KEY': CustomSecret(secret=SecretStr('api-key-value')),
        'DB_PASSWORD': CustomSecret(secret=SecretStr('db-password-value')),
    }
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(
        custom_secrets=custom_secrets, provider_tokens=provider_tokens
    )

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the GET request
    response = test_client.get('/secrets/search')
    print(response)
    assert response.status_code == 200

    # Check the response
    data = response.json()
    assert 'items' in data
    # Extract just the names from the list of custom secrets
    secret_names = [secret['name'] for secret in data['items']]
    assert sorted(secret_names) == ['API_KEY', 'DB_PASSWORD']
    # Verify pagination field exists
    assert 'next_page_id' in data

    # Verify that the original settings were not modified
    stored_settings = await file_secrets_store.load()
    assert (
        stored_settings.custom_secrets['API_KEY'].secret.get_secret_value()
        == 'api-key-value'
    )
    assert (
        stored_settings.custom_secrets['DB_PASSWORD'].secret.get_secret_value()
        == 'db-password-value'
    )
    assert ProviderType.GITHUB in stored_settings.provider_tokens


@pytest.mark.asyncio
async def test_search_custom_secrets_empty(test_client, file_secrets_store):
    """Test searching custom secrets when there are no custom secrets."""
    # Create initial settings with no custom secrets
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(provider_tokens=provider_tokens, custom_secrets={})

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the GET request
    response = test_client.get('/secrets/search')
    assert response.status_code == 200

    # Check the response
    data = response.json()
    assert 'items' in data
    assert data['items'] == []
    assert data['next_page_id'] is None


@pytest.mark.asyncio
async def test_search_custom_secrets_with_filter(test_client, file_secrets_store):
    """Test searching custom secrets with name filter."""
    # Create initial settings with custom secrets
    custom_secrets = {
        'API_KEY': CustomSecret(secret=SecretStr('api-key-value')),
        'DB_PASSWORD': CustomSecret(secret=SecretStr('db-password-value')),
        'DB_USER': CustomSecret(secret=SecretStr('db-user-value')),
    }
    user_secrets = Secrets(custom_secrets=custom_secrets)

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the GET request with filter
    response = test_client.get('/secrets/search', params={'name__contains': 'DB'})
    assert response.status_code == 200

    # Check the response
    data = response.json()
    assert 'items' in data
    secret_names = [secret['name'] for secret in data['items']]
    assert sorted(secret_names) == ['DB_PASSWORD', 'DB_USER']


@pytest.mark.asyncio
async def test_search_custom_secrets_pagination(test_client, file_secrets_store):
    """Test searching custom secrets with pagination."""
    # Create initial settings with many custom secrets
    custom_secrets = {
        f'SECRET_{i:02d}': CustomSecret(secret=SecretStr(f'value-{i}'))
        for i in range(5)
    }
    user_secrets = Secrets(custom_secrets=custom_secrets)

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the first GET request with limit
    response = test_client.get('/secrets/search', params={'limit': 2})
    assert response.status_code == 200

    # Check the response
    data = response.json()
    assert 'items' in data
    assert len(data['items']) == 2
    # Results should be sorted alphabetically
    assert data['items'][0]['name'] == 'SECRET_00'
    assert data['items'][1]['name'] == 'SECRET_01'
    # Since there are more items, next_page_id should be set
    assert data['next_page_id'] == 'SECRET_01'

    # Make the second GET request with page_id
    response = test_client.get(
        '/secrets/search', params={'limit': 2, 'page_id': data['next_page_id']}
    )
    assert response.status_code == 200

    # Check the response
    data = response.json()
    assert len(data['items']) == 2
    assert data['items'][0]['name'] == 'SECRET_02'
    assert data['items'][1]['name'] == 'SECRET_03'


@pytest.mark.asyncio
async def test_add_custom_secret(test_client, file_secrets_store):
    """Test adding a new custom secret."""
    # Create initial settings with provider tokens but no custom secrets
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(provider_tokens=provider_tokens)

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the POST request to add a custom secret
    add_secret_data = {'name': 'API_KEY', 'value': 'api-key-value', 'description': None}
    response = test_client.post('/secrets', json=add_secret_data)
    assert response.status_code == 201

    # Verify that the settings were stored with the new secret
    stored_settings = await file_secrets_store.load()

    # Check that the secret was added
    assert 'API_KEY' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['API_KEY'].secret.get_secret_value()
        == 'api-key-value'
    )


@pytest.mark.asyncio
async def test_create_custom_secret_with_no_existing_secrets(
    test_client, file_secrets_store
):
    """Test creating a custom secret when there are no existing secrets at all."""
    # Don't store any initial settings - this simulates a completely new user
    # or a situation where the secrets store is empty

    # Make the POST request to add a custom secret
    add_secret_data = {
        'name': 'NEW_API_KEY',
        'value': 'new-api-key-value',
        'description': 'Test API Key',
    }
    response = test_client.post('/secrets', json=add_secret_data)
    assert response.status_code == 201

    # Verify that the settings were stored with the new secret
    stored_settings = await file_secrets_store.load()

    # Check that the secret was added
    assert 'NEW_API_KEY' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['NEW_API_KEY'].secret.get_secret_value()
        == 'new-api-key-value'
    )
    assert stored_settings.custom_secrets['NEW_API_KEY'].description == 'Test API Key'

    # Check that provider_tokens is an empty dict, not None
    assert stored_settings.provider_tokens == {}


@pytest.mark.asyncio
async def test_update_existing_custom_secret(test_client, file_secrets_store):
    """Test updating an existing custom secret's name and description (cannot change value once set)."""
    # Create initial settings with a custom secret
    custom_secrets = {'API_KEY': CustomSecret(secret=SecretStr('old-api-key'))}
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(
        custom_secrets=custom_secrets, provider_tokens=provider_tokens
    )

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the PUT request to update the custom secret
    update_secret_data = {
        'name': 'API_KEY',
        'description': None,
    }
    response = test_client.put('/secrets/API_KEY', json=update_secret_data)
    assert response.status_code == 200

    # Verify that the settings were stored with the updated secret
    stored_settings = await file_secrets_store.load()

    # Check that the secret was updated
    assert 'API_KEY' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['API_KEY'].secret.get_secret_value()
        == 'old-api-key'
    )

    # Check that other settings were preserved
    assert ProviderType.GITHUB in stored_settings.provider_tokens


@pytest.mark.asyncio
async def test_add_multiple_custom_secrets(test_client, file_secrets_store):
    """Test adding multiple custom secrets at once."""
    # Create initial settings with one custom secret
    custom_secrets = {
        'EXISTING_SECRET': CustomSecret(secret=SecretStr('existing-value'))
    }
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(
        custom_secrets=custom_secrets, provider_tokens=provider_tokens
    )

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the POST request to add first custom secret
    add_secret_data1 = {
        'name': 'API_KEY',
        'value': 'api-key-value',
        'description': None,
    }
    response1 = test_client.post('/secrets', json=add_secret_data1)
    assert response1.status_code == 201

    # Make the POST request to add second custom secret
    add_secret_data2 = {
        'name': 'DB_PASSWORD',
        'value': 'db-password-value',
        'description': None,
    }
    response = test_client.post('/secrets', json=add_secret_data2)
    assert response.status_code == 201

    # Verify that the settings were stored with the new secrets
    stored_settings = await file_secrets_store.load()

    # Check that the new secrets were added
    assert 'API_KEY' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['API_KEY'].secret.get_secret_value()
        == 'api-key-value'
    )
    assert 'DB_PASSWORD' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['DB_PASSWORD'].secret.get_secret_value()
        == 'db-password-value'
    )

    # Check that existing secrets were preserved
    assert 'EXISTING_SECRET' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['EXISTING_SECRET'].secret.get_secret_value()
        == 'existing-value'
    )

    # Check that other settings were preserved
    assert ProviderType.GITHUB in stored_settings.provider_tokens


@pytest.mark.asyncio
async def test_delete_custom_secret(test_client, file_secrets_store):
    """Test deleting a custom secret."""
    # Create initial settings with multiple custom secrets
    custom_secrets = {
        'API_KEY': CustomSecret(secret=SecretStr('api-key-value')),
        'DB_PASSWORD': CustomSecret(secret=SecretStr('db-password-value')),
    }
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(
        custom_secrets=custom_secrets, provider_tokens=provider_tokens
    )

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the DELETE request to delete a custom secret
    response = test_client.delete('/secrets/API_KEY')
    assert response.status_code == 200

    # Verify that the settings were stored without the deleted secret
    stored_settings = await file_secrets_store.load()

    # Check that the specified secret was deleted
    assert 'API_KEY' not in stored_settings.custom_secrets

    # Check that other secrets were preserved
    assert 'DB_PASSWORD' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['DB_PASSWORD'].secret.get_secret_value()
        == 'db-password-value'
    )

    # Check that other settings were preserved
    assert ProviderType.GITHUB in stored_settings.provider_tokens


@pytest.mark.asyncio
async def test_delete_nonexistent_custom_secret(test_client, file_secrets_store):
    """Test deleting a custom secret that doesn't exist."""
    # Create initial settings with a custom secret
    custom_secrets = {
        'API_KEY': CustomSecret(secret=SecretStr('api-key-value'), description='')
    }
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(
        custom_secrets=custom_secrets, provider_tokens=provider_tokens
    )

    # Store the initial settings
    await file_secrets_store.store(user_secrets)

    # Make the DELETE request to delete a nonexistent custom secret
    response = test_client.delete('/secrets/NONEXISTENT_KEY')
    assert response.status_code == 404

    # Verify that the settings were stored without changes to existing secrets
    stored_settings = await file_secrets_store.load()

    # Check that the existing secret was preserved
    assert 'API_KEY' in stored_settings.custom_secrets
    assert (
        stored_settings.custom_secrets['API_KEY'].secret.get_secret_value()
        == 'api-key-value'
    )

    # Check that other settings were preserved
    assert ProviderType.GITHUB in stored_settings.provider_tokens


@pytest.mark.asyncio
async def test_add_git_providers_with_host(test_client, file_secrets_store):
    """Test adding git providers with host parameter."""
    # Create initial user secrets
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(token=SecretStr('github-token'))
    }
    user_secrets = Secrets(provider_tokens=provider_tokens)
    await file_secrets_store.store(user_secrets)

    # Mock check_provider_tokens to return empty string (no error)
    with patch(
        'openhands.app_server.secrets.secrets_router.check_provider_tokens',
        AsyncMock(return_value=''),
    ):
        # Add a GitHub provider with a host
        add_provider_data = {
            'provider_tokens': {
                'github': {'token': 'new-github-token', 'host': 'github.enterprise.com'}
            }
        }
        response = test_client.post('/secrets/git-providers', json=add_provider_data)
        assert response.status_code == 200

        # Verify that the settings were stored with the new provider token and host
        stored_secrets = await file_secrets_store.load()
        assert ProviderType.GITHUB in stored_secrets.provider_tokens
        assert (
            stored_secrets.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'new-github-token'
        )
        assert (
            stored_secrets.provider_tokens[ProviderType.GITHUB].host
            == 'github.enterprise.com'
        )


@pytest.mark.asyncio
async def test_add_git_providers_update_host_only(test_client, file_secrets_store):
    """Test updating only the host for an existing provider token."""
    # Create initial user secrets with a token
    provider_tokens = {
        ProviderType.GITHUB: ProviderToken(
            token=SecretStr('github-token'), host='github.com'
        )
    }
    user_secrets = Secrets(provider_tokens=provider_tokens)
    await file_secrets_store.store(user_secrets)

    # Mock check_provider_tokens to return empty string (no error)
    with patch(
        'openhands.app_server.secrets.secrets_router.check_provider_tokens',
        AsyncMock(return_value=''),
    ):
        # Update only the host
        update_host_data = {
            'provider_tokens': {
                'github': {
                    'token': '',  # Empty token means keep existing token
                    'host': 'github.enterprise.com',
                }
            }
        }
        response = test_client.post('/secrets/git-providers', json=update_host_data)
        assert response.status_code == 200

        # Verify that the host was updated but the token remains the same
        stored_secrets = await file_secrets_store.load()
        assert ProviderType.GITHUB in stored_secrets.provider_tokens
        assert (
            stored_secrets.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'github-token'
        )
        assert (
            stored_secrets.provider_tokens[ProviderType.GITHUB].host
            == 'github.enterprise.com'
        )


@pytest.mark.asyncio
async def test_add_git_providers_invalid_token_with_host(
    test_client, file_secrets_store
):
    """Test adding an invalid token with a host."""
    # Create initial user secrets
    user_secrets = Secrets()
    await file_secrets_store.store(user_secrets)

    # Mock validate_provider_token to return None (invalid token)
    with patch(
        'openhands.integrations.utils.validate_provider_token',
        AsyncMock(return_value=None),
    ):
        # Try to add an invalid GitHub provider with a host
        add_provider_data = {
            'provider_tokens': {
                'github': {'token': 'invalid-token', 'host': 'github.enterprise.com'}
            }
        }
        response = test_client.post('/secrets/git-providers', json=add_provider_data)
        assert response.status_code == 401
        assert 'Invalid token' in response.json()['detail']


@pytest.mark.asyncio
async def test_add_multiple_git_providers_with_hosts(test_client, file_secrets_store):
    """Test adding multiple git providers with different hosts."""
    # Create initial user secrets
    user_secrets = Secrets()
    await file_secrets_store.store(user_secrets)

    # Mock check_provider_tokens to return empty string (no error)
    with patch(
        'openhands.app_server.secrets.secrets_router.check_provider_tokens',
        AsyncMock(return_value=''),
    ):
        # Add multiple providers with hosts
        add_providers_data = {
            'provider_tokens': {
                'github': {'token': 'github-token', 'host': 'github.enterprise.com'},
                'gitlab': {'token': 'gitlab-token', 'host': 'gitlab.enterprise.com'},
            }
        }
        response = test_client.post('/secrets/git-providers', json=add_providers_data)
        assert response.status_code == 200

        # Verify that both providers were stored with their respective hosts
        stored_secrets = await file_secrets_store.load()
        assert ProviderType.GITHUB in stored_secrets.provider_tokens
        assert (
            stored_secrets.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
            == 'github-token'
        )
        assert (
            stored_secrets.provider_tokens[ProviderType.GITHUB].host
            == 'github.enterprise.com'
        )

        assert ProviderType.GITLAB in stored_secrets.provider_tokens
        assert (
            stored_secrets.provider_tokens[ProviderType.GITLAB].token.get_secret_value()
            == 'gitlab-token'
        )
        assert (
            stored_secrets.provider_tokens[ProviderType.GITLAB].host
            == 'gitlab.enterprise.com'
        )
