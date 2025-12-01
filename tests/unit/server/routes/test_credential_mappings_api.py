"""Tests for credential mapping API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openhands.server.routes.secrets import app as secrets_app
from openhands.storage import get_file_store
from openhands.storage.secrets.file_secrets_store import FileSecretsStore


@pytest.fixture
def test_client():
    """Create a test client for the secrets API."""
    app = FastAPI()
    app.include_router(secrets_app)

    # Mock SESSION_API_KEY to None to disable authentication in tests
    with patch.dict('os.environ', {'SESSION_API_KEY': ''}, clear=False):
        with patch('openhands.server.dependencies._SESSION_API_KEY', None):
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
async def test_get_credential_mappings_empty(test_client, file_secrets_store):
    """Test getting credential mappings when none exist."""
    response = test_client.get('/api/credentials/mappings')
    assert response.status_code == 200
    data = response.json()
    assert 'credential_mappings' in data
    assert data['credential_mappings'] == []


@pytest.mark.asyncio
async def test_create_credential_mapping(test_client, file_secrets_store):
    """Test creating a credential mapping."""
    # First create a custom secret
    secret_response = test_client.post(
        '/api/secrets',
        json={
            'name': 'NPM_TOKEN',
            'value': 'npm_token_value_123',
            'description': 'NPM registry token',
        },
    )
    assert secret_response.status_code == 201

    # Then create a credential mapping
    mapping_response = test_client.post(
        '/api/credentials/mappings',
        json={
            'resource_pattern': 'npm.example.com',
            'credential_name': 'NPM_TOKEN',
            'auth_method': 'bearer_token',
            'resource_type': 'npm',
            'description': 'NPM registry mapping',
        },
    )

    assert mapping_response.status_code == 201
    data = mapping_response.json()
    assert data['message'] == 'Credential mapping created successfully'
    assert 'mapping_id' in data


@pytest.mark.asyncio
async def test_create_credential_mapping_missing_credential(
    test_client, file_secrets_store
):
    """Test creating a credential mapping with non-existent credential."""
    response = test_client.post(
        '/api/credentials/mappings',
        json={
            'resource_pattern': 'test.com',
            'credential_name': 'NON_EXISTENT',
            'auth_method': 'bearer_token',
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert 'error' in data
    assert 'does not exist' in data['error']


@pytest.mark.asyncio
async def test_create_credential_mapping_header_required(
    test_client, file_secrets_store
):
    """Test that auth_header is required when auth_method is 'header'."""
    # Create a custom secret first
    test_client.post(
        '/api/secrets',
        json={'name': 'API_KEY', 'value': 'key123', 'description': 'API key'},
    )

    response = test_client.post(
        '/api/credentials/mappings',
        json={
            'resource_pattern': 'api.example.com',
            'credential_name': 'API_KEY',
            'auth_method': 'header',
            # Missing auth_header
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert 'error' in data
    assert 'auth_header' in data['error']


@pytest.mark.asyncio
async def test_get_credential_mappings(test_client, file_secrets_store):
    """Test getting all credential mappings."""
    # Create a secret and mapping
    test_client.post(
        '/api/secrets',
        json={'name': 'TOKEN', 'value': 'token123', 'description': 'Token'},
    )

    test_client.post(
        '/api/credentials/mappings',
        json={
            'resource_pattern': 'test.com',
            'credential_name': 'TOKEN',
            'auth_method': 'bearer_token',
        },
    )

    response = test_client.get('/api/credentials/mappings')
    assert response.status_code == 200
    data = response.json()
    assert 'credential_mappings' in data
    assert len(data['credential_mappings']) == 1
    assert data['credential_mappings'][0]['resource_pattern'] == 'test.com'
    assert data['credential_mappings'][0]['credential_name'] == 'TOKEN'


@pytest.mark.asyncio
async def test_update_credential_mapping(test_client, file_secrets_store):
    """Test updating a credential mapping."""
    # Create secret and mapping
    test_client.post(
        '/api/secrets',
        json={'name': 'TOKEN', 'value': 'token123', 'description': 'Token'},
    )

    create_response = test_client.post(
        '/api/credentials/mappings',
        json={
            'resource_pattern': 'test.com',
            'credential_name': 'TOKEN',
            'auth_method': 'bearer_token',
        },
    )
    mapping_id = create_response.json()['mapping_id']

    # Update the mapping
    update_response = test_client.put(
        f'/api/credentials/mappings/{mapping_id}',
        json={
            'resource_pattern': 'updated.com',
            'credential_name': 'TOKEN',
            'auth_method': 'api_key',
            'auth_header': 'X-API-Key',
            'description': 'Updated mapping',
        },
    )

    assert update_response.status_code == 200
    data = update_response.json()
    assert data['message'] == 'Credential mapping updated successfully'


@pytest.mark.asyncio
async def test_delete_credential_mapping(test_client, file_secrets_store):
    """Test deleting a credential mapping."""
    # Create secret and mapping
    test_client.post(
        '/api/secrets',
        json={'name': 'TOKEN', 'value': 'token123', 'description': 'Token'},
    )

    create_response = test_client.post(
        '/api/credentials/mappings',
        json={
            'resource_pattern': 'test.com',
            'credential_name': 'TOKEN',
            'auth_method': 'bearer_token',
        },
    )
    mapping_id = create_response.json()['mapping_id']

    # Delete the mapping
    delete_response = test_client.delete(f'/api/credentials/mappings/{mapping_id}')
    assert delete_response.status_code == 200

    # Verify it's gone
    get_response = test_client.get('/api/credentials/mappings')
    data = get_response.json()
    assert len(data['credential_mappings']) == 0


@pytest.mark.asyncio
async def test_resolve_credential(test_client, file_secrets_store):
    """Test resolving credentials for a URL."""
    # Create secret and mapping
    test_client.post(
        '/api/secrets',
        json={'name': 'TOKEN', 'value': 'token123', 'description': 'Token'},
    )

    test_client.post(
        '/api/credentials/mappings',
        json={
            'resource_pattern': 'test.com',
            'credential_name': 'TOKEN',
            'auth_method': 'bearer_token',
        },
    )

    response = test_client.get('/api/credentials/resolve?url=test.com')
    assert response.status_code == 200
    data = response.json()
    assert data['matched'] is True
    assert 'auth_headers' in data
    assert data['header_count'] > 0


@pytest.mark.asyncio
async def test_resolve_credential_no_match(test_client, file_secrets_store):
    """Test resolving credentials when no match is found."""
    response = test_client.get('/api/credentials/resolve?url=unknown.com')
    assert response.status_code == 200
    data = response.json()
    assert data['matched'] is False
