from datetime import timedelta
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.secrets import credential_binding
from openhands.app_server.secrets.credential_binding_models import (
    credential_binding_path,
)
from openhands.app_server.secrets.secrets_store import (
    CredentialVersionConflict,
    ManagedCredentialStore,
    SecretsStore,
)
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey

_CONVERSATION_ID = UUID('11111111-1111-1111-1111-111111111111')
_ORG_ID = UUID('22222222-2222-2222-2222-222222222222')


@pytest.fixture
def jwt_service():
    return JwtService(
        [EncryptionKey(id='test', key=SecretStr('test-secret'), active=True)]
    )


@pytest.fixture
def store():
    result = AsyncMock(spec=ManagedCredentialStore)
    result.load_managed.return_value = (
        '{"tokens":{"refresh_token":"r0"}}',
        'v0',
    )
    result.replace_managed.return_value = 'v1'
    return result


@pytest.fixture
def client(monkeypatch, jwt_service, store):
    user_auth = AsyncMock()
    secrets_store = Mock(spec=SecretsStore)
    secrets_store.managed_credentials = store
    user_auth.get_secrets_store.return_value = secrets_store
    get_for_user = AsyncMock(return_value=user_auth)
    monkeypatch.setattr(credential_binding, 'get_for_user', get_for_user)
    app = FastAPI()
    app.include_router(credential_binding.router)
    app.dependency_overrides[credential_binding.jwt_service_dependency.dependency] = (
        lambda: jwt_service
    )
    with TestClient(app) as test_client:
        yield test_client, get_for_user


def _token(jwt_service: JwtService, **updates) -> str:
    payload = {
        'purpose': 'credential-binding',
        'user_id': 'user-id',
        'organization_id': str(_ORG_ID),
        'conversation_id': str(_CONVERSATION_ID),
        'runtime_id': 'runtime-id',
        'secret_name': 'CODEX_AUTH_JSON',
        'actions': ['load', 'replace'],
    }
    payload.update(updates)
    return jwt_service.create_jws_token(payload, expires_in=timedelta(minutes=5))


def _path(conversation_id: UUID = _CONVERSATION_ID) -> str:
    return credential_binding_path(conversation_id, 'CODEX_AUTH_JSON')


def test_load_uses_token_scope(client, jwt_service, store):
    test_client, get_for_user = client

    response = test_client.get(
        _path(), headers={'Authorization': f'Bearer {_token(jwt_service)}'}
    )

    assert response.status_code == 200
    assert response.json() == {
        'value': '{"tokens":{"refresh_token":"r0"}}',
        'version': 'v0',
    }
    assert response.headers['Cache-Control'] == 'no-store'
    get_for_user.assert_awaited_once_with('user-id')
    store.load_managed.assert_awaited_once_with('CODEX_AUTH_JSON', _ORG_ID)


def test_replace_uses_compare_and_swap(client, jwt_service, store):
    test_client, _ = client
    replacement = '{"tokens":{"refresh_token":"r1"}}'

    response = test_client.put(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
        json={'expected_version': 'v0', 'value': replacement},
    )

    assert response.status_code == 200
    assert response.json() == {'version': 'v1'}
    assert response.headers['Cache-Control'] == 'no-store'
    store.replace_managed.assert_awaited_once_with(
        'CODEX_AUTH_JSON', 'v0', replacement, _ORG_ID
    )


def test_scope_mismatch_is_rejected_before_store(client, jwt_service, store):
    test_client, get_for_user = client

    response = test_client.get(
        _path(UUID('33333333-3333-3333-3333-333333333333')),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
    )

    assert response.status_code == 403
    get_for_user.assert_not_awaited()
    store.load_managed.assert_not_awaited()


def test_unknown_credential_is_not_exposed(client, jwt_service, store):
    test_client, get_for_user = client

    response = test_client.get(
        credential_binding_path(_CONVERSATION_ID, 'OTHER_SECRET'),
        headers={
            'Authorization': f'Bearer {_token(jwt_service, secret_name="OTHER_SECRET")}'
        },
    )

    assert response.status_code == 404
    get_for_user.assert_not_awaited()
    store.load_managed.assert_not_awaited()


def test_missing_exact_actions_is_rejected(client, jwt_service):
    test_client, _ = client

    response = test_client.get(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service, actions=["load"])}'},
    )

    assert response.status_code == 403


def test_invalid_replacement_is_rejected(client, jwt_service, store):
    test_client, _ = client

    response = test_client.put(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
        json={'expected_version': 'v0', 'value': '{}'},
    )

    assert response.status_code == 422
    store.replace_managed.assert_not_awaited()


def test_conflict_is_reported(client, jwt_service, store):
    test_client, _ = client
    store.replace_managed.side_effect = CredentialVersionConflict

    response = test_client.put(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
        json={
            'expected_version': 'stale',
            'value': '{"tokens":{"refresh_token":"r1"}}',
        },
    )

    assert response.status_code == 409
