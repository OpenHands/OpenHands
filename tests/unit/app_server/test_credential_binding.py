from datetime import timedelta
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.secrets import credential_binding
from openhands.app_server.secrets.credential_binding_models import (
    MAX_CREDENTIAL_BINDING_TOKEN_TIMEOUT_SECONDS,
    credential_binding_path,
    credential_binding_renewal_path,
)
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict
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
    result = AsyncMock()
    result.load_versioned.return_value = (
        '{"tokens":{"refresh_token":"r0"}}',
        'v0',
    )
    result.replace_versioned.return_value = 'v1'
    return result


@pytest.fixture
def sandbox_validator(monkeypatch):
    validator = AsyncMock(
        return_value=SandboxInfo(
            id='runtime-id',
            created_by_user_id='user-id',
            sandbox_spec_id='spec-id',
            status=SandboxStatus.RUNNING,
            session_api_key='session-key',
        )
    )
    monkeypatch.setattr(credential_binding, 'validate_session_key', validator)
    return validator


@pytest.fixture
def client(monkeypatch, jwt_service, store, sandbox_validator):
    user_auth = AsyncMock()
    user_auth.get_secrets_store.return_value = store
    get_for_user = AsyncMock(return_value=user_auth)
    monkeypatch.setattr(credential_binding, 'get_for_user', get_for_user)
    app = FastAPI()
    app.include_router(credential_binding.router)
    app.dependency_overrides[credential_binding.jwt_service_dependency.dependency] = (
        lambda: jwt_service
    )
    with TestClient(app) as test_client:
        yield test_client, get_for_user


def _token(
    jwt_service: JwtService,
    *,
    expires_in: timedelta = timedelta(minutes=5),
    include_renewal_ttl: bool = True,
    **updates,
) -> str:
    payload = {
        'purpose': 'credential-binding',
        'user_id': 'user-id',
        'organization_id': str(_ORG_ID),
        'conversation_id': str(_CONVERSATION_ID),
        'runtime_id': 'runtime-id',
        'secret_name': 'CODEX_AUTH_JSON',
        'actions': ['load', 'replace'],
    }
    if include_renewal_ttl:
        payload['renewal_ttl_seconds'] = 15 * 86400
    payload.update(updates)
    return jwt_service.create_jws_token(payload, expires_in=expires_in)


def _path(conversation_id: UUID = _CONVERSATION_ID) -> str:
    return credential_binding_path(conversation_id, 'CODEX_AUTH_JSON')


def _renewal_path(conversation_id: UUID = _CONVERSATION_ID) -> str:
    return credential_binding_renewal_path(conversation_id, 'CODEX_AUTH_JSON')


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
    store.load_versioned.assert_awaited_once_with('CODEX_AUTH_JSON', _ORG_ID)


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
    store.replace_versioned.assert_awaited_once_with(
        'CODEX_AUTH_JSON', 'v0', replacement, _ORG_ID
    )


def test_pre_renewal_token_remains_compatible(client, jwt_service, store):
    test_client, _ = client
    authorization = f'Bearer {_token(jwt_service, include_renewal_ttl=False)}'

    load_response = test_client.get(
        _path(),
        headers={'Authorization': authorization},
    )
    replace_response = test_client.put(
        _path(),
        headers={'Authorization': authorization},
        json={
            'expected_version': 'v0',
            'value': '{"tokens":{"refresh_token":"r1"}}',
        },
    )
    renewal_response = test_client.post(
        _renewal_path(),
        headers={
            'Authorization': authorization,
            'X-Session-API-Key': 'session-key',
        },
    )

    assert load_response.status_code == 200
    assert replace_response.status_code == 200
    assert renewal_response.status_code == 403
    store.load_versioned.assert_awaited_once()
    store.replace_versioned.assert_awaited_once()


def test_binding_token_remains_valid_after_previous_one_hour_boundary(
    client, jwt_service, store
):
    from openhands.agent_server.utils import utc_now

    with patch(
        'openhands.app_server.services.jwt_service.utc_now',
        return_value=utc_now() - timedelta(hours=2),
    ):
        token = _token(jwt_service, expires_in=timedelta(days=15))
    test_client, _ = client
    headers = {'Authorization': f'Bearer {token}'}

    load_response = test_client.get(_path(), headers=headers)
    replace_response = test_client.put(
        _path(),
        headers=headers,
        json={
            'expected_version': 'v0',
            'value': '{"tokens":{"refresh_token":"r1"}}',
        },
    )

    assert load_response.status_code == 200
    assert replace_response.status_code == 200
    store.load_versioned.assert_awaited_once()
    store.replace_versioned.assert_awaited_once()


def test_expired_binding_token_is_rejected(client, jwt_service, store):
    test_client, get_for_user = client
    token = _token(jwt_service, expires_in=timedelta(seconds=-1))

    response = test_client.get(_path(), headers={'Authorization': f'Bearer {token}'})

    assert response.status_code == 401
    get_for_user.assert_not_awaited()
    store.load_versioned.assert_not_awaited()


def test_expired_binding_token_can_be_successively_renewed(
    client, jwt_service, sandbox_validator
):
    test_client, _ = client
    headers = {
        'Authorization': (
            f'Bearer {_token(jwt_service, expires_in=timedelta(seconds=-1))}'
        ),
        'X-Session-API-Key': 'session-key',
    }

    first = test_client.post(_renewal_path(), headers=headers)
    assert first.status_code == 200
    first_authorization = first.json()['authorization']
    assert first_authorization.startswith('Bearer ')
    assert first.json()['authorization_expires_in_seconds'] == 15 * 86400
    first_claims = jwt_service.verify_jws_token(
        first_authorization.removeprefix('Bearer ')
    )
    assert first_claims['runtime_id'] == 'runtime-id'
    assert first_claims['renewal_ttl_seconds'] == 15 * 86400
    assert first_claims['exp'] - first_claims['iat'] == 15 * 86400

    second = test_client.post(
        _renewal_path(),
        headers={**headers, 'Authorization': first_authorization},
    )

    assert second.status_code == 200
    assert second.json()['authorization'].startswith('Bearer ')
    assert second.json()['authorization_expires_in_seconds'] == 15 * 86400
    assert second.headers['Cache-Control'] == 'no-store'
    assert sandbox_validator.await_count == 2


@pytest.mark.parametrize(
    ('sandbox_id', 'owner_id'),
    (('wrong-runtime', 'user-id'), ('runtime-id', 'wrong-user')),
)
def test_renewal_rejects_runtime_scope_mismatch(
    client,
    jwt_service,
    sandbox_validator,
    sandbox_id,
    owner_id,
):
    sandbox_validator.return_value = SandboxInfo(
        id=sandbox_id,
        created_by_user_id=owner_id,
        sandbox_spec_id='spec-id',
        status=SandboxStatus.RUNNING,
        session_api_key='session-key',
    )
    test_client, _ = client

    response = test_client.post(
        _renewal_path(),
        headers={
            'Authorization': f'Bearer {_token(jwt_service)}',
            'X-Session-API-Key': 'session-key',
        },
    )

    assert response.status_code == 403


def test_renewal_rejects_non_running_session(
    client,
    jwt_service,
    sandbox_validator,
):
    sandbox_validator.side_effect = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        'Sandbox is not running',
    )
    test_client, _ = client

    response = test_client.post(
        _renewal_path(),
        headers={
            'Authorization': f'Bearer {_token(jwt_service)}',
            'X-Session-API-Key': 'session-key',
        },
    )

    assert response.status_code == 401
    sandbox_validator.assert_awaited_once_with('session-key')


@pytest.mark.parametrize(
    'renewal_ttl_seconds',
    (True, 0, MAX_CREDENTIAL_BINDING_TOKEN_TIMEOUT_SECONDS + 1),
)
def test_renewal_rejects_invalid_signed_ttl(
    client,
    jwt_service,
    sandbox_validator,
    renewal_ttl_seconds,
):
    test_client, _ = client

    response = test_client.post(
        _renewal_path(),
        headers={
            'Authorization': (
                f'Bearer {_token(jwt_service, renewal_ttl_seconds=renewal_ttl_seconds)}'
            ),
            'X-Session-API-Key': 'session-key',
        },
    )

    assert response.status_code == 403
    sandbox_validator.assert_not_awaited()


def test_scope_mismatch_is_rejected_before_store(client, jwt_service, store):
    test_client, get_for_user = client

    response = test_client.get(
        _path(UUID('33333333-3333-3333-3333-333333333333')),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
    )

    assert response.status_code == 403
    get_for_user.assert_not_awaited()
    store.load_versioned.assert_not_awaited()


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
    store.load_versioned.assert_not_awaited()


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
    store.replace_versioned.assert_not_awaited()


def test_conflict_is_reported(client, jwt_service, store):
    test_client, _ = client
    store.replace_versioned.side_effect = CredentialVersionConflict

    response = test_client.put(
        _path(),
        headers={'Authorization': f'Bearer {_token(jwt_service)}'},
        json={
            'expected_version': 'stale',
            'value': '{"tokens":{"refresh_token":"r1"}}',
        },
    )

    assert response.status_code == 409
