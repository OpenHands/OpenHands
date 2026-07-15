import hashlib
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from server.routes import codex_auth

from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey


class _FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    async def set(self, key, value, *, ex, nx):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def get(self, key):
        return self.data.get(key)

    async def eval(self, script, _num_keys, key, *args):
        if script == codex_auth._REPLACE_OWNER_SCRIPT:
            current, replacement, _ttl = args
            if self.data.get(key) != current:
                return 0
            self.data[key] = replacement
            return 1
        if script == codex_auth._TOUCH_OWNER_SCRIPT:
            owner, _ttl = args
            return int(self.data.get(key) == owner)
        if script == codex_auth._RELEASE_OWNER_SCRIPT:
            (owner,) = args
            if self.data.get(key) != owner:
                return 0
            del self.data[key]
            return 1
        raise AssertionError('Unexpected Redis script')


class _FakeStore:
    def __init__(self, value: str):
        self.value = value
        self.update_calls = 0

    async def get_custom_secret_value(self, secret_name: str):
        assert secret_name == 'CODEX_AUTH_JSON'
        return self.value

    async def compare_and_swap_custom_secret(
        self, secret_name: str, expected_digest: str, value: str
    ):
        assert secret_name == 'CODEX_AUTH_JSON'
        self.update_calls += 1
        current_digest = hashlib.sha256(self.value.encode()).hexdigest()
        if current_digest != expected_digest:
            return False
        self.value = value
        return True


@pytest.fixture
def jwt_service():
    return JwtService(
        [
            EncryptionKey(
                id='test',
                key=SecretStr('signing-secret-at-least-32-bytes'),
                active=True,
            )
        ]
    )


@pytest.fixture
def app(jwt_service):
    app = FastAPI()
    app.include_router(codex_auth.router)
    app.dependency_overrides[codex_auth.jwt_service_dependency.dependency] = (
        lambda: jwt_service
    )
    return app


def _sandbox(sandbox_id: str) -> SandboxInfo:
    return SandboxInfo(
        id=sandbox_id,
        created_by_user_id='user-1',
        sandbox_spec_id='spec',
        status=SandboxStatus.RUNNING,
        session_api_key=f'session-{sandbox_id}',
    )


def _token(jwt_service: JwtService, conversation_id: UUID, sandbox_id: str) -> str:
    return jwt_service.create_jws_token(
        {
            'purpose': 'codex-auth',
            'user_id': 'user-1',
            'org_id': '11111111-1111-1111-1111-111111111111',
            'sandbox_id': sandbox_id,
            'conversation_id': str(conversation_id),
            'secret_name': 'CODEX_AUTH_JSON',
        }
    )


def _headers(jwt_service, conversation_id, sandbox_id):
    return {
        'X-Session-API-Key': f'session-{sandbox_id}',
        'X-Codex-Auth-Token': _token(jwt_service, conversation_id, sandbox_id),
    }


def test_rotation_is_returned_to_next_conversation(app, jwt_service):
    first_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    second_id = UUID('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')
    original = '{"tokens":{"refresh_token":"r0"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    store = _FakeStore(original)
    redis = _FakeRedis()
    sandboxes = {
        'session-a': _sandbox('a'),
        'session-b': _sandbox('b'),
    }

    with (
        patch.object(
            codex_auth,
            'validate_session_key',
            AsyncMock(side_effect=lambda key: sandboxes[key]),
        ),
        patch.object(codex_auth, '_get_store', AsyncMock(return_value=store)),
        patch.object(codex_auth, 'get_redis_client_async', return_value=redis),
    ):
        client = TestClient(app)
        first_headers = _headers(jwt_service, first_id, 'a')
        response = client.get(
            f'/api/internal/conversations/{first_id}/codex-auth',
            headers=first_headers,
        )
        assert response.status_code == 200
        assert response.text == original

        response = client.head(
            f'/api/internal/conversations/{first_id}/codex-auth',
            headers=first_headers,
        )
        assert response.status_code == 204

        response = client.put(
            f'/api/internal/conversations/{first_id}/codex-auth',
            headers=first_headers,
            json={
                'expected_digest': hashlib.sha256(original.encode()).hexdigest(),
                'value': rotated,
            },
        )
        assert response.status_code == 204
        assert store.value == rotated

        response = client.delete(
            f'/api/internal/conversations/{first_id}/codex-auth',
            headers=first_headers,
        )
        assert response.status_code == 204

        response = client.get(
            f'/api/internal/conversations/{second_id}/codex-auth',
            headers=_headers(jwt_service, second_id, 'b'),
        )
        assert response.status_code == 200
        assert response.text == rotated


def test_second_active_conversation_is_rejected(app, jwt_service):
    first_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    second_id = UUID('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')
    store = _FakeStore('{"tokens":{"refresh_token":"r0"}}')
    redis = _FakeRedis()
    sandboxes = {
        'session-a': _sandbox('a'),
        'session-b': _sandbox('b'),
    }

    with (
        patch.object(
            codex_auth,
            'validate_session_key',
            AsyncMock(side_effect=lambda key: sandboxes[key]),
        ),
        patch.object(codex_auth, '_get_store', AsyncMock(return_value=store)),
        patch.object(codex_auth, 'get_redis_client_async', return_value=redis),
        patch.object(codex_auth, '_sandbox_is_running', AsyncMock(return_value=True)),
    ):
        client = TestClient(app)
        first = client.get(
            f'/api/internal/conversations/{first_id}/codex-auth',
            headers=_headers(jwt_service, first_id, 'a'),
        )
        second = client.get(
            f'/api/internal/conversations/{second_id}/codex-auth',
            headers=_headers(jwt_service, second_id, 'b'),
        )

    assert first.status_code == 200
    assert second.status_code == 409
    assert 'already in use' in second.json()['detail']


def test_digest_conflict_never_overwrites_current_value(app, jwt_service):
    conversation_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    original = '{"tokens":{"refresh_token":"r0"}}'
    current = '{"tokens":{"refresh_token":"newer"}}'
    stale_update = '{"tokens":{"refresh_token":"stale"}}'
    store = _FakeStore(original)
    redis = _FakeRedis()

    with (
        patch.object(
            codex_auth,
            'validate_session_key',
            AsyncMock(return_value=_sandbox('a')),
        ),
        patch.object(codex_auth, '_get_store', AsyncMock(return_value=store)),
        patch.object(codex_auth, 'get_redis_client_async', return_value=redis),
    ):
        client = TestClient(app)
        headers = _headers(jwt_service, conversation_id, 'a')
        assert (
            client.get(
                f'/api/internal/conversations/{conversation_id}/codex-auth',
                headers=headers,
            ).status_code
            == 200
        )
        store.value = current
        response = client.put(
            f'/api/internal/conversations/{conversation_id}/codex-auth',
            headers=headers,
            json={
                'expected_digest': hashlib.sha256(original.encode()).hexdigest(),
                'value': stale_update,
            },
        )

    assert response.status_code == 409
    assert store.value == current


def test_invalid_document_does_not_echo_credentials(app, jwt_service):
    conversation_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    leaked_value = '{"tokens":{"access_token":"never-echo-this"}}'
    store = _FakeStore('{"tokens":{"refresh_token":"r0"}}')
    redis = _FakeRedis()

    with (
        patch.object(
            codex_auth,
            'validate_session_key',
            AsyncMock(return_value=_sandbox('a')),
        ),
        patch.object(codex_auth, '_get_store', AsyncMock(return_value=store)),
        patch.object(codex_auth, 'get_redis_client_async', return_value=redis),
    ):
        client = TestClient(app)
        headers = _headers(jwt_service, conversation_id, 'a')
        client.get(
            f'/api/internal/conversations/{conversation_id}/codex-auth',
            headers=headers,
        )
        response = client.put(
            f'/api/internal/conversations/{conversation_id}/codex-auth',
            headers=headers,
            json={
                'expected_digest': '0' * 64,
                'value': leaked_value,
            },
        )

    assert response.status_code == 400
    assert 'never-echo-this' not in response.text
    assert store.update_calls == 0


def test_conversation_scope_mismatch_cannot_read_credentials(app, jwt_service):
    conversation_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    other_id = UUID('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')
    store = _FakeStore('{"tokens":{"refresh_token":"never-return-this"}}')
    get_store = AsyncMock(return_value=store)

    with (
        patch.object(
            codex_auth,
            'validate_session_key',
            AsyncMock(return_value=_sandbox('a')),
        ),
        patch.object(codex_auth, '_get_store', get_store),
    ):
        response = TestClient(app).get(
            f'/api/internal/conversations/{conversation_id}/codex-auth',
            headers=_headers(jwt_service, other_id, 'a'),
        )

    assert response.status_code == 403
    assert 'never-return-this' not in response.text
    get_store.assert_not_awaited()
