import asyncio
import base64
import hashlib
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr
from server.routes import codex_auth

from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey

_FIRST_ID = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
_SECOND_ID = UUID('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')


class _FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.lock_calls: list[dict] = []

    async def set(self, key, value, *, ex, nx):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def get(self, key):
        return self.data.get(key)

    def lock(self, name, **kwargs):
        self.lock_calls.append({'name': name, **kwargs})
        return _FakeLock(self, name)


class _FakeLock:
    def __init__(self, redis: _FakeRedis, name: str):
        self.redis = redis
        self.name = name

    async def __aenter__(self):
        lock = self.redis.locks.setdefault(self.name, asyncio.Lock())
        await lock.acquire()
        self.redis.data[self.name] = 'locked'
        return self

    async def __aexit__(self, *_args):
        self.redis.data.pop(self.name, None)
        self.redis.locks[self.name].release()


class _FakeStore:
    def __init__(self, value: str):
        self.value = value
        self.update_calls = 0

    async def get_value(self):
        return self.value

    async def compare_and_swap(self, expected_digest: str, value: str):
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


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    redis = _FakeRedis()
    monkeypatch.setattr(codex_auth, 'get_redis_client_async', lambda: redis)
    return redis


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


@pytest.fixture
def broker(monkeypatch):
    store = _FakeStore('{"tokens":{"refresh_token":"r0"}}')
    sandboxes = {
        'session-a': _sandbox('a'),
        'session-b': _sandbox('b'),
    }
    validate_session_key = AsyncMock(side_effect=lambda key, **_kwargs: sandboxes[key])
    validate_teardown_session_key = AsyncMock(
        side_effect=lambda key: sandboxes[key]
    )
    get_store = AsyncMock(return_value=store)
    monkeypatch.setattr(codex_auth, 'validate_session_key', validate_session_key)
    monkeypatch.setattr(
        codex_auth,
        'validate_teardown_session_key',
        validate_teardown_session_key,
    )
    monkeypatch.setattr(codex_auth, '_get_store', get_store)
    return store, get_store


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
        'X-OH-Sandbox': f'session-{sandbox_id}',
        'X-OH-Codex': _token(jwt_service, conversation_id, sandbox_id),
    }


def _refresh_headers(jwt_service, conversation_id, sandbox_id):
    credentials = (
        f'session-{sandbox_id}:{_token(jwt_service, conversation_id, sandbox_id)}'
    )
    encoded = base64.b64encode(credentials.encode()).decode()
    return {'Authorization': f'Basic {encoded}'}


def test_rotation_is_returned_to_next_conversation(app, jwt_service, broker):
    original = '{"tokens":{"refresh_token":"r0"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    store, _ = broker
    store.value = original
    client = TestClient(app)
    first_headers = _headers(jwt_service, _FIRST_ID, 'a')
    response = client.get(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=first_headers,
    )
    assert response.status_code == 200
    assert response.text == original

    response = client.head(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=first_headers,
    )
    assert response.status_code == 204

    response = client.put(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=first_headers,
        json={
            'expected_digest': hashlib.sha256(original.encode()).hexdigest(),
            'value': rotated,
        },
    )
    assert response.status_code == 204
    assert store.value == rotated

    response = client.delete(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=first_headers,
    )
    assert response.status_code == 204
    assert (
        client.get(
            f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
            headers=first_headers,
        ).status_code
        == 401
    )
    assert (
        client.delete(
            f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
            headers=first_headers,
        ).status_code
        == 204
    )

    response = client.get(
        f'/api/internal/conversations/{_SECOND_ID}/codex-auth',
        headers=_headers(jwt_service, _SECOND_ID, 'b'),
    )
    assert response.status_code == 200
    assert response.text == rotated


def test_two_active_conversations_can_read_credentials(app, jwt_service, broker):
    store, _ = broker
    client = TestClient(app)
    first = client.get(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=_headers(jwt_service, _FIRST_ID, 'a'),
    )
    second = client.get(
        f'/api/internal/conversations/{_SECOND_ID}/codex-auth',
        headers=_headers(jwt_service, _SECOND_ID, 'b'),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.text == store.value


def test_revocation_rejects_equivalent_padded_jwt(app, jwt_service, broker):
    client = TestClient(app)
    headers = _headers(jwt_service, _FIRST_ID, 'a')
    response = client.delete(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=headers,
    )
    padded_headers = {**headers, 'X-OH-Codex': f'{headers["X-OH-Codex"]}='}
    padded_response = client.get(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=padded_headers,
    )

    assert response.status_code == 204
    assert padded_response.status_code == 401


@pytest.mark.asyncio
async def test_stale_session_receives_brokered_refresh(
    app, jwt_service, broker, fake_redis, monkeypatch
):
    original = (
        '{"auth_mode":"chatgpt","tokens":'
        '{"id_token":"id-r0","access_token":"access-r0",'
        '"refresh_token":"refresh-r0"}}'
    )
    store, _ = broker
    store.value = original

    async def refresh_upstream(_refresh_token):
        await asyncio.sleep(0.05)
        return codex_auth.httpx.Response(
            200,
            json={
                'id_token': 'id-r1',
                'access_token': 'access-r1',
                'refresh_token': 'refresh-r1',
            },
        )

    upstream = AsyncMock(side_effect=refresh_upstream)
    monkeypatch.setattr(codex_auth, '_request_token_refresh', upstream)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url='http://testserver'
    ) as client:
        first, second = await asyncio.gather(
            client.post(
                f'/api/internal/conversations/{_FIRST_ID}/codex-auth/refresh',
                headers=_refresh_headers(jwt_service, _FIRST_ID, 'a'),
                json={
                    'client_id': codex_auth._REFRESH_CLIENT_ID,
                    'grant_type': 'refresh_token',
                    'refresh_token': 'refresh-r0',
                },
            ),
            client.post(
                f'/api/internal/conversations/{_SECOND_ID}/codex-auth/refresh',
                headers=_refresh_headers(jwt_service, _SECOND_ID, 'b'),
                json={
                    'client_id': codex_auth._REFRESH_CLIENT_ID,
                    'grant_type': 'refresh_token',
                    'refresh_token': 'refresh-r0',
                },
            ),
        )

    assert first.status_code == 200
    assert first.json()['refresh_token'] == 'refresh-r1'
    assert second.status_code == 200
    assert second.json() == first.json()
    assert 'refresh-r1' in store.value
    upstream.assert_awaited_once_with('refresh-r0')
    assert fake_redis.data == {}


def test_head_returns_authoritative_digest(app, jwt_service, broker):
    value = '{"tokens":{"refresh_token":"r0"}}'
    store, _ = broker
    store.value = value
    response = TestClient(app).head(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=_headers(jwt_service, _FIRST_ID, 'a'),
    )

    assert response.status_code == 204
    assert (
        response.headers['X-Codex-Auth-Digest']
        == hashlib.sha256(value.encode()).hexdigest()
    )


def test_digest_conflict_never_overwrites_current_value(app, jwt_service, broker):
    original = '{"tokens":{"refresh_token":"r0"}}'
    current = '{"tokens":{"refresh_token":"newer"}}'
    stale_update = '{"tokens":{"refresh_token":"stale"}}'
    store, _ = broker
    store.value = original
    client = TestClient(app)
    headers = _headers(jwt_service, _FIRST_ID, 'a')
    assert (
        client.get(
            f'/api/internal/conversations/{_FIRST_ID}/codex-auth', headers=headers
        ).status_code
        == 200
    )
    store.value = current
    response = client.put(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=headers,
        json={
            'expected_digest': hashlib.sha256(original.encode()).hexdigest(),
            'value': stale_update,
        },
    )

    assert response.status_code == 409
    assert store.value == current


def test_update_uses_database_cas_without_redis_lock(
    app, jwt_service, broker, fake_redis
):
    store, _ = broker
    original = store.value
    response = TestClient(app).put(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=_headers(jwt_service, _FIRST_ID, 'a'),
        json={
            'expected_digest': hashlib.sha256(original.encode()).hexdigest(),
            'value': '{"tokens":{"refresh_token":"updated"}}',
        },
    )

    assert response.status_code == 204
    assert fake_redis.lock_calls == []


def test_teardown_routes_use_scoped_teardown_key(app, jwt_service, broker):
    codex_auth.validate_session_key.side_effect = HTTPException(401)
    client = TestClient(app)
    headers = _headers(jwt_service, _FIRST_ID, 'a')
    path = f'/api/internal/conversations/{_FIRST_ID}/codex-auth'
    responses = [
        client.get(path, headers=headers),
        client.head(path, headers=headers),
        client.put(
            path,
            headers=headers,
            json={
                'expected_digest': hashlib.sha256(
                    broker[0].value.encode()
                ).hexdigest(),
                'value': broker[0].value,
            },
        ),
        client.delete(path, headers=headers),
    ]

    assert [response.status_code for response in responses] == [200, 204, 204, 204]
    assert codex_auth.validate_teardown_session_key.await_count == 4


def test_refresh_does_not_accept_teardown_key(app, jwt_service, broker):
    codex_auth.validate_session_key.side_effect = HTTPException(401)
    response = TestClient(app).post(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth/refresh',
        headers=_refresh_headers(jwt_service, _FIRST_ID, 'a'),
        json={
            'client_id': codex_auth._REFRESH_CLIENT_ID,
            'grant_type': 'refresh_token',
            'refresh_token': 'r0',
        },
    )

    assert response.status_code == 401
    codex_auth.validate_teardown_session_key.assert_not_awaited()


def test_invalid_document_does_not_echo_credentials(app, jwt_service, broker):
    leaked_value = '{"tokens":{"access_token":"never-echo-this"}}'
    store, _ = broker
    client = TestClient(app)
    headers = _headers(jwt_service, _FIRST_ID, 'a')
    response = client.put(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=headers,
        json={'expected_digest': '0' * 64, 'value': leaked_value},
    )

    assert response.status_code == 400
    assert 'never-echo-this' not in response.text
    assert store.update_calls == 0


def test_conversation_scope_mismatch_cannot_read_credentials(app, jwt_service, broker):
    store, get_store = broker
    store.value = '{"tokens":{"refresh_token":"never-return-this"}}'
    response = TestClient(app).get(
        f'/api/internal/conversations/{_FIRST_ID}/codex-auth',
        headers=_headers(jwt_service, _SECOND_ID, 'a'),
    )

    assert response.status_code == 403
    assert 'never-return-this' not in response.text
    get_store.assert_not_awaited()
