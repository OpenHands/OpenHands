import hashlib
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import SecretStr
from storage.codex_auth_store import CodexAuthStore
from storage.stored_custom_secrets import StoredCustomSecrets

from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey

_ORG_ID = UUID('a1111111-1111-1111-1111-111111111111')
_OTHER_ORG_ID = UUID('b2222222-2222-2222-2222-222222222222')


@pytest.fixture
def jwt_svc():
    key = EncryptionKey(kid='test', key=SecretStr('test_secret'), active=True)
    return JwtService(keys=[key])


@pytest.fixture
def store(async_session_maker, jwt_svc):
    with patch('storage.codex_auth_store.a_session_maker', async_session_maker):
        yield CodexAuthStore('user-id', _ORG_ID, jwt_svc)


async def _insert(
    async_session_maker, jwt_svc, value: str, org_id: UUID = _ORG_ID
) -> None:
    async with async_session_maker() as session:
        session.add(
            StoredCustomSecrets(
                keycloak_user_id='user-id',
                org_id=org_id,
                secret_name='CODEX_AUTH_JSON',
                secret_value=jwt_svc.encrypt_value(value),
                description=None,
            )
        )
        await session.commit()


@pytest.mark.asyncio
# SQLite does not serialize SELECT FOR UPDATE.
async def test_compare_and_swap(async_session_maker, jwt_svc, store):
    original = '{"tokens":{"refresh_token":"r0"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    await _insert(async_session_maker, jwt_svc, original)

    assert await store.get_value() == original
    assert not await store.compare_and_swap('0' * 64, rotated)
    assert await store.get_value() == original
    assert await store.compare_and_swap(
        hashlib.sha256(original.encode()).hexdigest(), rotated
    )
    assert await store.get_value() == rotated


@pytest.mark.asyncio
async def test_compare_and_swap_is_scoped_to_org(async_session_maker, jwt_svc, store):
    original = '{"tokens":{"refresh_token":"r0"}}'
    other = '{"tokens":{"refresh_token":"other"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    await _insert(async_session_maker, jwt_svc, original)
    await _insert(async_session_maker, jwt_svc, other, _OTHER_ORG_ID)

    assert await store.compare_and_swap(
        hashlib.sha256(original.encode()).hexdigest(), rotated
    )
    other_store = CodexAuthStore('user-id', _OTHER_ORG_ID, jwt_svc)
    assert await other_store.get_value() == other


@pytest.mark.asyncio
async def test_compare_and_swap_requires_existing_auth(store):
    with pytest.raises(KeyError, match='CODEX_AUTH_JSON'):
        await store.compare_and_swap('0' * 64, '{}')
