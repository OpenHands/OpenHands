from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import SecretStr
from sqlalchemy import delete, select
from storage.stored_custom_secrets import StoredCustomSecrets
from storage.versioned_credential_store import SaasVersionedCredentialStore

from openhands.app_server.secrets.secrets_store import CredentialVersionConflict
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
    with patch(
        'storage.versioned_credential_store.a_session_maker', async_session_maker
    ):
        yield SaasVersionedCredentialStore('user-id', _ORG_ID, jwt_svc)


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
async def test_compare_and_swap(async_session_maker, jwt_svc, store):
    original = '{"tokens":{"refresh_token":"r0"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    await _insert(async_session_maker, jwt_svc, original)

    value, version = await store.load('CODEX_AUTH_JSON')
    assert value == original
    with pytest.raises(CredentialVersionConflict):
        await store.replace('CODEX_AUTH_JSON', 'stale', rotated)
    assert (await store.load('CODEX_AUTH_JSON')) == (original, version)
    successor = await store.replace('CODEX_AUTH_JSON', version, rotated)
    assert successor != version
    assert await store.load('CODEX_AUTH_JSON') == (rotated, successor)


@pytest.mark.asyncio
async def test_delete_and_identical_recreate_changes_version(
    async_session_maker, jwt_svc, store
):
    await _insert(async_session_maker, jwt_svc, 'same')
    _, version = await store.load('CODEX_AUTH_JSON')
    async with async_session_maker() as session:
        await session.execute(
            delete(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == 'user-id',
                StoredCustomSecrets.org_id == _ORG_ID,
                StoredCustomSecrets.secret_name == 'CODEX_AUTH_JSON',
            )
        )
        await session.commit()
    await _insert(async_session_maker, jwt_svc, 'same')

    assert (await store.load('CODEX_AUTH_JSON'))[1] != version


@pytest.mark.asyncio
async def test_replace_converges_duplicate_rows(async_session_maker, jwt_svc, store):
    stale = '{"tokens":{"refresh_token":"stale"}}'
    current = '{"tokens":{"refresh_token":"current"}}'
    rotated = '{"tokens":{"refresh_token":"rotated"}}'
    await _insert(async_session_maker, jwt_svc, stale)
    await _insert(async_session_maker, jwt_svc, current)

    value, version = await store.load('CODEX_AUTH_JSON')
    assert value == current
    await store.replace('CODEX_AUTH_JSON', version, rotated)
    async with async_session_maker() as session:
        result = await session.execute(
            select(StoredCustomSecrets).filter(
                StoredCustomSecrets.keycloak_user_id == 'user-id',
                StoredCustomSecrets.org_id == _ORG_ID,
                StoredCustomSecrets.secret_name == 'CODEX_AUTH_JSON',
            )
        )
        values = {
            jwt_svc.decrypt_value(row.secret_value) for row in result.scalars().all()
        }
    assert values == {rotated}


@pytest.mark.asyncio
async def test_replace_is_scoped_to_org(async_session_maker, jwt_svc, store):
    original = '{"tokens":{"refresh_token":"r0"}}'
    other = '{"tokens":{"refresh_token":"other"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    await _insert(async_session_maker, jwt_svc, original)
    await _insert(async_session_maker, jwt_svc, other, _OTHER_ORG_ID)

    _, version = await store.load('CODEX_AUTH_JSON')
    await store.replace('CODEX_AUTH_JSON', version, rotated)
    other_store = SaasVersionedCredentialStore('user-id', _OTHER_ORG_ID, jwt_svc)
    assert (await other_store.load('CODEX_AUTH_JSON'))[0] == other


@pytest.mark.asyncio
async def test_replace_requires_existing_auth(store):
    with pytest.raises(KeyError, match='CODEX_AUTH_JSON'):
        await store.replace('CODEX_AUTH_JSON', 'stale', '{}')
