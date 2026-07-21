from types import MappingProxyType

import pytest

from openhands.app_server.file_store.local import LocalFileStore
from openhands.app_server.integrations.provider import CustomSecret
from openhands.app_server.secrets.file_secrets_store import FileSecretsStore
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict

_ORIGINAL = '{"tokens":{"refresh_token":"r0"}}'
_ROTATED = '{"tokens":{"refresh_token":"r1"}}'


def _secrets(codex: str | None, other: str | None = None) -> Secrets:
    values = {}
    if codex is not None:
        values['CODEX_AUTH_JSON'] = CustomSecret.from_value(
            {'secret': codex, 'description': 'Codex login'}
        )
    if other is not None:
        values['OTHER'] = CustomSecret.from_value({'secret': other, 'description': ''})
    return Secrets(custom_secrets=MappingProxyType(values))


@pytest.fixture
def store(tmp_path):
    return FileSecretsStore(LocalFileStore(root=str(tmp_path)))


@pytest.mark.asyncio
async def test_load_initializes_persistent_opaque_version(store):
    await store.store(_secrets(_ORIGINAL))

    value, version = await store.load_versioned('CODEX_AUTH_JSON')
    second = FileSecretsStore(store.file_store)

    assert value == _ORIGINAL
    assert version != _ORIGINAL
    assert await second.load_versioned('CODEX_AUTH_JSON') == (_ORIGINAL, version)


@pytest.mark.asyncio
async def test_replace_is_compare_and_swap(store):
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned('CODEX_AUTH_JSON')

    with pytest.raises(CredentialVersionConflict):
        await store.replace_versioned('CODEX_AUTH_JSON', 'stale', _ROTATED)
    successor = await store.replace_versioned('CODEX_AUTH_JSON', version, _ROTATED)

    assert successor != version
    assert await store.load_versioned('CODEX_AUTH_JSON') == (_ROTATED, successor)


@pytest.mark.asyncio
async def test_delete_and_identical_recreate_changes_version(store):
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned('CODEX_AUTH_JSON')

    await store.store(_secrets(None))
    await store.store(_secrets(_ORIGINAL))

    assert (await store.load_versioned('CODEX_AUTH_JSON'))[1] != version


@pytest.mark.asyncio
async def test_stale_whole_save_preserves_rotation_and_other_edits(store):
    await store.store(_secrets(_ORIGINAL, 'old'))
    stale_store = FileSecretsStore(store.file_store)
    stale = await stale_store.load()
    assert stale is not None
    _, version = await store.load_versioned('CODEX_AUTH_JSON')
    successor = await store.replace_versioned('CODEX_AUTH_JSON', version, _ROTATED)

    updated = dict(stale.custom_secrets)
    updated['OTHER'] = CustomSecret.from_value({'secret': 'new', 'description': ''})
    await stale_store.store(
        stale.model_copy(update={'custom_secrets': MappingProxyType(updated)})
    )

    loaded = await store.load()
    assert loaded is not None
    assert (
        loaded.custom_secrets['CODEX_AUTH_JSON'].secret.get_secret_value() == _ROTATED
    )
    assert loaded.custom_secrets['OTHER'].secret.get_secret_value() == 'new'
    assert await store.load_versioned('CODEX_AUTH_JSON') == (_ROTATED, successor)


@pytest.mark.asyncio
async def test_stale_whole_save_preserves_deletion(store):
    await store.store(_secrets(_ORIGINAL, 'old'))
    stale_store = FileSecretsStore(store.file_store)
    deleting_store = FileSecretsStore(store.file_store)
    stale = await stale_store.load()
    deleting = await deleting_store.load()
    assert stale is not None and deleting is not None

    deleted = dict(deleting.custom_secrets)
    deleted.pop('CODEX_AUTH_JSON')
    await deleting_store.store(
        deleting.model_copy(update={'custom_secrets': MappingProxyType(deleted)})
    )
    updated = dict(stale.custom_secrets)
    updated['OTHER'] = CustomSecret.from_value({'secret': 'new', 'description': ''})
    await stale_store.store(
        stale.model_copy(update={'custom_secrets': MappingProxyType(updated)})
    )

    loaded = await store.load()
    assert loaded is not None
    assert 'CODEX_AUTH_JSON' not in loaded.custom_secrets
    assert loaded.custom_secrets['OTHER'].secret.get_secret_value() == 'new'
