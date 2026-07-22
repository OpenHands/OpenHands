import asyncio
import json
from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.file_store.local import LocalFileStore
from openhands.app_server.file_store.memory import InMemoryFileStore
from openhands.app_server.integrations.provider import CustomSecret
from openhands.app_server.secrets import file_secrets_store as file_secrets_store_module
from openhands.app_server.secrets.file_secrets_store import FileSecretsStore
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict

_ORIGINAL = '{"tokens":{"refresh_token":"r0"}}'
_ROTATED = '{"tokens":{"refresh_token":"r1"}}'


class _ReadOnlyMemoryFileStore(InMemoryFileStore):
    def write(self, path: str, contents: str | bytes) -> None:
        raise PermissionError(path)


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
async def test_store_initializes_persistent_opaque_version(store):
    await store.store(_secrets(_ORIGINAL))

    value, version = await store.load_versioned('CODEX_AUTH_JSON')
    second = FileSecretsStore(store.file_store)

    assert value == _ORIGINAL
    assert version != _ORIGINAL
    assert await second.load_versioned('CODEX_AUTH_JSON') == (_ORIGINAL, version)


@pytest.mark.asyncio
async def test_load_versioned_never_writes_missing_version():
    raw = {
        'custom_secrets': {'CODEX_AUTH_JSON': {'secret': _ORIGINAL, 'description': ''}}
    }
    file_store = _ReadOnlyMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = FileSecretsStore(file_store)

    with pytest.raises(NotImplementedError):
        await store.load_versioned('CODEX_AUTH_JSON')

    assert json.loads(file_store.read('secrets.json')) == raw


@pytest.mark.asyncio
async def test_load_versioned_does_not_open_lock_file(store, monkeypatch):
    await store.store(_secrets(_ORIGINAL))

    def fail_open(*args, **kwargs):
        raise AssertionError('load attempted to open a write lock')

    monkeypatch.setattr(file_secrets_store_module.os, 'open', fail_open)

    assert (await store.load_versioned('CODEX_AUTH_JSON'))[0] == _ORIGINAL


@pytest.mark.asyncio
async def test_bootstrap_merges_version_into_raw_document():
    raw = {
        'provider_tokens': {
            'github': {'token': '', 'host': None, 'user_id': None},
            'future': None,
        },
        'custom_secrets': {
            'CODEX_AUTH_JSON': {
                'secret': _ORIGINAL,
                'description': '',
                'future': {'enabled': True},
            }
        },
        'future_top_level': {'value': 1},
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = FileSecretsStore(file_store)

    await store.ensure_versioned('CODEX_AUTH_JSON')
    updated = json.loads(file_store.read('secrets.json'))
    version = updated.pop('_credential_versions')['CODEX_AUTH_JSON']

    assert updated == raw
    assert await store.load_versioned('CODEX_AUTH_JSON') == (_ORIGINAL, version)


@pytest.mark.asyncio
async def test_bootstrap_is_stable_across_concurrent_stores(tmp_path):
    file_store = LocalFileStore(root=str(tmp_path))
    file_store.write(
        'secrets.json',
        json.dumps(
            {
                'custom_secrets': {
                    'CODEX_AUTH_JSON': {'secret': _ORIGINAL, 'description': ''}
                }
            }
        ),
    )
    stores = [FileSecretsStore(file_store) for _ in range(8)]

    await asyncio.gather(
        *(store.ensure_versioned('CODEX_AUTH_JSON') for store in stores)
    )
    versions = {(await store.load_versioned('CODEX_AUTH_JSON'))[1] for store in stores}

    assert len(versions) == 1


@pytest.mark.asyncio
async def test_bootstrap_failure_leaves_raw_document_unchanged():
    raw = {
        'custom_secrets': {'CODEX_AUTH_JSON': {'secret': _ORIGINAL, 'description': ''}}
    }
    file_store = _ReadOnlyMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = FileSecretsStore(file_store)

    with pytest.raises(PermissionError):
        await store.ensure_versioned('CODEX_AUTH_JSON')

    assert json.loads(file_store.read('secrets.json')) == raw


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
async def test_concurrent_replacements_have_one_winner(tmp_path):
    file_store = LocalFileStore(root=str(tmp_path))
    store = FileSecretsStore(file_store)
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned('CODEX_AUTH_JSON')
    replacements = [f'{{"tokens":{{"refresh_token":"r{i}"}}}}' for i in range(8)]

    results = await asyncio.gather(
        *(
            FileSecretsStore(file_store).replace_versioned(
                'CODEX_AUTH_JSON', version, value
            )
            for value in replacements
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, CredentialVersionConflict) for result in results) == 7


@pytest.mark.asyncio
async def test_replace_preserves_unrecognized_file_data():
    raw = {
        'provider_tokens': {'github': {'token': ''}, 'future': None},
        'custom_secrets': {
            'CODEX_AUTH_JSON': {
                'secret': _ORIGINAL,
                'description': '',
                'future': True,
            }
        },
        'future_top_level': ['preserve'],
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = FileSecretsStore(file_store)
    await store.ensure_versioned('CODEX_AUTH_JSON')
    _, version = await store.load_versioned('CODEX_AUTH_JSON')

    await store.replace_versioned('CODEX_AUTH_JSON', version, _ROTATED)
    updated = json.loads(file_store.read('secrets.json'))

    assert updated['provider_tokens'] == raw['provider_tokens']
    assert updated['future_top_level'] == raw['future_top_level']
    assert updated['custom_secrets']['CODEX_AUTH_JSON'] == {
        **raw['custom_secrets']['CODEX_AUTH_JSON'],
        'secret': _ROTATED,
    }


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


@pytest.mark.asyncio
async def test_versioned_bindings_reject_store_without_cross_process_lock():
    file_store = MagicMock(spec=FileStore)
    file_store.read.side_effect = FileNotFoundError
    store = FileSecretsStore(file_store)

    with pytest.raises(NotImplementedError):
        await store.load_versioned('CODEX_AUTH_JSON')
    with pytest.raises(NotImplementedError):
        await store.replace_versioned('CODEX_AUTH_JSON', 'version', _ROTATED)

    file_store.read.assert_not_called()
