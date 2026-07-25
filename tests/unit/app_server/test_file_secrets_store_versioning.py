import asyncio
import json
from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from openhands.app_server.file_store import local as local_file_store_module
from openhands.app_server.file_store.files import FileStore
from openhands.app_server.file_store.local import LocalFileStore
from openhands.app_server.file_store.memory import InMemoryFileStore
from openhands.app_server.integrations.provider import CustomSecret
from openhands.app_server.secrets.credential_binding_models import (
    CODEX_AUTH_SECRET_NAME,
)
from openhands.app_server.secrets.file_secrets_store import (
    FileSecretsStore,
    Loaded,
    Unloaded,
    _managed_save_decision,
    _ManagedSaveDecision,
)
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict

_ORIGINAL = '{"tokens":{"refresh_token":"r0"}}'
_ROTATED = '{"tokens":{"refresh_token":"r1"}}'
_MANAGED_NAME = CODEX_AUTH_SECRET_NAME


def _store(file_store: FileStore) -> FileSecretsStore:
    return FileSecretsStore(file_store)


class _ReadOnlyMemoryFileStore(InMemoryFileStore):
    def write(self, path: str, contents: str | bytes) -> None:
        raise PermissionError(path)


def _secrets(managed: str | None, other: str | None = None) -> Secrets:
    values = {}
    if managed is not None:
        values[_MANAGED_NAME] = CustomSecret.from_value(
            {'secret': managed, 'description': 'Managed login'}
        )
    if other is not None:
        values['OTHER'] = CustomSecret.from_value({'secret': other, 'description': ''})
    return Secrets(custom_secrets=MappingProxyType(values))


@pytest.mark.parametrize(
    ('baseline', 'submitted', 'expected'),
    [
        (Unloaded(), None, _ManagedSaveDecision.PRESERVE),
        (Unloaded(), _ORIGINAL, _ManagedSaveDecision.EDIT),
        (Loaded(_ORIGINAL, 'v0'), _ORIGINAL, _ManagedSaveDecision.PRESERVE),
        (Loaded(_ORIGINAL, 'v0'), _ROTATED, _ManagedSaveDecision.EDIT),
        (Loaded(_ORIGINAL, 'v0'), None, _ManagedSaveDecision.DELETE),
    ],
)
def test_managed_save_decision(baseline, submitted, expected):
    assert _managed_save_decision(baseline, submitted) == expected


@pytest.fixture
def store(tmp_path):
    return _store(LocalFileStore(root=str(tmp_path)))


@pytest.mark.asyncio
async def test_store_initializes_persistent_opaque_version(store):
    await store.store(_secrets(_ORIGINAL))

    value, version = await store.load_versioned(_MANAGED_NAME)
    second = _store(store.file_store)

    assert value == _ORIGINAL
    assert version != _ORIGINAL
    assert await second.load_versioned(_MANAGED_NAME) == (_ORIGINAL, version)


@pytest.mark.asyncio
async def test_load_versioned_bootstraps_missing_version():
    raw = {'custom_secrets': {_MANAGED_NAME: {'secret': _ORIGINAL, 'description': ''}}}
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = _store(file_store)

    value, version = await store.load_versioned(_MANAGED_NAME)

    assert value == _ORIGINAL
    assert json.loads(file_store.read('secrets.json'))['_credential_versions'] == {
        _MANAGED_NAME: version
    }


@pytest.mark.asyncio
async def test_load_versioned_does_not_open_lock_file(store, monkeypatch):
    await store.store(_secrets(_ORIGINAL))

    def fail_open(*args, **kwargs):
        raise AssertionError('load attempted to open a write lock')

    monkeypatch.setattr(local_file_store_module.os, 'open', fail_open)

    assert (await store.load_versioned(_MANAGED_NAME))[0] == _ORIGINAL


@pytest.mark.asyncio
async def test_bootstrap_merges_version_into_raw_document():
    raw = {
        'provider_tokens': {
            'github': {'token': '', 'host': None, 'user_id': None},
            'future': None,
        },
        'custom_secrets': {
            _MANAGED_NAME: {
                'secret': _ORIGINAL,
                'description': '',
                'future': {'enabled': True},
            }
        },
        'future_top_level': {'value': 1},
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = _store(file_store)

    await store.load_versioned(_MANAGED_NAME)
    updated = json.loads(file_store.read('secrets.json'))
    version = updated.pop('_credential_versions')[_MANAGED_NAME]

    assert updated == raw
    assert await store.load_versioned(_MANAGED_NAME) == (_ORIGINAL, version)


@pytest.mark.asyncio
async def test_bootstrap_is_stable_across_concurrent_stores(tmp_path):
    file_store = LocalFileStore(root=str(tmp_path))
    file_store.write(
        'secrets.json',
        json.dumps(
            {
                'custom_secrets': {
                    _MANAGED_NAME: {'secret': _ORIGINAL, 'description': ''}
                }
            }
        ),
    )
    stores = [_store(file_store) for _ in range(8)]

    await asyncio.gather(*(store.load_versioned(_MANAGED_NAME) for store in stores))
    versions = {(await store.load_versioned(_MANAGED_NAME))[1] for store in stores}

    assert len(versions) == 1


@pytest.mark.asyncio
async def test_bootstrap_failure_leaves_raw_document_unchanged():
    raw = {'custom_secrets': {_MANAGED_NAME: {'secret': _ORIGINAL, 'description': ''}}}
    file_store = _ReadOnlyMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = _store(file_store)

    with pytest.raises(PermissionError):
        await store.load_versioned(_MANAGED_NAME)

    assert json.loads(file_store.read('secrets.json')) == raw


@pytest.mark.asyncio
async def test_replace_is_compare_and_swap(store):
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned(_MANAGED_NAME)

    with pytest.raises(CredentialVersionConflict):
        await store.replace_versioned(_MANAGED_NAME, 'stale', _ROTATED)
    successor = await store.replace_versioned(_MANAGED_NAME, version, _ROTATED)

    assert successor != version
    second_runtime = _store(store.file_store)
    assert await second_runtime.load_versioned(_MANAGED_NAME) == (
        _ROTATED,
        successor,
    )


@pytest.mark.asyncio
async def test_concurrent_replacements_have_one_winner(tmp_path):
    file_store = LocalFileStore(root=str(tmp_path))
    store = _store(file_store)
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned(_MANAGED_NAME)
    replacements = [f'{{"tokens":{{"refresh_token":"r{i}"}}}}' for i in range(8)]

    results = await asyncio.gather(
        *(
            _store(file_store).replace_versioned(_MANAGED_NAME, version, value)
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
            _MANAGED_NAME: {
                'secret': _ORIGINAL,
                'description': '',
                'future': True,
            }
        },
        'future_top_level': ['preserve'],
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = _store(file_store)
    _, version = await store.load_versioned(_MANAGED_NAME)

    await store.replace_versioned(_MANAGED_NAME, version, _ROTATED)
    updated = json.loads(file_store.read('secrets.json'))

    assert updated['provider_tokens'] == raw['provider_tokens']
    assert updated['future_top_level'] == raw['future_top_level']
    assert updated['custom_secrets'][_MANAGED_NAME] == {
        **raw['custom_secrets'][_MANAGED_NAME],
        'secret': _ROTATED,
    }


@pytest.mark.asyncio
async def test_delete_and_identical_recreate_changes_version(store):
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned(_MANAGED_NAME)

    await store.store(_secrets(None))
    await store.store(_secrets(_ORIGINAL))

    assert (await store.load_versioned(_MANAGED_NAME))[1] != version


@pytest.mark.asyncio
async def test_stale_whole_save_preserves_rotation_and_other_edits(store):
    await store.store(_secrets(_ORIGINAL, 'old'))
    stale_store = _store(store.file_store)
    stale = await stale_store.load()
    assert stale is not None
    _, version = await store.load_versioned(_MANAGED_NAME)
    successor = await store.replace_versioned(_MANAGED_NAME, version, _ROTATED)

    updated = dict(stale.custom_secrets)
    updated['OTHER'] = CustomSecret.from_value({'secret': 'new', 'description': ''})
    stale_update = stale.model_copy(
        update={'custom_secrets': MappingProxyType(updated)}
    )
    await stale_store.store(stale_update)
    await stale_store.store(stale_update)

    loaded = await store.load()
    assert loaded is not None
    assert loaded.custom_secrets[_MANAGED_NAME].secret.get_secret_value() == _ROTATED
    assert loaded.custom_secrets['OTHER'].secret.get_secret_value() == 'new'
    assert await store.load_versioned(_MANAGED_NAME) == (_ROTATED, successor)


@pytest.mark.asyncio
async def test_versioned_replace_does_not_rebase_whole_save(store):
    await store.store(_secrets(_ORIGINAL))
    stale = await store.load()
    assert stale is not None
    _, version = await store.load_versioned(_MANAGED_NAME)

    successor = await store.replace_versioned(
        _MANAGED_NAME,
        version,
        _ROTATED,
    )
    await store.store(stale)

    assert await store.load_versioned(_MANAGED_NAME) == (_ROTATED, successor)


@pytest.mark.asyncio
async def test_stale_whole_save_preserves_deletion(store):
    await store.store(_secrets(_ORIGINAL, 'old'))
    stale_store = _store(store.file_store)
    deleting_store = _store(store.file_store)
    stale = await stale_store.load()
    deleting = await deleting_store.load()
    assert stale is not None and deleting is not None

    deleted = dict(deleting.custom_secrets)
    deleted.pop(_MANAGED_NAME)
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
    assert _MANAGED_NAME not in loaded.custom_secrets
    assert loaded.custom_secrets['OTHER'].secret.get_secret_value() == 'new'


@pytest.mark.asyncio
async def test_versioned_bindings_reject_store_without_cross_process_lock():
    file_store = MagicMock(spec=FileStore)
    file_store.supports_locked_update = False
    file_store.read.side_effect = FileNotFoundError
    store = _store(file_store)

    with pytest.raises(NotImplementedError):
        await store.load_versioned(_MANAGED_NAME)
    with pytest.raises(NotImplementedError):
        await store.replace_versioned(_MANAGED_NAME, 'version', _ROTATED)

    file_store.read.assert_not_called()


@pytest.mark.asyncio
async def test_remote_file_store_keeps_unversioned_load_and_store():
    original = _secrets(_ORIGINAL)
    file_store = MagicMock(spec=FileStore)
    file_store.supports_locked_update = False
    file_store.read.return_value = original.model_dump_json(
        context={'expose_secrets': True}
    )
    store = _store(file_store)

    loaded = await store.load()

    assert loaded == original
    assert store._managed_baselines == {}

    updated = _secrets(_ROTATED, 'other')
    file_store.reset_mock()
    await store.store(updated)

    file_store.read.assert_not_called()
    file_store.write.assert_called_once_with(
        'secrets.json',
        updated.model_dump_json(context={'expose_secrets': True}),
    )


@pytest.mark.asyncio
async def test_whole_save_preserves_unrecognized_top_level_data():
    raw = {
        'custom_secrets': {},
        'future_top_level': {'enabled': True},
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = _store(file_store)

    await store.store(_secrets(None, 'new'))

    assert json.loads(file_store.read('secrets.json'))['future_top_level'] == {
        'enabled': True
    }


@pytest.mark.asyncio
async def test_whole_save_preserves_unrecognized_version_metadata():
    raw = {
        'custom_secrets': {_MANAGED_NAME: {'secret': _ORIGINAL, 'description': ''}},
        '_credential_versions': {
            _MANAGED_NAME: 'v0',
            'future': {'scheme': 'v2'},
        },
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = _store(file_store)
    loaded = await store.load()
    assert loaded is not None

    await store.store(loaded)

    assert (
        json.loads(file_store.read('secrets.json'))['_credential_versions']
        == raw['_credential_versions']
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'raw',
    [
        '{',
        json.dumps([]),
        json.dumps({'provider_tokens': []}),
        json.dumps({'custom_secrets': []}),
    ],
)
async def test_whole_save_rejects_invalid_file(raw):
    file_store = InMemoryFileStore(files={'secrets.json': raw})
    store = _store(file_store)

    with pytest.raises(ValueError):
        await store.store(_secrets(_ORIGINAL, 'new'))

    assert file_store.read('secrets.json') == raw
