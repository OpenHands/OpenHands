import asyncio
import json
import multiprocessing
from queue import Empty
from unittest.mock import MagicMock

import pytest

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.file_store.local import LocalFileStore
from openhands.app_server.file_store.memory import InMemoryFileStore
from openhands.app_server.integrations.provider import CustomSecret
from openhands.app_server.secrets.file_secrets_store import FileSecretsStore
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict

_NAME = 'CODEX_AUTH_JSON'
_ORIGINAL = '{"tokens":{"refresh_token":"r0"}}'
_ROTATED = '{"tokens":{"refresh_token":"r1"}}'


def _secrets(
    managed: str | None,
    other: str | None = None,
    description: str = 'Managed login',
) -> Secrets:
    values = {}
    if managed is not None:
        values[_NAME] = CustomSecret.from_value(
            {'secret': managed, 'description': description}
        )
    if other is not None:
        values['OTHER'] = CustomSecret.from_value({'secret': other, 'description': ''})
    return Secrets(custom_secrets=values)


def _replace_in_process(
    root: str,
    expected_version: str,
    value: str,
    results,
) -> None:
    async def replace() -> str:
        store = FileSecretsStore(LocalFileStore(root=root))
        return await store.replace_versioned(_NAME, expected_version, value)

    try:
        results.put(('ok', asyncio.run(replace())))
    except Exception as exc:
        results.put((type(exc).__name__, str(exc)))


@pytest.fixture
def store(tmp_path):
    return FileSecretsStore(LocalFileStore(root=str(tmp_path)))


@pytest.mark.asyncio
async def test_store_creates_persistent_opaque_generation(store):
    await store.store(_secrets(_ORIGINAL))

    value, version = await store.load_versioned(_NAME)
    second = FileSecretsStore(store.file_store)

    assert value == _ORIGINAL
    assert version != _ORIGINAL
    assert await second.load_versioned(_NAME) == (_ORIGINAL, version)


@pytest.mark.asyncio
async def test_load_versioned_bootstraps_without_rewriting_raw_fields():
    raw = {
        'custom_secrets': {
            _NAME: {
                'secret': _ORIGINAL,
                'description': 'Managed login',
                'future': {'enabled': True},
            }
        },
        'future_top_level': ['keep'],
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = FileSecretsStore(file_store)

    value, version = await store.load_versioned(_NAME)
    updated = json.loads(file_store.read('secrets.json'))
    versions = updated.pop('_credential_versions')

    assert (value, versions[_NAME]) == (_ORIGINAL, version)
    assert updated == raw


@pytest.mark.asyncio
async def test_replace_is_compare_and_swap_and_preserves_raw_fields():
    raw = {
        'custom_secrets': {
            _NAME: {
                'secret': _ORIGINAL,
                'description': 'Managed login',
                'future': True,
            }
        },
        'future_top_level': {'keep': True},
    }
    file_store = InMemoryFileStore(files={'secrets.json': json.dumps(raw)})
    store = FileSecretsStore(file_store)
    _, version = await store.load_versioned(_NAME)

    with pytest.raises(CredentialVersionConflict):
        await store.replace_versioned(_NAME, 'stale', _ROTATED)
    successor = await store.replace_versioned(_NAME, version, _ROTATED)
    updated = json.loads(file_store.read('secrets.json'))

    assert successor != version
    assert updated['future_top_level'] == raw['future_top_level']
    assert updated['custom_secrets'][_NAME] == {
        **raw['custom_secrets'][_NAME],
        'secret': _ROTATED,
    }


@pytest.mark.asyncio
async def test_delete_and_identical_recreate_rejects_old_generation(store):
    await store.store(_secrets(_ORIGINAL))
    _, original_version = await store.load_versioned(_NAME)

    await store.store(_secrets(None))
    await store.store(_secrets(_ORIGINAL))
    _, recreated_version = await store.load_versioned(_NAME)

    assert recreated_version != original_version
    with pytest.raises(CredentialVersionConflict):
        await store.replace_versioned(_NAME, original_version, _ROTATED)


@pytest.mark.asyncio
async def test_stale_whole_save_preserves_rotation_and_unrelated_edits(store):
    await store.store(_secrets(_ORIGINAL, 'old'))
    raw = json.loads(store.file_store.read('secrets.json'))
    raw['custom_secrets'][_NAME]['future'] = 'keep'
    raw['future_top_level'] = {'keep': True}
    store.file_store.write('secrets.json', json.dumps(raw))

    stale_store = FileSecretsStore(store.file_store)
    stale = await stale_store.load()
    assert stale is not None
    _, version = await store.load_versioned(_NAME)
    successor = await store.replace_versioned(_NAME, version, _ROTATED)

    updated = dict(stale.custom_secrets)
    updated[_NAME] = CustomSecret.from_value(
        {'secret': _ORIGINAL, 'description': 'Updated description'}
    )
    updated['OTHER'] = CustomSecret.from_value({'secret': 'new', 'description': ''})
    await stale_store.store(stale.model_copy(update={'custom_secrets': updated}))

    loaded = await store.load()
    saved_raw = json.loads(store.file_store.read('secrets.json'))
    assert loaded is not None
    assert loaded.custom_secrets[_NAME].secret.get_secret_value() == _ROTATED
    assert loaded.custom_secrets[_NAME].description == 'Updated description'
    assert loaded.custom_secrets['OTHER'].secret.get_secret_value() == 'new'
    assert await store.load_versioned(_NAME) == (_ROTATED, successor)
    assert saved_raw['custom_secrets'][_NAME]['future'] == 'keep'
    assert saved_raw['future_top_level'] == {'keep': True}


@pytest.mark.asyncio
async def test_stale_whole_save_preserves_concurrent_deletion(store):
    await store.store(_secrets(_ORIGINAL, 'old'))
    stale_store = FileSecretsStore(store.file_store)
    deleting_store = FileSecretsStore(store.file_store)
    stale = await stale_store.load()
    deleting = await deleting_store.load()
    assert stale is not None and deleting is not None

    deleted = dict(deleting.custom_secrets)
    deleted.pop(_NAME)
    await deleting_store.store(deleting.model_copy(update={'custom_secrets': deleted}))

    updated = dict(stale.custom_secrets)
    updated['OTHER'] = CustomSecret.from_value({'secret': 'new', 'description': ''})
    await stale_store.store(stale.model_copy(update={'custom_secrets': updated}))

    loaded = await store.load()
    assert loaded is not None
    assert _NAME not in loaded.custom_secrets
    assert loaded.custom_secrets['OTHER'].secret.get_secret_value() == 'new'


@pytest.mark.asyncio
async def test_explicit_edit_after_rotation_mints_generation(store):
    await store.store(_secrets(_ORIGINAL))
    editing_store = FileSecretsStore(store.file_store)
    editing = await editing_store.load()
    assert editing is not None
    _, version = await store.load_versioned(_NAME)
    rotated_version = await store.replace_versioned(_NAME, version, _ROTATED)

    changed = _secrets('{"tokens":{"refresh_token":"r2"}}')
    await editing_store.store(changed)
    value, changed_version = await store.load_versioned(_NAME)

    assert value == '{"tokens":{"refresh_token":"r2"}}'
    assert changed_version != rotated_version


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['local', 'memory'])
async def test_concurrent_replacements_have_one_winner(kind, tmp_path):
    file_store: FileStore
    if kind == 'local':
        file_store = LocalFileStore(root=str(tmp_path))
    else:
        file_store = InMemoryFileStore()
    store = FileSecretsStore(file_store)
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned(_NAME)

    results = await asyncio.gather(
        *(
            FileSecretsStore(file_store).replace_versioned(
                _NAME,
                version,
                f'{{"tokens":{{"refresh_token":"r{i}"}}}}',
            )
            for i in range(8)
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, CredentialVersionConflict) for result in results) == 7


@pytest.mark.asyncio
async def test_local_compare_and_swap_is_cross_process(store, tmp_path):
    await store.store(_secrets(_ORIGINAL))
    _, version = await store.load_versioned(_NAME)
    context = multiprocessing.get_context('spawn')
    results = context.Queue()
    processes = [
        context.Process(
            target=_replace_in_process,
            args=(str(tmp_path), version, f'value-{index}', results),
        )
        for index in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    received = []
    for _ in processes:
        try:
            received.append(results.get(timeout=5))
        except Empty:
            pytest.fail('A replacement process returned no result')

    assert sum(kind == 'ok' for kind, _ in received) == 1
    assert (
        sum(kind == 'CredentialVersionConflict' for kind, _ in received)
        == len(processes) - 1
    )


@pytest.mark.asyncio
async def test_unsupported_file_store_retains_legacy_path():
    file_store = MagicMock(spec=FileStore)
    file_store.supports_locked_update = False
    file_store.read.return_value = _secrets(_ORIGINAL).model_dump_json(
        context={'expose_secrets': True}
    )
    store = FileSecretsStore(file_store)

    assert await store.load() == _secrets(_ORIGINAL)
    with pytest.raises(NotImplementedError):
        await store.load_versioned(_NAME)
    with pytest.raises(NotImplementedError):
        await store.replace_versioned(_NAME, 'version', _ROTATED)

    file_store.reset_mock()
    updated = _secrets(_ROTATED, 'new')
    await store.store(updated)
    file_store.write.assert_called_once_with(
        'secrets.json',
        updated.model_dump_json(context={'expose_secrets': True}),
    )
