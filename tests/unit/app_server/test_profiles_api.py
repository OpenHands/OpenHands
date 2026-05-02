"""Integration tests for the LLM profile endpoints in the settings router.

Covers every endpoint under ``/api/v1/settings/profiles``:
- list / get / save / delete / activate — happy paths and 404s.
- 422 on malformed ``llm`` payload (Pydantic validates the request body).
"""

import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.app import app
from openhands.app_server.file_store import get_file_store
from openhands.app_server.integrations.provider import ProviderToken, ProviderType
from openhands.app_server.integrations.service_types import UserGitInfo
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.settings.file_settings_store import FileSettingsStore
from openhands.app_server.settings.llm_profiles import MAX_PROFILES_PER_USER
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_router import _user_profile_locks
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.user_auth.user_auth import UserAuth
from openhands.sdk.llm import LLM
from openhands.sdk.settings import AgentSettings


@pytest.fixture(autouse=True)
def _reset_profile_locks():
    """Locks bind to the event loop that first awaited them; FastAPI TestClient
    spins a fresh loop per test, so any stale Lock carried over from a previous
    test would be attached to a dead loop. Clearing between tests fixes it."""
    _user_profile_locks.clear()
    yield
    _user_profile_locks.clear()


class _MockUserAuth(UserAuth):
    def __init__(self, settings_store: SettingsStore) -> None:
        self._settings = None
        self._settings_store = settings_store

    async def get_user_id(self) -> str | None:
        return 'test-user'

    async def get_user_email(self) -> str | None:
        return 'test-email@example.com'

    async def get_access_token(self) -> SecretStr | None:
        return SecretStr('test-token')

    async def get_provider_tokens(
        self,
    ) -> dict[ProviderType, ProviderToken] | None:
        return None

    async def get_user_settings_store(self) -> SettingsStore | None:
        return self._settings_store

    async def get_secrets_store(self) -> SecretsStore | None:
        return None

    async def get_secrets(self) -> Secrets | None:
        return None

    async def get_mcp_api_key(self) -> str | None:
        return None

    async def get_user_git_info(self) -> UserGitInfo | None:
        return None

    @classmethod
    async def get_instance(cls, request: Request) -> UserAuth:
        raise NotImplementedError  # patched per-test

    @classmethod
    async def get_for_user(cls, user_id: str) -> UserAuth:
        raise NotImplementedError  # patched per-test


@pytest.fixture
def settings_store(tmp_path: Path) -> FileSettingsStore:
    return FileSettingsStore(get_file_store('local', str(tmp_path)))


@pytest.fixture
def test_client(settings_store):
    """TestClient wired to an in-memory settings store the test can seed directly."""
    auth = _MockUserAuth(settings_store)
    with (
        patch.dict(
            os.environ,
            {'SESSION_API_KEY': '', 'ALLOW_SHORT_CONTEXT_WINDOWS': 'true'},
            clear=False,
        ),
        patch('openhands.app_server.utils.dependencies._SESSION_API_KEY', None),
        patch(
            'openhands.app_server.user_auth.user_auth.UserAuth.get_instance',
            return_value=auth,
        ),
        patch(
            'openhands.app_server.settings.file_settings_store.FileSettingsStore.get_instance',
            AsyncMock(return_value=settings_store),
        ),
    ):
        yield TestClient(app)


def _base_settings() -> Settings:
    """A Settings instance with an LLM configured so 'snapshot current' works."""
    return Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='openai/gpt-4o',
                api_key=SecretStr('sk-current'),
            ),
        ),
    )


async def _seed(store: FileSettingsStore, settings: Settings) -> None:
    await store.store(settings)


@contextmanager
def _client_for_user(user_id: str, store: FileSettingsStore):
    """Yield a TestClient scoped to a specific user_id + settings store.

    Used by multi-user isolation tests; the module-level ``test_client``
    fixture is pinned to a single mock user.
    """

    class _Scoped(_MockUserAuth):
        async def get_user_id(self) -> str | None:  # type: ignore[override]
            return user_id

    auth = _Scoped(store)
    with (
        patch.dict(
            os.environ,
            {'SESSION_API_KEY': '', 'ALLOW_SHORT_CONTEXT_WINDOWS': 'true'},
            clear=False,
        ),
        patch('openhands.app_server.utils.dependencies._SESSION_API_KEY', None),
        patch(
            'openhands.app_server.user_auth.user_auth.UserAuth.get_instance',
            return_value=auth,
        ),
        patch(
            'openhands.app_server.settings.file_settings_store.FileSettingsStore.get_instance',
            AsyncMock(return_value=store),
        ),
    ):
        yield TestClient(app)


# ── GET /profiles ────────────────────────────────────────────────


def test_list_profiles_returns_empty_when_no_settings(test_client):
    response = test_client.get('/api/v1/settings/profiles')

    assert response.status_code == 200
    assert response.json() == {'profiles': [], 'active_profile': None}


@pytest.mark.asyncio
async def test_list_profiles_returns_saved_profiles(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save(
        'my-gpt4',
        LLM(model='openai/gpt-4o', api_key=SecretStr('sk-1')),
    )
    settings.llm_profiles.save('my-claude', LLM(model='anthropic/claude-opus-4'))
    settings.llm_profiles.active = 'my-claude'
    await _seed(settings_store, settings)

    response = test_client.get('/api/v1/settings/profiles')

    assert response.status_code == 200
    body = response.json()
    assert body['active_profile'] == 'my-claude'
    names = {p['name'] for p in body['profiles']}
    assert names == {'my-gpt4', 'my-claude'}


# ── GET /profiles/{name} ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_profile_returns_null_api_key_with_set_flag(
    test_client, settings_store
):
    """``api_key`` is never echoed — the sibling ``api_key_set`` flag
    tells the UI whether a key is stored. Prevents the GET→edit→POST
    round-trip from poisoning the stored key with a mask string."""
    settings = _base_settings()
    settings.llm_profiles.save(
        'p', LLM(model='openai/gpt-4o', api_key=SecretStr('sk-secret'))
    )
    await _seed(settings_store, settings)

    response = test_client.get('/api/v1/settings/profiles/p')

    assert response.status_code == 200
    body = response.json()
    assert body['name'] == 'p'
    assert body['config']['model'] == 'openai/gpt-4o'
    assert body['config']['api_key'] is None
    assert body['api_key_set'] is True


def test_get_profile_returns_404_when_unknown(test_client):
    response = test_client.get('/api/v1/settings/profiles/nope')

    assert response.status_code == 404
    assert "'nope'" in response.json()['detail']


# ── POST /profiles/{name} ────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_profile_with_explicit_llm_persists(test_client, settings_store):
    await _seed(settings_store, _base_settings())

    response = test_client.post(
        '/api/v1/settings/profiles/my-new',
        json={
            'llm': {
                'model': 'anthropic/claude-opus-4',
                'api_key': 'sk-new',
            },
        },
    )

    assert response.status_code == 201
    assert response.json()['name'] == 'my-new'

    stored = await settings_store.load()
    assert stored.llm_profiles.has('my-new')
    saved_llm = stored.llm_profiles.get('my-new')
    assert saved_llm.model == 'anthropic/claude-opus-4'
    assert saved_llm.api_key.get_secret_value() == 'sk-new'


@pytest.mark.asyncio
async def test_save_profile_snapshots_current_llm_when_no_body(
    test_client, settings_store
):
    await _seed(settings_store, _base_settings())

    response = test_client.post('/api/v1/settings/profiles/snapshot', json={})

    assert response.status_code == 201
    stored = await settings_store.load()
    snap = stored.llm_profiles.get('snapshot')
    assert snap is not None
    assert snap.model == 'openai/gpt-4o'
    assert snap.api_key.get_secret_value() == 'sk-current'


@pytest.mark.asyncio
async def test_save_profile_without_secrets_clears_api_key(test_client, settings_store):
    await _seed(settings_store, _base_settings())

    response = test_client.post(
        '/api/v1/settings/profiles/no-key',
        json={
            'include_secrets': False,
            'llm': {'model': 'openai/gpt-4o', 'api_key': 'sk-abc'},
        },
    )

    assert response.status_code == 201
    stored = await settings_store.load()
    assert stored.llm_profiles.get('no-key').api_key is None


@pytest.mark.asyncio
async def test_save_profile_overwrites_existing(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save('p', LLM(model='openai/gpt-4o'))
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/p',
        json={'llm': {'model': 'anthropic/claude-opus-4'}},
    )

    assert response.status_code == 201
    stored = await settings_store.load()
    assert stored.llm_profiles.get('p').model == 'anthropic/claude-opus-4'


@pytest.mark.asyncio
async def test_save_overwrite_of_active_profile_clears_active(
    test_client, settings_store
):
    """Overwriting the currently-active profile must drop ``active`` —
    otherwise the UI claims profile X is in use while ``agent_settings.llm``
    still points at the *old* X. Mirrors the safety net the main settings
    POST already enforces via ``reconcile_active_profile``.
    """
    settings = _base_settings()
    settings.llm_profiles.save('p', LLM(model='openai/gpt-4o'))
    settings.switch_to_profile('p')  # makes 'p' active *and* the running llm
    await _seed(settings_store, settings)
    assert test_client.get('/api/v1/settings/profiles').json()['active_profile'] == 'p'

    # Save a different config under the same name.
    response = test_client.post(
        '/api/v1/settings/profiles/p',
        json={'llm': {'model': 'anthropic/claude-opus-4'}},
    )
    assert response.status_code == 201

    body = test_client.get('/api/v1/settings/profiles').json()
    assert body['active_profile'] is None
    # The saved profile reflects the new config; the active marker is gone
    # because agent_settings.llm still runs the previous one.
    stored = await settings_store.load()
    assert stored.llm_profiles.get('p').model == 'anthropic/claude-opus-4'
    assert stored.agent_settings.llm.model == 'openai/gpt-4o'


@pytest.mark.asyncio
async def test_save_overwrite_of_inactive_profile_preserves_active(
    test_client, settings_store
):
    """Overwriting a non-active profile must NOT touch the active marker —
    only the active profile can diverge from agent_settings.llm.
    """
    settings = _base_settings()
    settings.llm_profiles.save('active', LLM(model='openai/gpt-4o'))
    settings.llm_profiles.save('other', LLM(model='openai/gpt-4o'))
    settings.switch_to_profile('active')
    await _seed(settings_store, settings)

    test_client.post(
        '/api/v1/settings/profiles/other',
        json={'llm': {'model': 'anthropic/claude-opus-4'}},
    )

    body = test_client.get('/api/v1/settings/profiles').json()
    assert body['active_profile'] == 'active'


@pytest.mark.asyncio
async def test_save_profile_rejects_invalid_llm_with_422(test_client, settings_store):
    await _seed(settings_store, _base_settings())

    # Missing required `model`; StrictLLM would also reject the unknown key.
    response = test_client.post(
        '/api/v1/settings/profiles/bad',
        json={'llm': {'not_a_real_llm_field': True}},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_edit_profile_round_trip_preserves_api_key(test_client, settings_store):
    """Frontend GET→edit→POST flow must not corrupt the stored key.

    The GET response returns ``api_key: null``; when the frontend echoes
    that back in the POST body, the server has to preserve the stored
    key rather than overwrite with None. This is the concrete
    mask-poisoning regression we fixed.
    """
    await _seed(settings_store, _base_settings())
    test_client.post(
        '/api/v1/settings/profiles/p',
        json={'llm': {'model': 'openai/gpt-4o', 'api_key': 'REAL-KEY-42'}},
    )

    fetched = test_client.get('/api/v1/settings/profiles/p').json()
    fetched['config']['model'] = 'anthropic/claude-opus-4'  # user edits model
    assert fetched['config']['api_key'] is None  # GET returns null, not mask

    resp = test_client.post(
        '/api/v1/settings/profiles/p', json={'llm': fetched['config']}
    )
    assert resp.status_code == 201

    stored = await settings_store.load()
    preserved = stored.llm_profiles.get('p')
    assert preserved.model == 'anthropic/claude-opus-4'
    assert preserved.api_key.get_secret_value() == 'REAL-KEY-42'


@pytest.mark.asyncio
async def test_edit_profile_with_new_api_key_replaces_old(test_client, settings_store):
    """Counter-test to the round-trip guard: when the user actually types
    a new api_key in the edit form, the server must replace the stored
    one. Catches a logic regression where ``preserve`` runs
    unconditionally and swallows intentional updates.
    """
    await _seed(settings_store, _base_settings())
    test_client.post(
        '/api/v1/settings/profiles/p',
        json={'llm': {'model': 'openai/gpt-4o', 'api_key': 'OLD-KEY'}},
    )

    resp = test_client.post(
        '/api/v1/settings/profiles/p',
        json={'llm': {'model': 'openai/gpt-4o', 'api_key': 'NEW-KEY'}},
    )
    assert resp.status_code == 201

    stored = await settings_store.load()
    assert stored.llm_profiles.get('p').api_key.get_secret_value() == 'NEW-KEY'


@pytest.mark.asyncio
async def test_list_profiles_reports_api_key_set_per_row(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save(
        'with-key',
        LLM(model='openai/gpt-4o', api_key=SecretStr('sk-abc')),
    )
    settings.llm_profiles.save(
        'no-key',
        LLM(model='ollama/llama3', base_url='http://localhost:11434'),
    )
    await _seed(settings_store, settings)

    rows = {
        p['name']: p
        for p in test_client.get('/api/v1/settings/profiles').json()['profiles']
    }

    assert rows['with-key']['api_key_set'] is True
    assert rows['no-key']['api_key_set'] is False


@pytest.mark.asyncio
async def test_save_profile_rejects_unknown_llm_field(test_client, settings_store):
    """StrictLLM forbids extras → typo in an LLM field returns 422 instead of a silent 201."""
    await _seed(settings_store, _base_settings())

    response = test_client.post(
        '/api/v1/settings/profiles/typo',
        json={'llm': {'model': 'openai/gpt-4o', 'custom_header': 'x'}},
    )

    assert response.status_code == 422


# ── DELETE /profiles/{name} ──────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_profile_removes_it(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save('p', LLM(model='openai/gpt-4o'))
    await _seed(settings_store, settings)

    response = test_client.delete('/api/v1/settings/profiles/p')

    assert response.status_code == 200
    stored = await settings_store.load()
    assert not stored.llm_profiles.has('p')


@pytest.mark.asyncio
async def test_delete_profile_is_idempotent(test_client, settings_store):
    await _seed(settings_store, _base_settings())

    response = test_client.delete('/api/v1/settings/profiles/never-existed')

    assert response.status_code == 200
    assert response.json()['name'] == 'never-existed'


# ── POST /profiles/{name}/activate ───────────────────────────────


@pytest.mark.asyncio
async def test_activate_profile_updates_agent_llm(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save(
        'my-claude',
        LLM(
            model='anthropic/claude-opus-4',
            api_key=SecretStr('sk-claude'),
        ),
    )
    await _seed(settings_store, settings)

    response = test_client.post('/api/v1/settings/profiles/my-claude/activate')

    assert response.status_code == 200
    body = response.json()
    assert body['name'] == 'my-claude'
    assert body['model'] == 'anthropic/claude-opus-4'

    stored = await settings_store.load()
    assert stored.agent_settings.llm.model == 'anthropic/claude-opus-4'
    assert stored.llm_profiles.active == 'my-claude'


def test_activate_profile_returns_404_when_unknown(test_client):
    response = test_client.post('/api/v1/settings/profiles/ghost/activate')

    assert response.status_code == 404
    assert "'ghost'" in response.json()['detail']


@pytest.mark.asyncio
async def test_activate_profile_applies_base_url_fixup(test_client, settings_store):
    """Activating a profile with no base_url should get the provider default."""
    settings = _base_settings()
    settings.llm_profiles.save(
        'oh-profile',
        LLM(model='openhands/claude-sonnet-4-20250514'),
    )
    await _seed(settings_store, settings)

    response = test_client.post('/api/v1/settings/profiles/oh-profile/activate')
    assert response.status_code == 200

    stored = await settings_store.load()
    # Fixup inferred the proxy URL; profile itself remains unchanged.
    assert stored.agent_settings.llm.base_url is not None


# ── POST /profiles/{name}/rename ─────────────────────────────────


@pytest.mark.asyncio
async def test_rename_profile_renames_and_preserves_api_key(
    test_client, settings_store
):
    settings = _base_settings()
    settings.llm_profiles.save(
        'old',
        LLM(model='openai/gpt-4o', api_key=SecretStr('sk-keep')),
    )
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/old/rename',
        json={'new_name': 'new'},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['name'] == 'new'

    stored = await settings_store.load()
    assert stored.llm_profiles.get('old') is None
    renamed = stored.llm_profiles.get('new')
    assert renamed is not None
    assert renamed.api_key.get_secret_value() == 'sk-keep'


@pytest.mark.asyncio
async def test_rename_profile_preserves_active_flag(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save('p', LLM(model='openai/gpt-4o'))
    settings.llm_profiles.active = 'p'
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/p/rename',
        json={'new_name': 'q'},
    )
    assert response.status_code == 200

    stored = await settings_store.load()
    assert stored.llm_profiles.active == 'q'


@pytest.mark.asyncio
async def test_rename_profile_returns_404_when_unknown(test_client, settings_store):
    await _seed(settings_store, _base_settings())

    response = test_client.post(
        '/api/v1/settings/profiles/ghost/rename',
        json={'new_name': 'new'},
    )

    assert response.status_code == 404
    assert "'ghost'" in response.json()['detail']


@pytest.mark.asyncio
async def test_rename_profile_returns_409_when_target_exists(
    test_client, settings_store
):
    settings = _base_settings()
    settings.llm_profiles.save('a', LLM(model='openai/gpt-4o'))
    settings.llm_profiles.save('b', LLM(model='anthropic/claude-opus-4'))
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/a/rename',
        json={'new_name': 'b'},
    )

    assert response.status_code == 409
    assert "'b'" in response.json()['detail']

    # Both originals should be intact after the failed rename.
    stored = await settings_store.load()
    assert stored.llm_profiles.has('a')
    assert stored.llm_profiles.has('b')


@pytest.mark.asyncio
async def test_rename_profile_to_same_name_is_noop(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save('p', LLM(model='openai/gpt-4o'))
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/p/rename',
        json={'new_name': 'p'},
    )

    assert response.status_code == 200
    stored = await settings_store.load()
    assert stored.llm_profiles.has('p')


@pytest.mark.asyncio
async def test_rename_profile_rejects_invalid_new_name(test_client, settings_store):
    settings = _base_settings()
    settings.llm_profiles.save('p', LLM(model='openai/gpt-4o'))
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/p/rename',
        json={'new_name': 'has space'},
    )

    assert response.status_code == 422


# ── Name validation ──────────────────────────────────────────────


@pytest.mark.parametrize(
    'bad_name',
    [
        'a' * 65,  # too long
        'with space',  # disallowed char
        'with/slash',  # slash splits the path → endpoint not matched
        'weird$chars',  # disallowed char
    ],
)
def test_save_profile_rejects_invalid_name(test_client, bad_name):
    response = test_client.post(
        f'/api/v1/settings/profiles/{bad_name}',
        json={'llm': {'model': 'openai/gpt-4o'}},
    )
    # Invalid chars/length → 422 from Path validation; slash → 404/405 routing miss.
    assert response.status_code in (404, 405, 422)


# ── Profile count cap ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_profile_returns_409_past_limit(test_client, settings_store):
    settings = _base_settings()
    for i in range(MAX_PROFILES_PER_USER):
        settings.llm_profiles.save(f'p{i}', LLM(model='openai/gpt-4o'))
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/one-too-many',
        json={'llm': {'model': 'openai/gpt-4o'}},
    )

    assert response.status_code == 409
    assert 'limit' in response.json()['detail'].lower()


@pytest.mark.asyncio
async def test_save_profile_at_limit_can_still_overwrite(test_client, settings_store):
    settings = _base_settings()
    for i in range(MAX_PROFILES_PER_USER):
        settings.llm_profiles.save(f'p{i}', LLM(model='openai/gpt-4o'))
    await _seed(settings_store, settings)

    response = test_client.post(
        '/api/v1/settings/profiles/p0',
        json={'llm': {'model': 'anthropic/claude-opus-4'}},
    )

    assert response.status_code == 201
    stored = await settings_store.load()
    assert stored.llm_profiles.get('p0').model == 'anthropic/claude-opus-4'


# ── Orphan active auto-heals ─────────────────────────────────────


@pytest.mark.asyncio
async def test_list_profiles_clears_orphan_active(test_client, settings_store):
    """A persisted state with active pointing at a missing profile should
    self-heal on the next load — ``active_profile`` is reported as None."""
    settings = _base_settings()
    settings.llm_profiles.save('real', LLM(model='openai/gpt-4o'))
    # Bypass the invariant validator to persist a corrupt state.
    object.__setattr__(settings.llm_profiles, 'active', 'ghost')
    await _seed(settings_store, settings)

    response = test_client.get('/api/v1/settings/profiles')

    assert response.status_code == 200
    assert response.json()['active_profile'] is None


# ── Real-world scenarios ─────────────────────────────────────────
#
# Each test below maps to a specific production risk surfaced by the
# ``.pr/probe_profiles.py`` exploration script. Single assertion-per-
# behaviour; one test per risk.
#
# Scenario 1 — API key never leaks in any response
# Scenario 2 — Direct LLM edits invalidate a previously-active profile
# Scenario 3 — The generic settings POST cannot inject or mutate profiles
# Scenario 4 — The profile-count cap frees a slot when a profile is deleted


_SECRET = 'sk-PROBE-MUST-NOT-LEAK'


@pytest.mark.asyncio
async def test_api_key_never_leaks_across_response_paths(test_client, settings_store):
    """Scenario 1 — API key never leaks in any response.

    User saves a profile that carries an api_key, then exercises every
    endpoint that might echo the stored LLM back: the save response
    itself, ``GET /profiles/{name}``, the activate response, and the
    big ``GET /api/v1/settings`` payload that embeds all profiles. One
    regression in any serializer would leak the key in logs, UI, or
    exported settings — this test catches that across the full surface.
    """
    await _seed(settings_store, _base_settings())

    save = test_client.post(
        '/api/v1/settings/profiles/leak-check',
        json={'llm': {'model': 'openai/gpt-4o', 'api_key': _SECRET}},
    )
    assert save.status_code == 201 and _SECRET not in save.text
    assert _SECRET not in test_client.get('/api/v1/settings/profiles/leak-check').text
    assert (
        _SECRET
        not in test_client.post('/api/v1/settings/profiles/leak-check/activate').text
    )
    assert _SECRET not in test_client.get('/api/v1/settings').text


@pytest.mark.asyncio
async def test_journey_direct_llm_edit_clears_active(test_client, settings_store):
    """Scenario 2 — direct LLM edits invalidate a previously-active profile.

    User saves profile ``j``, activates it, then opens the main Settings
    page and edits the model/api_key. The currently-running LLM now
    differs from the saved profile, so the UI must stop showing ``j`` as
    active. This end-to-end HTTP flow exercises the same reconciliation
    the unit tests cover, but through the real ``POST /api/v1/settings``
    path users actually hit.
    """
    await _seed(settings_store, _base_settings())
    test_client.post(
        '/api/v1/settings/profiles/j',
        json={'llm': {'model': 'openai/gpt-4o'}},
    )
    test_client.post('/api/v1/settings/profiles/j/activate')
    assert test_client.get('/api/v1/settings/profiles').json()['active_profile'] == 'j'

    test_client.post(
        '/api/v1/settings',
        json={'agent_settings_diff': {'llm': {'model': 'anthropic/claude-opus-4'}}},
    )

    assert test_client.get('/api/v1/settings/profiles').json()['active_profile'] is None


@pytest.mark.asyncio
async def test_main_settings_endpoint_ignores_llm_profiles_payload(
    test_client, settings_store
):
    """Scenario 3 — the generic settings POST cannot inject or mutate profiles.

    A malicious or buggy client might stuff ``llm_profiles`` into a
    ``POST /api/v1/settings`` body to bypass the dedicated profile
    endpoints (which enforce name rules, the count cap, and the per-user
    lock). The probe found this path used to crash with 500; the server
    must now silently drop the key instead, keeping all profile state
    untouched and returning 200.
    """
    await _seed(settings_store, _base_settings())

    resp = test_client.post(
        '/api/v1/settings',
        json={
            'llm_profiles': {
                'profiles': {'ATTACKER': {'model': 'openai/gpt-4o'}},
                'active': 'ATTACKER',
            },
        },
    )

    assert resp.status_code == 200  # silently ignored, not crashed
    listing = test_client.get('/api/v1/settings/profiles').json()
    assert 'ATTACKER' not in {p['name'] for p in listing['profiles']}


@pytest.mark.asyncio
async def test_delete_active_profile_clears_active_and_allows_recovery(
    test_client, settings_store
):
    """Scenario 5 — Delete the currently-active profile.

    User activates profile ``A``, then deletes it (misclick or intentional).
    ``GET /profiles`` must show ``active_profile: null`` afterwards, and
    the user must be able to immediately save and activate a replacement
    — i.e. the in-memory state is not wedged by removing the active entry.
    """
    await _seed(settings_store, _base_settings())
    test_client.post(
        '/api/v1/settings/profiles/A',
        json={'llm': {'model': 'openai/gpt-4o'}},
    )
    test_client.post('/api/v1/settings/profiles/A/activate')

    test_client.delete('/api/v1/settings/profiles/A')

    assert test_client.get('/api/v1/settings/profiles').json()['active_profile'] is None

    # Recovery: save + activate a replacement.
    test_client.post(
        '/api/v1/settings/profiles/B',
        json={'llm': {'model': 'anthropic/claude-opus-4'}},
    )
    assert test_client.post('/api/v1/settings/profiles/B/activate').status_code == 200


@pytest.mark.asyncio
async def test_switching_between_profiles_updates_active_and_model(
    test_client, settings_store
):
    """Scenario 6 — Switch from profile A to profile B.

    Two-profile user activates ``A`` first, then ``B``. ``active_profile``
    must flip and ``agent_settings.llm.model`` must follow — no stale
    caching or residual state from the previous activation.
    """
    await _seed(settings_store, _base_settings())
    test_client.post(
        '/api/v1/settings/profiles/A',
        json={'llm': {'model': 'openai/gpt-4o'}},
    )
    test_client.post(
        '/api/v1/settings/profiles/B',
        json={'llm': {'model': 'anthropic/claude-opus-4'}},
    )

    test_client.post('/api/v1/settings/profiles/A/activate')
    s1 = test_client.get('/api/v1/settings').json()
    assert s1['agent_settings']['llm']['model'] == 'openai/gpt-4o'
    assert test_client.get('/api/v1/settings/profiles').json()['active_profile'] == 'A'

    test_client.post('/api/v1/settings/profiles/B/activate')
    s2 = test_client.get('/api/v1/settings').json()
    assert s2['agent_settings']['llm']['model'] == 'anthropic/claude-opus-4'
    assert test_client.get('/api/v1/settings/profiles').json()['active_profile'] == 'B'


@pytest.mark.asyncio
async def test_profiles_are_isolated_between_users(tmp_path_factory):
    """Scenario 7 — One user's profiles don't appear in another user's response.

    Alice saves a profile against her store; Bob lists profiles against
    his store. Bob must see none of Alice's — this pins the absence of
    router-level cross-user state leaks (e.g. module-level caches that
    don't key on user_id, or settings stores that accidentally share
    data).
    """
    store_a = FileSettingsStore(
        get_file_store('local', str(tmp_path_factory.mktemp('alice')))
    )
    store_b = FileSettingsStore(
        get_file_store('local', str(tmp_path_factory.mktemp('bob')))
    )
    await store_a.store(_base_settings())
    await store_b.store(_base_settings())

    with _client_for_user('alice', store_a) as ca:
        assert (
            ca.post(
                '/api/v1/settings/profiles/alice-only',
                json={'llm': {'model': 'openai/gpt-4o'}},
            ).status_code
            == 201
        )

    with _client_for_user('bob', store_b) as cb:
        body = cb.get('/api/v1/settings/profiles').json()
        assert 'alice-only' not in {p['name'] for p in body['profiles']}


@pytest.mark.asyncio
async def test_cap_frees_after_delete(test_client, settings_store):
    """Scenario 4 — the profile-count cap frees a slot when a profile is deleted.

    User already has ``MAX_PROFILES_PER_USER`` profiles and tries to
    save another (409). They delete one profile via the UI, then retry
    the save: it must now succeed. Without this, a user who hits the
    cap once is stuck forever.
    """
    settings = _base_settings()
    for i in range(MAX_PROFILES_PER_USER):
        settings.llm_profiles.save(f'p{i}', LLM(model='openai/gpt-4o'))
    await _seed(settings_store, settings)

    assert (
        test_client.post(
            '/api/v1/settings/profiles/over',
            json={'llm': {'model': 'openai/gpt-4o'}},
        ).status_code
        == 409
    )

    test_client.delete('/api/v1/settings/profiles/p0')

    assert (
        test_client.post(
            '/api/v1/settings/profiles/over',
            json={'llm': {'model': 'openai/gpt-4o'}},
        ).status_code
        == 201
    )


# ── Lost-update race ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_writes_all_persist(tmp_path: Path):
    """Drive the handler under a single event loop with N truly concurrent
    coroutines. Each handler loads → mutates → stores under the per-user
    lock; all N profiles must land without any getting clobbered.

    Bypasses ``TestClient`` (which spawns its own loop per request and makes
    the module-level ``asyncio.Lock`` unreachable across calls)."""
    import asyncio

    from openhands.app_server.settings.settings_router import (
        SaveProfileRequest,
        save_profile,
    )

    store = FileSettingsStore(get_file_store('local', str(tmp_path)))
    await store.store(_base_settings())

    async def _save_one(i: int) -> None:
        await save_profile(
            name=f'p{i}',
            request=SaveProfileRequest.model_validate(
                {'llm': {'model': 'openai/gpt-4o'}}
            ),
            user_id='same-user',
            settings_store=store,
        )

    await asyncio.gather(*(_save_one(i) for i in range(10)))

    stored = await store.load()
    assert {p['name'] for p in stored.llm_profiles.summaries()} == {
        f'p{i}' for i in range(10)
    }
