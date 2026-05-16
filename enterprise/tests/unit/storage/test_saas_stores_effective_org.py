"""Unit tests for ``effective_org_id`` plumbing through the SaaS stores.

The SaaS settings / secrets stores have historically scoped all reads
and writes to ``user.current_org_id``. That made every API-key-driven
request load the *user's last-selected* org's data — even when the
caller was authenticated via an API key bound to a *different* org.

These tests verify that:

* When ``effective_org_id`` is supplied to the store constructor or
  ``get_instance``, the store uses it instead of ``user.current_org_id``.
* When it is not supplied, the store still falls back to
  ``user.current_org_id`` so background callers (webhook resolvers,
  CLI flows) keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from storage.saas_secrets_store import SaasSecretsStore
from storage.saas_settings_store import SaasSettingsStore

USER_ID = '00000000-0000-0000-0000-00000000aaaa'
CURRENT_ORG_ID = UUID('00000000-0000-0000-0000-00000000bbbb')
EFFECTIVE_ORG_ID = UUID('00000000-0000-0000-0000-00000000cccc')


# --------------------------------------------------------------------- #
# Lightweight fakes — we explicitly do NOT spin up SQLite for these
# tests. We're verifying which org_id is *chosen* by the store, not the
# SQL it emits. Hitting the DB would test SQLAlchemy more than the
# resolver, and would tie the test to schema details unrelated to the
# bug we're guarding against.
# --------------------------------------------------------------------- #


@dataclass
class FakeOrgMember:
    org_id: UUID


@dataclass
class FakeUser:
    """Minimal stand-in for ``storage.user.User`` covering only the
    attributes the stores touch on the happy path of load().
    """

    current_org_id: UUID
    org_members: list[FakeOrgMember] = field(default_factory=list)


# ----------------------------- SettingsStore --------------------------- #


def test_settings_store_default_effective_org_id_is_none():
    """The new field must default to None so existing constructions
    (e.g. ``SaasSettingsStore(user_id)``) keep working unchanged."""
    store = SaasSettingsStore(user_id=USER_ID)
    assert store.effective_org_id is None


def test_settings_store_resolve_org_id_prefers_effective_over_current():
    user = FakeUser(current_org_id=CURRENT_ORG_ID)
    store = SaasSettingsStore(user_id=USER_ID, effective_org_id=EFFECTIVE_ORG_ID)
    assert store._resolve_org_id(user) == EFFECTIVE_ORG_ID


def test_settings_store_resolve_org_id_falls_back_to_current_org():
    user = FakeUser(current_org_id=CURRENT_ORG_ID)
    store = SaasSettingsStore(user_id=USER_ID)  # no effective_org_id
    assert store._resolve_org_id(user) == CURRENT_ORG_ID


@pytest.mark.asyncio
async def test_settings_store_load_returns_none_when_user_not_member_of_effective_org():
    """If the request resolved an effective org the user isn't a member
    of, the store must NOT silently fall back to current_org_id — it
    must refuse the load (the upstream resolver is responsible for
    rejecting unauthorized X-Org-Id values; this is defense-in-depth)."""
    user = FakeUser(
        current_org_id=CURRENT_ORG_ID,
        org_members=[FakeOrgMember(org_id=CURRENT_ORG_ID)],
    )
    store = SaasSettingsStore(user_id=USER_ID, effective_org_id=EFFECTIVE_ORG_ID)

    with patch(
        'storage.saas_settings_store.UserStore.get_user_by_id',
        new=AsyncMock(return_value=user),
    ):
        result = await store.load()

    assert result is None


@pytest.mark.asyncio
async def test_settings_store_get_instance_propagates_effective_org_id():
    store = await SaasSettingsStore.get_instance(
        USER_ID, effective_org_id=EFFECTIVE_ORG_ID
    )
    assert isinstance(store, SaasSettingsStore)
    assert store.user_id == USER_ID
    assert store.effective_org_id == EFFECTIVE_ORG_ID


@pytest.mark.asyncio
async def test_settings_store_get_instance_defaults_to_none():
    """Webhook resolvers call ``get_instance(user_id)`` without an org;
    that path must remain legal."""
    store = await SaasSettingsStore.get_instance(USER_ID)
    assert store.effective_org_id is None


# ----------------------------- SecretsStore ---------------------------- #


def _make_secrets_store(effective_org_id: UUID | None = None) -> SaasSecretsStore:
    return SaasSecretsStore(
        user_id=USER_ID,
        _jwt_svc=MagicMock(),
        effective_org_id=effective_org_id,
    )


def test_secrets_store_default_effective_org_id_is_none():
    store = _make_secrets_store()
    assert store.effective_org_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'effective_org_id,expected_org_id',
    [
        (EFFECTIVE_ORG_ID, EFFECTIVE_ORG_ID),  # uses effective override
        (None, CURRENT_ORG_ID),  # falls back to user.current_org_id
    ],
)
async def test_secrets_store_load_filters_by_resolved_org_id(
    effective_org_id, expected_org_id
):
    """``load()`` builds a query of the shape
    ``SELECT … WHERE keycloak_user_id = :u AND org_id = :o`` — assert
    the org id bound into that filter matches the resolved org."""
    user = FakeUser(current_org_id=CURRENT_ORG_ID)

    captured_queries: list[object] = []

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return []

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, query):
            captured_queries.append(query)
            return _Result()

    store = _make_secrets_store(effective_org_id=effective_org_id)

    with (
        patch(
            'storage.saas_secrets_store.UserStore.get_user_by_id',
            new=AsyncMock(return_value=user),
        ),
        patch(
            'storage.saas_secrets_store.a_session_maker',
            return_value=_FakeSession(),
        ),
    ):
        result = await store.load()

    assert result is not None
    assert captured_queries, 'load() should have executed at least one query'

    # Compile the SELECT with literal binds so we can read the org id
    # value out of the WHERE clause without needing a real DB driver.
    # SQLAlchemy renders UUID literals as the 32-char hex form (no
    # hyphens), so compare against `.hex` rather than `str(...)`.
    compiled = str(captured_queries[-1].compile(compile_kwargs={'literal_binds': True}))
    assert expected_org_id.hex in compiled, (
        f'Expected query to filter by org_id={expected_org_id!s} '
        f'(hex={expected_org_id.hex}), compiled SQL was: {compiled}'
    )


@pytest.mark.asyncio
async def test_secrets_store_store_uses_effective_org_id_when_set():
    """``store()`` must delete/insert under the effective org, not the
    user's current org. We capture the org id by intercepting the
    ``StoredCustomSecrets`` row that would be inserted."""

    captured_org_ids: list[UUID] = []
    user = FakeUser(current_org_id=CURRENT_ORG_ID)

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, _query):
            return MagicMock()

        def add(self, row):
            captured_org_ids.append(row.org_id)

        async def commit(self):
            return None

    # Build a Secrets object with one custom secret. We use the real
    # model so encryption / model_dump behave correctly.
    from pydantic import SecretStr

    from openhands.app_server.integrations.provider import CustomSecret
    from openhands.app_server.secrets.secrets_models import Secrets

    item = Secrets(
        custom_secrets={
            'API_KEY': CustomSecret(secret=SecretStr('shh'), description='example'),
        }
    )

    store = _make_secrets_store(effective_org_id=EFFECTIVE_ORG_ID)

    with (
        patch(
            'storage.saas_secrets_store.UserStore.get_user_by_id',
            new=AsyncMock(return_value=user),
        ),
        patch(
            'storage.saas_secrets_store.a_session_maker',
            return_value=_FakeSession(),
        ),
    ):
        await store.store(item)

    assert captured_org_ids == [
        EFFECTIVE_ORG_ID
    ], f'store() wrote under {captured_org_ids[0]!s}, expected {EFFECTIVE_ORG_ID!s}'


@pytest.mark.asyncio
async def test_secrets_store_store_falls_back_to_current_org_when_unset():
    """The legacy/webhook code path (no effective org supplied) must
    keep writing under user.current_org_id."""
    captured_org_ids: list[UUID] = []
    user = FakeUser(current_org_id=CURRENT_ORG_ID)

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, _query):
            return MagicMock()

        def add(self, row):
            captured_org_ids.append(row.org_id)

        async def commit(self):
            return None

    from pydantic import SecretStr

    from openhands.app_server.integrations.provider import CustomSecret
    from openhands.app_server.secrets.secrets_models import Secrets

    item = Secrets(
        custom_secrets={
            'API_KEY': CustomSecret(secret=SecretStr('shh'), description='example'),
        }
    )

    store = _make_secrets_store()  # no effective org

    with (
        patch(
            'storage.saas_secrets_store.UserStore.get_user_by_id',
            new=AsyncMock(return_value=user),
        ),
        patch(
            'storage.saas_secrets_store.a_session_maker',
            return_value=_FakeSession(),
        ),
    ):
        await store.store(item)

    assert captured_org_ids == [CURRENT_ORG_ID]


@pytest.mark.asyncio
async def test_secrets_store_get_instance_propagates_effective_org_id():
    with patch(
        'storage.encrypt_utils.get_jwt_service',
        return_value=MagicMock(),
    ):
        store = await SaasSecretsStore.get_instance(
            USER_ID, effective_org_id=EFFECTIVE_ORG_ID
        )
    assert store.effective_org_id == EFFECTIVE_ORG_ID


@pytest.mark.asyncio
async def test_secrets_store_get_instance_defaults_to_none():
    with patch(
        'storage.encrypt_utils.get_jwt_service',
        return_value=MagicMock(),
    ):
        store = await SaasSecretsStore.get_instance(USER_ID)
    assert store.effective_org_id is None


# ----------------- SaasUserAuth -> store wiring ----------------------- #


def _make_saas_user_auth():
    """``SaasUserAuth`` requires a refresh_token; tests don't exercise
    it, so a dummy SecretStr is sufficient."""
    from pydantic import SecretStr
    from server.auth.saas_user_auth import SaasUserAuth

    return SaasUserAuth(refresh_token=SecretStr('test-refresh'), user_id=USER_ID)


@pytest.mark.asyncio
async def test_saas_user_auth_get_user_settings_store_passes_effective_org():
    """The auth helper must pull the effective org from
    ``get_effective_org_id()`` and hand it to the settings store."""
    from server.auth.saas_user_auth import SaasUserAuth

    auth = _make_saas_user_auth()

    with patch.object(
        SaasUserAuth,
        'get_effective_org_id',
        new=AsyncMock(return_value=EFFECTIVE_ORG_ID),
    ):
        store = await auth.get_user_settings_store()

    assert isinstance(store, SaasSettingsStore)
    assert store.effective_org_id == EFFECTIVE_ORG_ID

    # Second call returns the cached instance, doesn't re-resolve.
    cached = await auth.get_user_settings_store()
    assert cached is store


@pytest.mark.asyncio
async def test_saas_user_auth_get_secrets_store_passes_effective_org():
    from server.auth.saas_user_auth import SaasUserAuth

    auth = _make_saas_user_auth()

    with (
        patch.object(
            SaasUserAuth,
            'get_effective_org_id',
            new=AsyncMock(return_value=EFFECTIVE_ORG_ID),
        ),
        patch(
            'storage.encrypt_utils.get_jwt_service',
            return_value=MagicMock(),
        ),
    ):
        store = await auth.get_secrets_store()

    assert isinstance(store, SaasSecretsStore)
    assert store.effective_org_id == EFFECTIVE_ORG_ID
