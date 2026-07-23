"""Unit + integration tests for the cloud Agent Profiles surface (#15044).

Covers the ``AgentProfiles`` container (SDK ``AgentProfileStoreProtocol``
conformance), the flat ``/api/agent-profiles`` router, the LLM-profile FK guard
wired into ``org_profiles``, and ``SaasSettingsStore._resolve_active_agent_profile``.
Mirrors the harness in ``test_org_profiles.py``: handlers are called directly
(``Depends`` resolved as kwargs) against a real SQLite Org row.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role
from storage.user import User

from openhands.app_server.settings.agent_profiles import (
    MAX_AGENT_PROFILES,
    AgentProfiles,
)
from openhands.app_server.user_auth import get_user_id
from openhands.sdk.profiles import (
    ACPAgentProfile,
    AgentProfileStoreProtocol,
    OpenHandsAgentProfile,
    save_profile_preserving_identity,
)

# Mock the database module before importing the routers so module-level imports
# don't touch a real engine (matches test_org_profiles.py).
with patch('storage.database.a_session_maker'):
    from server.routes.agent_profiles import (
        ActivateAgentProfileResponse,
        AgentProfileListResponse,
        RenameAgentProfileRequest,
        activate_agent_profile,
        delete_agent_profile,
        get_agent_profile,
        list_agent_profiles,
        materialize_agent_profile,
        rename_agent_profile,
        save_agent_profile,
    )
    from server.routes.agent_profiles import router as agent_profiles_router
    from server.routes.org_profiles import (
        RenameProfileRequest,
        delete_profile,
        rename_profile,
    )
    from storage.agent_profile_resolution import (
        load_agent_profiles,
        member_mcp_config,
    )

ORG_ID = uuid.UUID('6694c7b6-f959-4b81-92e9-b09c206f5081')
USER_ID = uuid.UUID('6694c7b6-f959-4b81-92e9-b09c206f5082')


# ── Container model ────────────────────────────────────────────────────────


class TestAgentProfilesContainer:
    def test_conforms_to_sdk_protocol(self):
        assert isinstance(AgentProfiles(), AgentProfileStoreProtocol)
        assert MAX_AGENT_PROFILES == 50

    def test_id_lifecycle_and_name_keyed_protocol_ops(self):
        store = AgentProfiles()
        created = save_profile_preserving_identity(
            store, OpenHandsAgentProfile(name='reviewer', llm_profile_ref='gpt')
        )
        assert created.revision == 0
        # overwrite keeps id, bumps revision; create mints a fresh id
        again = save_profile_preserving_identity(
            store, OpenHandsAgentProfile(name='reviewer', llm_profile_ref='haiku')
        )
        assert again.id == created.id and again.revision == 1
        assert store.load('reviewer').llm_profile_ref == 'haiku'
        assert store.name_for_id(created.id) == 'reviewer'

        save_profile_preserving_identity(
            store, ACPAgentProfile(name='claude', acp_server='claude-code')
        )
        summaries = {s['name']: s for s in store.list_summaries()}
        assert summaries['reviewer']['llm_profile_ref'] == 'haiku'
        assert summaries['claude']['llm_profile_ref'] is None  # ACP carries no ref

    def test_rename_preserves_id_and_active_pointer(self):
        store = AgentProfiles()
        p = save_profile_preserving_identity(
            store, OpenHandsAgentProfile(name='a', llm_profile_ref='gpt')
        )
        store.active = str(p.id)
        store.rename('a', 'b')
        assert store.name_for_id(p.id) == 'b'
        assert store.active == str(p.id)  # id-keyed pointer survives rename
        with pytest.raises(FileNotFoundError):
            store.load('a')

    def test_delete_clears_org_active_pointer(self):
        store = AgentProfiles()
        p = save_profile_preserving_identity(
            store, OpenHandsAgentProfile(name='a', llm_profile_ref='gpt')
        )
        store.active = str(p.id)
        store.delete('a')
        assert store.active is None

    def test_limit_enforced(self):
        store = AgentProfiles()
        for i in range(MAX_AGENT_PROFILES):
            save_profile_preserving_identity(
                store, OpenHandsAgentProfile(name=f'p{i}', llm_profile_ref='gpt')
            )
        from openhands.sdk.profiles import ProfileLimitExceeded

        with pytest.raises(ProfileLimitExceeded):
            save_profile_preserving_identity(
                store,
                OpenHandsAgentProfile(name='over', llm_profile_ref='gpt'),
                max_profiles=MAX_AGENT_PROFILES,
            )

    def test_encrypted_json_roundtrip_via_dict(self):
        store = AgentProfiles()
        save_profile_preserving_identity(
            store, OpenHandsAgentProfile(name='r', llm_profile_ref='gpt')
        )
        dumped = store.model_dump(mode='json', context={'expose_secrets': True})
        reloaded = AgentProfiles.model_validate(dumped)
        assert reloaded.load('r').llm_profile_ref == 'gpt'

    def test_invalid_entry_is_skipped_not_fatal(self):
        store = AgentProfiles.model_validate(
            {'profiles': {'bad': {'agent_kind': 'nonsense'}}, 'active': None}
        )
        assert store.list_summaries() == []


def test_load_agent_profiles_defaults_empty_and_degrades():
    org = MagicMock(spec=Org)
    org.id = ORG_ID
    org.agent_profiles = None
    assert load_agent_profiles(org).list_summaries() == []
    # Garbage envelope degrades to empty rather than raising.
    org.agent_profiles = {'profiles': 'not-a-dict'}
    assert load_agent_profiles(org).list_summaries() == []


def test_member_mcp_config_degrades_on_non_validation_error():
    """coerce_mcp_config failures beyond ValidationError degrade to {}, not raise."""
    member = MagicMock(spec=OrgMember)
    member.mcp_config = {'mcpServers': {}}
    member.agent_settings_diff = {}
    with patch(
        'storage.agent_profile_resolution.coerce_mcp_config',
        side_effect=TypeError('contract drift'),
    ):
        assert member_mcp_config(member) == {}


# ── Router integration (real Org row over SQLite) ──────────────────────────


@pytest.fixture
def seeded_org(session_maker):
    with session_maker() as session:
        session.add(Role(id=20, name='member', rank=3))
        session.add(
            Org(
                id=ORG_ID,
                name='agent-profile-test-org',
                org_version=1,
                enable_proactive_conversation_starters=True,
                # An LLM profile the seed/resolve can reference.
                llm_profiles={
                    'profiles': {'Default': {'model': 'gpt-4o', 'api_key': 'k'}},
                    'active': 'Default',
                },
            )
        )
        session.add(
            User(id=USER_ID, current_org_id=ORG_ID, user_consents_to_analytics=True)
        )
        session.add(
            OrgMember(
                org_id=ORG_ID,
                user_id=USER_ID,
                role_id=20,
                llm_api_key='initial-key',
                agent_settings_diff={},
                conversation_settings_diff={},
                status='active',
            )
        )
        session.commit()
    return ORG_ID


@pytest.fixture
def patch_agent_routes(async_session_maker, seeded_org):
    async def _fake_get_org(org_id, user_id):  # noqa: ARG001
        async with async_session_maker() as session:
            result = await session.execute(select(Org).where(Org.id == org_id))
            return result.scalars().first()

    with (
        patch('server.routes.agent_profiles.a_session_maker', async_session_maker),
        patch(
            'server.routes.agent_profiles.OrgService.get_org_by_id',
            side_effect=_fake_get_org,
        ),
    ):
        yield seeded_org


async def _read_member(async_session_maker, org_id, user_id):
    async with async_session_maker() as session:
        result = await session.execute(
            select(OrgMember).where(
                OrgMember.org_id == org_id, OrgMember.user_id == user_id
            )
        )
        return result.scalars().first()


class TestAgentProfileRouterLifecycle:
    @pytest.mark.asyncio
    async def test_save_list_get_rename_activate_delete(
        self, async_session_maker, patch_agent_routes
    ):
        org_id = patch_agent_routes
        uid = str(USER_ID)

        # save (create)
        await save_agent_profile(
            name='reviewer',
            body={'llm_profile_ref': 'Default'},
            effective_org_id=org_id,
            user_id=uid,
        )
        # list shows it
        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)
        assert [p.name for p in listing.profiles] == ['reviewer']
        profile_id = listing.profiles[0].id
        assert profile_id is not None
        assert listing.profiles[0].llm_profile_ref == 'Default'

        # get detail (profile is the typed discriminated union, not a dict)
        detail = await get_agent_profile(
            name='reviewer', effective_org_id=org_id, user_id=uid
        )
        assert detail.profile.agent_kind == 'openhands'
        assert detail.profile.llm_profile_ref == 'Default'

        # overwrite bumps revision, keeps id
        await save_agent_profile(
            name='reviewer',
            body={'llm_profile_ref': 'Default', 'tool_concurrency_limit': 3},
            effective_org_id=org_id,
            user_id=uid,
        )
        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)
        assert listing.profiles[0].id == profile_id
        assert listing.profiles[0].revision == 1

        # rename preserves id
        await rename_agent_profile(
            name='reviewer',
            request=RenameAgentProfileRequest(new_name='lead-reviewer'),
            effective_org_id=org_id,
            user_id=uid,
        )
        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)
        assert listing.profiles[0].name == 'lead-reviewer'
        assert listing.profiles[0].id == profile_id

        # activate writes the per-member pointer (pointer-only)
        resp = await activate_agent_profile(
            profile_id=profile_id, effective_org_id=org_id, user_id=uid
        )
        assert isinstance(resp, ActivateAgentProfileResponse)
        assert resp.agent_settings_applied is False
        member = await _read_member(async_session_maker, org_id, USER_ID)
        assert member.active_agent_profile_id == profile_id

        # delete clears the pointer
        await delete_agent_profile(
            name='lead-reviewer', effective_org_id=org_id, user_id=uid
        )
        member = await _read_member(async_session_maker, org_id, USER_ID)
        assert member.active_agent_profile_id is None
        # Read the row directly (calling list here would lazily re-seed a default).
        async with async_session_maker() as session:
            org = (
                (await session.execute(select(Org).where(Org.id == org_id)))
                .scalars()
                .first()
            )
        assert load_agent_profiles(org).list_summaries() == []


class TestDeleteClearsAllMemberPointers:
    """Deleting a profile must clear every org member's pointer to it, not
    just the acting member's (activation is per-member)."""

    @pytest.mark.asyncio
    async def test_delete_clears_other_members_active_pointer(
        self, async_session_maker, patch_agent_routes
    ):
        org_id = patch_agent_routes
        uid = str(USER_ID)
        other_user_id = uuid.UUID('6694c7b6-f959-4b81-92e9-b09c206f5099')

        async with async_session_maker() as session:
            session.add(
                User(
                    id=other_user_id,
                    current_org_id=org_id,
                    user_consents_to_analytics=True,
                )
            )
            session.add(
                OrgMember(
                    org_id=org_id,
                    user_id=other_user_id,
                    role_id=20,
                    llm_api_key='other-initial-key',
                    agent_settings_diff={},
                    conversation_settings_diff={},
                    status='active',
                )
            )
            await session.commit()

        await save_agent_profile(
            name='shared',
            body={'llm_profile_ref': 'Default'},
            effective_org_id=org_id,
            user_id=uid,
        )
        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)
        profile_id = listing.profiles[0].id
        assert profile_id is not None

        # Both members activate the same profile independently.
        await activate_agent_profile(
            profile_id=profile_id, effective_org_id=org_id, user_id=uid
        )
        await activate_agent_profile(
            profile_id=profile_id, effective_org_id=org_id, user_id=str(other_user_id)
        )
        other_member = await _read_member(async_session_maker, org_id, other_user_id)
        assert other_member.active_agent_profile_id == profile_id

        # The acting member deletes the profile.
        await delete_agent_profile(name='shared', effective_org_id=org_id, user_id=uid)

        acting_member = await _read_member(async_session_maker, org_id, USER_ID)
        assert acting_member.active_agent_profile_id is None
        other_member = await _read_member(async_session_maker, org_id, other_user_id)
        assert other_member.active_agent_profile_id is None, (
            "other member's pointer must be cleared too"
        )


class TestAgentProfileRouterErrors:
    @pytest.mark.asyncio
    async def test_get_missing_404(self, patch_agent_routes):
        with pytest.raises(HTTPException) as exc:
            await get_agent_profile(
                name='nope', effective_org_id=patch_agent_routes, user_id=str(USER_ID)
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_missing_is_idempotent_200(self, patch_agent_routes):
        # A missing name resolves 200 (no raise), matching the ts-client
        # AgentProfilesClient contract and the local agent-server delete_profile.
        # Canvas's delete mutation has no 404 branch.
        resp = await delete_agent_profile(
            name='never-existed',
            effective_org_id=patch_agent_routes,
            user_id=str(USER_ID),
        )
        assert resp.name == 'never-existed'

    @pytest.mark.asyncio
    async def test_activate_unknown_id_404(self, patch_agent_routes):
        with pytest.raises(HTTPException) as exc:
            await activate_agent_profile(
                profile_id=str(uuid.uuid4()),
                effective_org_id=patch_agent_routes,
                user_id=str(USER_ID),
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_save_invalid_payload_422(self, patch_agent_routes):
        with pytest.raises(HTTPException) as exc:
            # cross-variant mongrel: acp_kind payload carrying llm_profile_ref
            await save_agent_profile(
                name='bad',
                body={'agent_kind': 'acp', 'llm_profile_ref': 'x'},
                effective_org_id=patch_agent_routes,
                user_id=str(USER_ID),
            )
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_materialize_dangling_llm_ref_is_invalid_not_404(
        self, patch_agent_routes
    ):
        org_id = patch_agent_routes
        uid = str(USER_ID)
        await save_agent_profile(
            name='dangler',
            body={'llm_profile_ref': 'does-not-exist'},
            effective_org_id=org_id,
            user_id=uid,
        )
        diag = await materialize_agent_profile(
            name='dangler', effective_org_id=org_id, user_id=uid
        )
        assert diag.valid is False
        assert any('does-not-exist' in e for e in diag.errors)

    @pytest.mark.asyncio
    async def test_materialize_valid_profile(self, patch_agent_routes):
        org_id = patch_agent_routes
        uid = str(USER_ID)
        await save_agent_profile(
            name='ok',
            body={'llm_profile_ref': 'Default'},
            effective_org_id=org_id,
            user_id=uid,
        )
        diag = await materialize_agent_profile(
            name='ok', effective_org_id=org_id, user_id=uid
        )
        assert diag.valid is True
        assert diag.llm_profile_resolved is True


class TestListAgentProfiles:
    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_list_no_implicit_write(
        self, async_session_maker, patch_agent_routes
    ):
        """No auto-seed: an org with no profiles yet just gets an empty list,
        and nothing is written (no shared profile, no pointer)."""
        org_id = patch_agent_routes
        uid = str(USER_ID)

        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)

        assert isinstance(listing, AgentProfileListResponse)
        assert listing.profiles == []
        assert listing.active_agent_profile_id is None
        member = await _read_member(async_session_maker, org_id, USER_ID)
        assert member.active_agent_profile_id is None

    @pytest.mark.asyncio
    async def test_lists_saved_profiles_and_member_pointer(
        self, async_session_maker, patch_agent_routes
    ):
        org_id = patch_agent_routes
        uid = str(USER_ID)

        await save_agent_profile(
            name='reviewer',
            body={'llm_profile_ref': 'Default'},
            effective_org_id=org_id,
            user_id=uid,
        )
        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)

        assert [p.name for p in listing.profiles] == ['reviewer']
        assert listing.active_agent_profile_id is None

    @pytest.mark.asyncio
    async def test_falls_back_to_org_wide_active_pointer(
        self, async_session_maker, patch_agent_routes
    ):
        """A member with no pointer of their own resolves to the org-wide
        default (set explicitly via activate, not by any implicit write)."""
        org_id = patch_agent_routes
        uid = str(USER_ID)

        store = AgentProfiles()
        profile = save_profile_preserving_identity(
            store, OpenHandsAgentProfile(name='default', llm_profile_ref='Default')
        )
        object.__setattr__(store, 'active', str(profile.id))
        await _set_agent_profiles(async_session_maker, org_id, store)

        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)

        assert listing.active_agent_profile_id == str(profile.id)


# ── FK guard wired into the LLM-profile router ─────────────────────────────


@pytest.fixture
def patch_org_profile_routes(async_session_maker, seeded_org):
    async def _fake_get_org(org_id, user_id):  # noqa: ARG001
        async with async_session_maker() as session:
            result = await session.execute(select(Org).where(Org.id == org_id))
            return result.scalars().first()

    with (
        patch('server.routes.org_profiles.a_session_maker', async_session_maker),
        patch(
            'server.routes.org_profiles.OrgService.get_org_by_id',
            side_effect=_fake_get_org,
        ),
    ):
        yield seeded_org


async def _set_agent_profiles(async_session_maker, org_id, agent_profiles):
    async with async_session_maker() as session:
        org = (
            (await session.execute(select(Org).where(Org.id == org_id)))
            .scalars()
            .first()
        )
        org.agent_profiles = agent_profiles.model_dump(
            mode='json', context={'expose_secrets': True}
        )
        await session.commit()


async def _set_title_llm_profile(async_session_maker, org_id, name):
    async with async_session_maker() as session:
        await session.execute(
            update(OrgMember)
            .where(
                OrgMember.org_id == org_id,
                OrgMember.user_id == USER_ID,
            )
            .values(title_llm_profile=name)
        )
        await session.commit()


class TestLLMProfileFKGuard:
    @pytest.mark.asyncio
    async def test_delete_clears_member_title_profile_pointer(
        self, async_session_maker, patch_org_profile_routes
    ):
        org_id = patch_org_profile_routes
        await _set_title_llm_profile(async_session_maker, org_id, 'Default')

        await delete_profile(org_id=org_id, name='Default', user_id=str(USER_ID))

        member = await _read_member(async_session_maker, org_id, USER_ID)
        assert member.title_llm_profile is None

    @pytest.mark.asyncio
    async def test_delete_blocked_by_referencing_agent_profile(
        self, async_session_maker, patch_org_profile_routes
    ):
        org_id = patch_org_profile_routes
        # Reference the org's 'Default' LLM profile from an agent profile.
        ap = AgentProfiles()
        save_profile_preserving_identity(
            ap, OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        await _set_agent_profiles(async_session_maker, org_id, ap)

        with pytest.raises(HTTPException) as exc:
            await delete_profile(org_id=org_id, name='Default', user_id=str(USER_ID))
        assert exc.value.status_code == 409
        assert 'reviewer' in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_rename_cascades_to_agent_profile_ref(
        self, async_session_maker, patch_org_profile_routes
    ):
        org_id = patch_org_profile_routes
        ap = AgentProfiles()
        save_profile_preserving_identity(
            ap, OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        await _set_agent_profiles(async_session_maker, org_id, ap)
        await _set_title_llm_profile(async_session_maker, org_id, 'Default')

        await rename_profile(
            org_id=org_id,
            name='Default',
            request=RenameProfileRequest(new_name='Default-v2'),
            user_id=str(USER_ID),
        )

        async with async_session_maker() as session:
            org = (
                (await session.execute(select(Org).where(Org.id == org_id)))
                .scalars()
                .first()
            )
            reloaded = load_agent_profiles(org)
            assert reloaded.load('reviewer').llm_profile_ref == 'Default-v2'
            assert 'Default-v2' in (org.llm_profiles or {}).get('profiles', {})
        member = await _read_member(async_session_maker, org_id, USER_ID)
        assert member.title_llm_profile == 'Default-v2'


# ── SaasSettingsStore resolution + provenance ──────────────────────────────


class TestResolveActiveAgentProfile:
    def _store(self):
        with patch('storage.database.a_session_maker'):
            from storage.saas_settings_store import SaasSettingsStore
        return SaasSettingsStore(str(USER_ID))

    def _org_with(self, agent_profile):
        org = MagicMock(spec=Org)
        org.id = ORG_ID
        ap = AgentProfiles()
        save_profile_preserving_identity(ap, agent_profile)
        org.agent_profiles = ap.model_dump(
            mode='json', context={'expose_secrets': True}
        )
        org.llm_profiles = {
            'profiles': {'Default': {'model': 'gpt-4o', 'api_key': 'orgkey'}},
            'active': 'Default',
        }
        return org, next(iter(ap.profiles))

    def test_no_pointer_returns_none(self):
        store = self._store()
        org = MagicMock(spec=Org)
        org.id = ORG_ID
        member = MagicMock(spec=OrgMember)
        member.active_agent_profile_id = None
        assert store._resolve_active_agent_profile(org, member, {}, None) is None

    def test_falls_back_to_org_wide_active_pointer_when_member_pointer_unset(self):
        """A member with no per-member pointer of their own must still
        resolve to the org-wide default (``AgentProfiles.active``), if one
        happens to be set, rather than falling back to composed settings."""
        store = self._store()
        org, pid = self._org_with(
            OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        ap = AgentProfiles.model_validate(org.agent_profiles)
        ap.active = pid
        org.agent_profiles = ap.model_dump(
            mode='json', context={'expose_secrets': True}
        )

        member = MagicMock(spec=OrgMember)
        member.active_agent_profile_id = None

        result = store._resolve_active_agent_profile(org, member, {}, None)
        assert result is not None
        _dump, resolved_id, _revision = result
        assert resolved_id == pid

    def test_stale_pointer_falls_back_to_none(self):
        store = self._store()
        org = MagicMock(spec=Org)
        org.id = ORG_ID
        org.agent_profiles = None
        member = MagicMock(spec=OrgMember)
        member.active_agent_profile_id = str(uuid.uuid4())
        # Profile deleted out from under the pointer -> graceful None, no raise.
        assert store._resolve_active_agent_profile(org, member, {}, None) is None

    def test_resolves_openhands_profile_and_returns_provenance(self):
        store = self._store()
        org, pid = self._org_with(
            OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        member = MagicMock(spec=OrgMember)
        member.active_agent_profile_id = pid

        result = store._resolve_active_agent_profile(org, member, {}, None)
        assert result is not None
        dump, resolved_id, revision = result
        assert resolved_id == pid
        assert revision == 0
        assert dump['agent_kind'] == 'openhands'
        # The resolved LLM came from the referenced org LLM profile.
        assert dump['llm']['model'] == 'gpt-4o'

    def test_resolve_canonicalizes_legacy_litellm_proxy_llm(self):
        """A profile referencing an org LLM profile with a legacy
        ``litellm_proxy/`` managed name must resolve to the canonical
        ``openhands/`` name (proxy base_url dropped), matching the non-profile
        composed path so a profile launch and a plain launch normalize an
        org's pre-canonical llm_profiles identically."""
        from server.constants import LITE_LLM_API_URL

        store = self._store()
        org = MagicMock(spec=Org)
        org.id = ORG_ID
        ap = AgentProfiles()
        save_profile_preserving_identity(
            ap, OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        pid = next(iter(ap.profiles))
        org.agent_profiles = ap.model_dump(
            mode='json', context={'expose_secrets': True}
        )
        org.llm_profiles = {
            'profiles': {
                'Default': {
                    'model': 'litellm_proxy/claude-opus-4-8',
                    'base_url': LITE_LLM_API_URL,
                    'api_key': 'orgkey',
                }
            },
            'active': 'Default',
        }
        member = MagicMock(spec=OrgMember)
        member.active_agent_profile_id = pid

        result = store._resolve_active_agent_profile(org, member, {}, None)
        assert result is not None
        dump, _resolved_id, _revision = result
        assert dump['llm']['model'] == 'openhands/claude-opus-4-8'
        assert dump['llm'].get('base_url') is None

    def test_override_id_wins_over_member_pointer(self):
        store = self._store()
        org, pid = self._org_with(
            OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        member = MagicMock(spec=OrgMember)
        member.active_agent_profile_id = None  # no ambient pointer at all

        result = store._resolve_active_agent_profile(
            org, member, {}, None, override_agent_profile_id=pid
        )
        assert result is not None
        _dump, resolved_id, _revision = result
        assert resolved_id == pid

    def test_override_id_does_not_mutate_member_pointer(self):
        store = self._store()
        org, pid = self._org_with(
            OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        member = MagicMock(spec=OrgMember)
        member.active_agent_profile_id = None

        store._resolve_active_agent_profile(
            org, member, {}, None, override_agent_profile_id=pid
        )
        # The override must never be written back to the member's own pointer.
        assert member.active_agent_profile_id is None

    @pytest.mark.asyncio
    async def test_load_override_resolves_without_persisting_pointer(
        self, async_session_maker, patch_agent_routes
    ):
        """Loading with an explicit override_agent_profile_id resolves that
        profile's settings and stamps its id as provenance, but leaves
        org_member.active_agent_profile_id untouched in the database — the
        one-off launch override must never persist as the new default."""
        org_id = patch_agent_routes
        uid = str(USER_ID)

        await save_agent_profile(
            name='reviewer',
            body={'llm_profile_ref': 'Default'},
            effective_org_id=org_id,
            user_id=uid,
        )
        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)
        override_id = listing.profiles[0].id
        assert override_id is not None
        # The member has NO active pointer at all — the ambient default path
        # would return composed settings, not this profile.
        member_before = await _read_member(async_session_maker, org_id, USER_ID)
        assert member_before.active_agent_profile_id is None

        # Settings.enable_sound_notifications is non-nullable but the User
        # column defaults to NULL; seeded_org doesn't set it since no other
        # test in this file exercises a full load().
        async with async_session_maker() as session:
            user = await session.get(User, USER_ID)
            user.enable_sound_notifications = True
            await session.commit()

        from storage.saas_settings_store import SaasSettingsStore

        with (
            patch('storage.saas_settings_store.a_session_maker', async_session_maker),
            patch('storage.user_store.a_session_maker', async_session_maker),
            patch('storage.org_store.a_session_maker', async_session_maker),
        ):
            store = SaasSettingsStore(uid, effective_org_id=org_id)
            settings = await store.load(override_agent_profile_id=override_id)

        assert settings is not None
        assert settings.active_agent_profile_id == override_id

        member_after = await _read_member(async_session_maker, org_id, USER_ID)
        assert member_after.active_agent_profile_id is None


# ── Persisted vs resolved settings views ────────────────────────────────────


async def _read_org_raw(async_session_maker, org_id):
    async with async_session_maker() as session:
        result = await session.execute(select(Org).where(Org.id == org_id))
        return result.scalars().first()


MEMBER_MCP_SERVERS = {
    'a': {'url': 'https://a.example/mcp'},
    'b': {'url': 'https://b.example/mcp'},
    'c': {'url': 'https://c.example/mcp'},
}


class TestPersistedVsResolvedSettingsView:
    """Plain load() is the persisted view; resolution is a launch-only opt-in
    whose result store() refuses, so a profile's resolved dump (ref-filtered
    mcp_config, the referenced LLM profile's key) can never round-trip into
    the member/org rows."""

    async def _setup_active_profile(self, async_session_maker, org_id, mcp_server_refs):
        """Profile with the given refs, member pointed at it, member
        mcp_config with three servers, full load() viable."""
        ap = AgentProfiles()
        profile = save_profile_preserving_identity(
            ap,
            OpenHandsAgentProfile(
                name='reviewer',
                llm_profile_ref='Default',
                mcp_server_refs=mcp_server_refs,
            ),
        )
        await _set_agent_profiles(async_session_maker, org_id, ap)
        async with async_session_maker() as session:
            user = await session.get(User, USER_ID)
            user.enable_sound_notifications = True
            member = (
                (
                    await session.execute(
                        select(OrgMember).where(
                            OrgMember.org_id == org_id,
                            OrgMember.user_id == USER_ID,
                        )
                    )
                )
                .scalars()
                .first()
            )
            member.active_agent_profile_id = str(profile.id)
            member.agent_settings_diff = {
                'llm': {
                    'model': 'gpt-4o',
                    'base_url': 'https://api.openai.com/v1',
                },
            }
            member.mcp_config = dict(MEMBER_MCP_SERVERS)
            await session.commit()
        return profile

    def _store_patches(self, async_session_maker):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch('storage.saas_settings_store.a_session_maker', async_session_maker)
        )
        stack.enter_context(
            patch('storage.user_store.a_session_maker', async_session_maker)
        )
        stack.enter_context(
            patch('storage.org_store.a_session_maker', async_session_maker)
        )
        return stack

    @pytest.mark.asyncio
    @pytest.mark.parametrize('refs', [['a'], []])
    async def test_plain_load_round_trip_preserves_member_mcp_config(
        self, async_session_maker, patch_agent_routes, refs
    ):
        """F1: a routine load() -> store() round-trip (e.g. a settings PATCH
        touching an unrelated field) while a ref-filtering profile is active
        must not rewrite the member's mcp_config with the filtered view."""
        org_id = patch_agent_routes
        uid = str(USER_ID)
        await self._setup_active_profile(async_session_maker, org_id, refs)

        from storage.saas_settings_store import SaasSettingsStore

        with self._store_patches(async_session_maker):
            store = SaasSettingsStore(uid, effective_org_id=org_id)
            settings = await store.load()

            # The persisted view: no profile resolution happened.
            assert settings is not None
            assert settings.active_agent_profile_id is None
            assert settings.agent_settings.mcp_config is not None
            assert set(settings.agent_settings.mcp_config) == {
                'a',
                'b',
                'c',
            }

            settings.enable_sound_notifications = False  # the unrelated edit
            await store.store(settings)

        member = await _read_member(async_session_maker, org_id, USER_ID)
        stored_mcp = member.mcp_config or {}
        assert set(stored_mcp) == {'a', 'b', 'c'}

    @pytest.mark.asyncio
    async def test_plain_round_trip_keeps_member_llm_key(
        self, async_session_maker, patch_agent_routes
    ):
        """F2: a routine save while a profile is active must not overwrite the
        member's own LLM key with the referenced LLM profile's key."""
        from storage.encrypt_utils import decrypt_value

        org_id = patch_agent_routes
        uid = str(USER_ID)
        await self._setup_active_profile(async_session_maker, org_id, ['a'])

        from storage.saas_settings_store import SaasSettingsStore

        with self._store_patches(async_session_maker):
            store = SaasSettingsStore(uid, effective_org_id=org_id)
            settings = await store.load()
            assert settings is not None
            # The persisted view carries the member's effective key, not the
            # 'Default' LLM profile's key ('k').
            assert settings.agent_settings.llm.api_key is not None
            assert (
                settings.agent_settings.llm.api_key.get_secret_value() == 'initial-key'
            )

            settings.enable_sound_notifications = False
            await store.store(settings)

        member = await _read_member(async_session_maker, org_id, USER_ID)
        assert decrypt_value(member._llm_api_key) == 'initial-key'

    @pytest.mark.asyncio
    async def test_shared_agent_settings_edit_persists_while_profile_active(
        self, async_session_maker, patch_agent_routes
    ):
        """F3: with a profile active, org-level agent-settings edits made
        through the settings API must still persist (no silent write-drop)."""
        org_id = patch_agent_routes
        uid = str(USER_ID)
        await self._setup_active_profile(async_session_maker, org_id, ['a'])

        from storage.saas_settings_store import SaasSettingsStore

        with self._store_patches(async_session_maker):
            store = SaasSettingsStore(uid, effective_org_id=org_id)
            settings = await store.load()
            assert settings is not None
            settings.agent_settings = settings.agent_settings.model_copy(
                update={'enable_sub_agents': True}
            )
            await store.store(settings)

        org = await _read_org_raw(async_session_maker, org_id)
        assert (org.agent_settings or {}).get('enable_sub_agents') is True

    @pytest.mark.asyncio
    async def test_resolved_load_is_launch_view_and_store_refuses_it(
        self, async_session_maker, patch_agent_routes
    ):
        """resolve_agent_profile=True returns the resolved launch view (the
        profile replaces agent_settings) and store() refuses that object."""
        org_id = patch_agent_routes
        uid = str(USER_ID)
        profile = await self._setup_active_profile(async_session_maker, org_id, ['a'])

        from storage.saas_settings_store import SaasSettingsStore

        with self._store_patches(async_session_maker):
            store = SaasSettingsStore(uid, effective_org_id=org_id)
            settings = await store.load(resolve_agent_profile=True)

            assert settings is not None
            assert settings.active_agent_profile_id == str(profile.id)
            assert settings.active_agent_profile_revision == profile.revision
            # mcp_server_refs=['a'] filtered the member's three servers.
            assert settings.agent_settings.mcp_config is not None
            assert set(settings.agent_settings.mcp_config) == {'a'}
            # The resolved LLM is the referenced 'Default' org LLM profile.
            assert settings.agent_settings.llm.model == 'gpt-4o'

            with pytest.raises(ValueError, match='resolved Agent-Profile'):
                await store.store(settings)

    @pytest.mark.asyncio
    async def test_resolver_crash_falls_back_to_composed_settings(
        self, async_session_maker, patch_agent_routes
    ):
        """F5: an unexpected resolver exception (e.g. an SDK signature change
        raising TypeError) degrades to the composed settings, never a 500."""
        org_id = patch_agent_routes
        uid = str(USER_ID)
        await self._setup_active_profile(async_session_maker, org_id, ['a'])

        from storage.saas_settings_store import SaasSettingsStore

        with (
            self._store_patches(async_session_maker),
            patch(
                'storage.saas_settings_store.resolve_agent_profile',
                side_effect=TypeError(
                    'resolve_agent_profile() missing 1 required '
                    "keyword-only argument: 'available_skills'"
                ),
            ),
        ):
            store = SaasSettingsStore(uid, effective_org_id=org_id)
            settings = await store.load(resolve_agent_profile=True)

        assert settings is not None
        assert settings.active_agent_profile_id is None
        assert settings.agent_settings.mcp_config is not None
        assert set(settings.agent_settings.mcp_config) == {'a', 'b', 'c'}


# ── Best-effort load must not amplify into data loss ────────────────────────


class TestNoWriteBackWithoutMutation:
    @pytest.mark.asyncio
    async def test_activate_preserves_unparseable_profile(
        self, async_session_maker, patch_agent_routes
    ):
        """F4: /activate is pointer-only — it must not serialize the (best-
        effort loaded) collection back, which would silently erase a stored
        profile that merely failed to parse."""
        org_id = patch_agent_routes
        uid = str(USER_ID)

        await save_agent_profile(
            name='reviewer',
            body={'llm_profile_ref': 'Default'},
            effective_org_id=org_id,
            user_id=uid,
        )
        listing = await list_agent_profiles(effective_org_id=org_id, user_id=uid)
        valid_id = listing.profiles[0].id

        # Simulate schema drift: a stored entry the current model rejects
        # (name violates min_length) — _skip_invalid_profiles drops it on load.
        invalid_id = str(uuid.uuid4())
        org = await _read_org_raw(async_session_maker, org_id)
        blob_before = dict(org.agent_profiles)
        blob_before['profiles'] = {
            **blob_before['profiles'],
            invalid_id: {'name': '', 'agent_kind': 'openhands'},
        }
        async with async_session_maker() as session:
            org = (
                (await session.execute(select(Org).where(Org.id == org_id)))
                .scalars()
                .first()
            )
            org.agent_profiles = blob_before
            await session.commit()

        result = await activate_agent_profile(
            profile_id=valid_id, effective_org_id=org_id, user_id=uid
        )
        assert isinstance(result, ActivateAgentProfileResponse)

        member = await _read_member(async_session_maker, org_id, USER_ID)
        assert member.active_agent_profile_id == valid_id
        org = await _read_org_raw(async_session_maker, org_id)
        # The blob is byte-for-byte untouched: the unparseable entry survives.
        assert org.agent_profiles == blob_before
        assert invalid_id in org.agent_profiles['profiles']

    @pytest.mark.asyncio
    async def test_materialize_survives_dry_run_crash(
        self, async_session_maker, patch_agent_routes
    ):
        """F5 (router): a dry-run resolver crash surfaces as an invalid
        diagnostics report, not a 500."""
        org_id = patch_agent_routes
        uid = str(USER_ID)

        await save_agent_profile(
            name='reviewer',
            body={'llm_profile_ref': 'Default'},
            effective_org_id=org_id,
            user_id=uid,
        )

        with patch(
            'server.routes.agent_profiles.resolve_agent_profile_dry_run',
            side_effect=TypeError(
                'resolve_agent_profile_dry_run() missing 1 required '
                "keyword-only argument: 'available_skills'"
            ),
        ):
            diagnostics = await materialize_agent_profile(
                name='reviewer', effective_org_id=org_id, user_id=uid
            )

        assert diagnostics.valid is False
        assert any('Failed to resolve profile' in e for e in diagnostics.errors)


# ── Router permission-boundary integration (real HTTP, real Depends chain) ──
#
# Every other test above calls the route handler *functions* directly, so
# ``require_permission``/``EFFECTIVE_ORG_ID`` never actually run -- the tests
# supply ``user_id``/``effective_org_id`` as plain kwargs. These tests mount
# the real router behind a ``TestClient`` so the actual FastAPI dependency
# chain executes, proving the VIEW/EDIT_ORG_SETTINGS gate for real rather
# than by inspection (mirrors ``test_org_git_claims.py``'s pattern, which
# neither ``org_profiles.py`` nor ``agent_profiles.py`` had before).


@pytest.fixture
def agent_profiles_app():
    """Real app mounting the actual router. ``get_user_id`` is overridden
    (identity only); org resolution stays on the real code path but is
    pinned to ``ORG_ID`` on both sides (``EFFECTIVE_ORG_ID`` and
    ``require_permission``'s own no-path-org_id fallback) so they agree,
    same as a legitimate single-org request would. ``get_user_org_role`` is
    left unpatched here -- each test controls it to select the role under
    test.
    """
    from server.auth.org_context import resolve_effective_org_id

    app = FastAPI()
    app.include_router(agent_profiles_router)
    app.dependency_overrides[get_user_id] = lambda: str(USER_ID)
    app.dependency_overrides[resolve_effective_org_id] = lambda: ORG_ID
    with (
        patch(
            'server.auth.org_context.resolve_target_org_id_for_permission_check',
            AsyncMock(return_value=ORG_ID),
        ),
        patch(
            'server.auth.authorization.get_user_super_role',
            AsyncMock(return_value=None),
        ),
    ):
        yield app


def _role(name):
    role = MagicMock()
    role.name = name
    return role


class TestAgentProfilesRouterAuthorizationBoundary:
    def test_member_gets_403_on_save(self, agent_profiles_app):
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=_role('member')),
        ):
            client = TestClient(agent_profiles_app)
            response = client.post(
                '/api/agent-profiles/reviewer', json={'llm_profile_ref': 'Default'}
            )
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'edit_org_settings' in response.json()['detail'].lower()

    def test_member_gets_403_on_delete(self, agent_profiles_app):
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=_role('member')),
        ):
            client = TestClient(agent_profiles_app)
            response = client.delete('/api/agent-profiles/reviewer')
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'edit_org_settings' in response.json()['detail'].lower()

    def test_member_gets_403_on_rename(self, agent_profiles_app):
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=_role('member')),
        ):
            client = TestClient(agent_profiles_app)
            response = client.post(
                '/api/agent-profiles/reviewer/rename', json={'new_name': 'renamed'}
            )
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'edit_org_settings' in response.json()['detail'].lower()

    def test_non_member_gets_403_on_list(self, agent_profiles_app):
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=None),
        ):
            client = TestClient(agent_profiles_app)
            response = client.get('/api/agent-profiles')
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'not a member' in response.json()['detail'].lower()

    def test_non_member_gets_403_on_save(self, agent_profiles_app):
        """A non-admin lacking even VIEW_ORG_SETTINGS must not reach the
        EDIT_ORG_SETTINGS check via some other path either."""
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=None),
        ):
            client = TestClient(agent_profiles_app)
            response = client.post(
                '/api/agent-profiles/reviewer', json={'llm_profile_ref': 'Default'}
            )
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'not a member' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_member_can_list_profiles_200(
        self, agent_profiles_app, async_session_maker, patch_agent_routes
    ):
        """A MEMBER has VIEW_ORG_SETTINGS -- the read path must stay open;
        the fix for the write-side gap must not overshoot into blocking
        legitimate reads."""
        org_id = patch_agent_routes
        ap = AgentProfiles()
        save_profile_preserving_identity(
            ap, OpenHandsAgentProfile(name='reviewer', llm_profile_ref='Default')
        )
        await _set_agent_profiles(async_session_maker, org_id, ap)

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=_role('member')),
        ):
            client = TestClient(agent_profiles_app)
            response = client.get('/api/agent-profiles')

        assert response.status_code == status.HTTP_200_OK
        assert [p['name'] for p in response.json()['profiles']] == ['reviewer']

    @pytest.mark.asyncio
    async def test_admin_can_save_profile_201(
        self, agent_profiles_app, async_session_maker, patch_agent_routes
    ):
        """An ADMIN has EDIT_ORG_SETTINGS -- the write path must stay open."""
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=_role('admin')),
        ):
            client = TestClient(agent_profiles_app)
            response = client.post(
                '/api/agent-profiles/reviewer', json={'llm_profile_ref': 'Default'}
            )

        assert response.status_code == status.HTTP_201_CREATED
