"""Unit and integration tests for organization LLM profiles router."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role
from storage.user import User

from openhands.app_server.settings.llm_profiles import (
    MAX_PROFILES_PER_USER,
    LLMProfiles,
    StrictLLM,
)
from openhands.app_server.utils.llm import MASKED_API_KEY

# Mock the database module before importing the router — matches the
# test_saas_settings_store.py pattern so module-level imports don't try to
# touch a real engine.
with patch('storage.database.a_session_maker'):
    from server.routes.org_profiles import (
        ProfileListResponse,
        ProfileMutationResponse,
        RenameProfileRequest,
        SaveProfileRequest,
        _load_profiles,
        activate_profile,
        delete_profile,
        get_profile,
        list_profiles,
        rename_profile,
        save_profile,
    )


@pytest.fixture
def sample_org():
    """Create a sample org for testing."""
    org = MagicMock(spec=Org)
    org.id = uuid.uuid4()
    org.llm_profiles = None
    org.agent_settings = {'llm': {'model': 'gpt-4', 'api_key': 'test-key'}}
    return org


@pytest.fixture
def org_with_profiles():
    """Create an org with existing profiles."""
    org = MagicMock(spec=Org)
    org.id = uuid.uuid4()
    org.llm_profiles = {
        'profiles': {
            'my-profile': {'model': 'claude-3', 'api_key': 'claude-key'},
            'backup': {'model': 'gpt-4o', 'api_key': 'openai-key'},
        },
        'active': 'my-profile',
    }
    org.agent_settings = {'llm': {'model': 'gpt-4', 'api_key': 'test-key'}}
    return org


class TestLoadProfiles:
    """Test the _load_profiles helper function."""

    def test_load_profiles_empty(self, sample_org):
        """Test loading profiles when org has none."""
        profiles = _load_profiles(sample_org)
        assert isinstance(profiles, LLMProfiles)
        assert profiles.active is None
        assert len(profiles.summaries()) == 0

    def test_load_profiles_with_data(self, org_with_profiles):
        """Test loading profiles when org has existing profiles."""
        profiles = _load_profiles(org_with_profiles)
        assert isinstance(profiles, LLMProfiles)
        assert profiles.active == 'my-profile'
        summaries = profiles.summaries()
        assert len(summaries) == 2
        names = [s['name'] for s in summaries]
        assert 'my-profile' in names
        assert 'backup' in names

    def test_load_profiles_invalid_data(self, sample_org):
        """Test loading profiles when org has invalid data."""
        sample_org.llm_profiles = {'invalid': 'data'}
        profiles = _load_profiles(sample_org)
        # Should return empty profiles on parse error
        assert isinstance(profiles, LLMProfiles)


class TestProfileListResponse:
    """Test ProfileListResponse model."""

    def test_empty_list(self):
        """Test empty profile list response."""
        response = ProfileListResponse(profiles=[], active_profile=None)
        assert response.profiles == []
        assert response.active_profile is None

    def test_with_profiles(self):
        """Test profile list response with data."""
        from server.routes.org_profiles import ProfileInfo

        profiles = [
            ProfileInfo(name='test', model='gpt-4', base_url=None, api_key_set=True),
        ]
        response = ProfileListResponse(profiles=profiles, active_profile='test')
        assert len(response.profiles) == 1
        assert response.active_profile == 'test'


class TestProfileMutationResponse:
    """Test ProfileMutationResponse model."""

    def test_create_response(self):
        """Test creating a mutation response."""
        response = ProfileMutationResponse(
            name='new-profile', message="Profile 'new-profile' saved"
        )
        assert response.name == 'new-profile'
        assert 'saved' in response.message


class TestSaveProfileRequest:
    """Test SaveProfileRequest model."""

    def test_default_values(self):
        """Test default values for save request."""
        request = SaveProfileRequest()
        assert request.include_secrets is True
        assert request.llm is None

    def test_with_llm(self):
        """Test save request with LLM config."""
        request = SaveProfileRequest(
            include_secrets=False, llm=StrictLLM(model='gpt-4')
        )
        assert request.include_secrets is False
        assert request.llm is not None
        assert request.llm.model == 'gpt-4'


class TestRenameProfileRequest:
    """Test RenameProfileRequest model."""

    def test_valid_name(self):
        """Test valid rename request."""
        request = RenameProfileRequest(new_name='new-name')
        assert request.new_name == 'new-name'

    def test_name_validation(self):
        """Test name length validation."""
        # Should accept reasonable names
        request = RenameProfileRequest(new_name='a' * 100)
        assert len(request.new_name) == 100

        # Should reject empty names (min_length=1)
        with pytest.raises(ValueError):
            RenameProfileRequest(new_name='')

        # Should reject too-long names (max_length=100)
        with pytest.raises(ValueError):
            RenameProfileRequest(new_name='a' * 101)


# ── Integration tests ──────────────────────────────────────────────────────
#
# Exercise the route handlers end-to-end against a real SQLite-backed Org +
# OrgMember row. They verify the new ``SELECT FOR UPDATE`` transaction helper
# round-trips correctly, the activate handler writes both the org marker and
# the member diff atomically, and the exception-to-HTTP mapping for the
# 404/409 paths.

ORG_ID = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')
ADMIN_USER_ID = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5082')


@pytest.fixture
def seeded_org(session_maker):
    """Create a personal-org-shaped row with one admin member."""
    with session_maker() as session:
        session.add(Role(id=10, name='member', rank=3))
        session.add(
            Org(
                id=ORG_ID,
                name='profile-test-org',
                org_version=1,
                enable_proactive_conversation_starters=True,
            )
        )
        session.add(
            User(
                id=ADMIN_USER_ID,
                current_org_id=ORG_ID,
                user_consents_to_analytics=True,
            )
        )
        session.add(
            OrgMember(
                org_id=ORG_ID,
                user_id=ADMIN_USER_ID,
                role_id=10,
                llm_api_key='initial-key',
                agent_settings_diff={
                    'llm': {'model': 'before-activate', 'base_url': None},
                },
                conversation_settings_diff={},
                status='active',
            )
        )
        session.commit()
    return {'org_id': ORG_ID, 'admin_user_id': ADMIN_USER_ID}


@pytest.fixture
def patch_route_db(async_session_maker, seeded_org):
    """Wire the router's db session + OrgService.get_org_by_id to the test
    SQLite fixture so direct handler calls hit the real schema. ``get_org_by_id``
    is patched (rather than seeding the full membership graph) because its
    inner OrgMemberStore call opens sessions outside ``async_session_maker``.
    """
    org_id = seeded_org['org_id']

    async def _fake_get_org(org_id, user_id):  # noqa: ARG001
        async with async_session_maker() as session:
            result = await session.execute(select(Org).where(Org.id == org_id))
            return result.scalars().first()

    with (
        patch(
            'server.routes.org_profiles.a_session_maker',
            async_session_maker,
        ),
        patch(
            'server.routes.org_profiles.OrgService.get_org_by_id',
            side_effect=_fake_get_org,
        ),
    ):
        yield org_id


async def _read_org(async_session_maker, org_id):
    async with async_session_maker() as session:
        result = await session.execute(select(Org).where(Org.id == org_id))
        return result.scalars().first()


async def _read_member(async_session_maker, org_id, user_id):
    # ``user_id`` accepts either str or UUID — coerce so the SQLite test
    # backend's strict Uuid binding doesn't error on str inputs.
    user_uuid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(user_id)
    async with async_session_maker() as session:
        result = await session.execute(
            select(OrgMember).where(
                OrgMember.org_id == org_id, OrgMember.user_id == user_uuid
            )
        )
        return result.scalars().first()


class TestProfileLifecycleIntegration:
    """Round-trip CRUD against a real Org row."""

    @pytest.mark.asyncio
    async def test_save_then_list_persists_profile(
        self, async_session_maker, patch_route_db
    ):
        org_id = patch_route_db

        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(model='anthropic/claude-3-5-sonnet')
            ),
            user_id=str(ADMIN_USER_ID),
        )

        listing = await list_profiles(org_id=org_id, user_id=str(ADMIN_USER_ID))
        assert [p.name for p in listing.profiles] == ['work']
        assert listing.profiles[0].model == 'anthropic/claude-3-5-sonnet'
        assert listing.active_profile is None

    @pytest.mark.asyncio
    async def test_get_profile_returns_details_without_secret(
        self, async_session_maker, patch_route_db
    ):
        org_id = patch_route_db

        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(
                    model='anthropic/claude-3-5-sonnet',
                    api_key='secret-value',
                )
            ),
            user_id=str(ADMIN_USER_ID),
        )

        detail = await get_profile(
            org_id=org_id, name='work', user_id=str(ADMIN_USER_ID)
        )
        assert detail.name == 'work'
        assert detail.llm['model'] == 'anthropic/claude-3-5-sonnet'
        # ``expose_secrets=False`` on the response masks the secret rather
        # than dropping it (so the response shape is stable). Just confirm
        # the raw value never leaks back.
        assert detail.llm.get('api_key') != 'secret-value'

    @pytest.mark.asyncio
    async def test_delete_removes_profile_and_repeat_returns_404(
        self, async_session_maker, patch_route_db
    ):
        org_id = patch_route_db

        await save_profile(
            org_id=org_id,
            name='gone',
            request=SaveProfileRequest(llm=StrictLLM(model='openai/gpt-4o')),
            user_id=str(ADMIN_USER_ID),
        )
        await delete_profile(org_id=org_id, name='gone', user_id=str(ADMIN_USER_ID))

        listing = await list_profiles(org_id=org_id, user_id=str(ADMIN_USER_ID))
        assert listing.profiles == []

        with pytest.raises(HTTPException) as exc:
            await delete_profile(org_id=org_id, name='gone', user_id=str(ADMIN_USER_ID))
        assert exc.value.status_code == 404


class TestProfileErrorPaths:
    """Exception-to-HTTP mapping for the mutating endpoints."""

    @pytest.mark.asyncio
    async def test_save_raises_409_at_profile_limit(
        self, async_session_maker, patch_route_db
    ):
        org_id = patch_route_db
        # Fill to the cap, then attempt a new name.
        for i in range(MAX_PROFILES_PER_USER):
            await save_profile(
                org_id=org_id,
                name=f'p{i}',
                request=SaveProfileRequest(llm=StrictLLM(model=f'openai/m-{i}')),
                user_id=str(ADMIN_USER_ID),
            )

        with pytest.raises(HTTPException) as exc:
            await save_profile(
                org_id=org_id,
                name='overflow',
                request=SaveProfileRequest(llm=StrictLLM(model='openai/m-x')),
                user_id=str(ADMIN_USER_ID),
            )
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_rename_raises_409_on_name_collision(
        self, async_session_maker, patch_route_db
    ):
        org_id = patch_route_db

        for name in ('a', 'b'):
            await save_profile(
                org_id=org_id,
                name=name,
                request=SaveProfileRequest(llm=StrictLLM(model=f'openai/{name}')),
                user_id=str(ADMIN_USER_ID),
            )

        with pytest.raises(HTTPException) as exc:
            await rename_profile(
                org_id=org_id,
                name='a',
                request=RenameProfileRequest(new_name='b'),
                user_id=str(ADMIN_USER_ID),
            )
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_activate_raises_404_for_unknown_profile(
        self, async_session_maker, patch_route_db
    ):
        org_id = patch_route_db

        with pytest.raises(HTTPException) as exc:
            await activate_profile(
                org_id=org_id, name='missing', user_id=str(ADMIN_USER_ID)
            )
        assert exc.value.status_code == 404


async def _set_org_agent_settings(async_session_maker, org_id, agent_settings):
    async with async_session_maker() as session:
        result = await session.execute(select(Org).where(Org.id == org_id))
        org = result.scalars().first()
        org.agent_settings = agent_settings
        await session.commit()


class TestSaveApiKeyPreservation:
    """No-new-key saves must not clobber a profile's stored api_key."""

    @pytest.mark.asyncio
    async def test_snapshot_save_with_preserve_flag_keeps_existing_key(
        self, async_session_maker, patch_route_db
    ):
        """The UI edit-save snapshots org defaults (active key included);
        the flag keeps the profile's own key while the snapshot's model lands.
        """
        org_id = patch_route_db
        await _set_org_agent_settings(
            async_session_maker,
            org_id,
            {'llm': {'model': 'openai/gpt-4o', 'api_key': 'org-active-key'}},
        )
        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(
                    model='anthropic/claude-3-5-sonnet', api_key='profile-key'
                )
            ),
            user_id=str(ADMIN_USER_ID),
        )

        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(preserve_existing_api_key=True),
            user_id=str(ADMIN_USER_ID),
        )

        org = await _read_org(async_session_maker, org_id)
        saved = _load_profiles(org).get('work')
        assert saved.model == 'openai/gpt-4o'
        assert saved.api_key.get_secret_value() == 'profile-key'

    @pytest.mark.asyncio
    async def test_snapshot_save_without_flag_keeps_snapshot_key(
        self, async_session_maker, patch_route_db
    ):
        """Counter-test: a plain snapshot save still captures the org key."""
        org_id = patch_route_db
        await _set_org_agent_settings(
            async_session_maker,
            org_id,
            {'llm': {'model': 'openai/gpt-4o', 'api_key': 'org-active-key'}},
        )
        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(
                    model='anthropic/claude-3-5-sonnet', api_key='profile-key'
                )
            ),
            user_id=str(ADMIN_USER_ID),
        )

        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(),
            user_id=str(ADMIN_USER_ID),
        )

        org = await _read_org(async_session_maker, org_id)
        saved = _load_profiles(org).get('work')
        assert saved.api_key.get_secret_value() == 'org-active-key'

    @pytest.mark.asyncio
    async def test_snapshot_save_with_preserve_flag_keeps_profile_keyless(
        self, async_session_maker, patch_route_db
    ):
        """A keyless profile must not silently inherit the org's active key."""
        org_id = patch_route_db
        await _set_org_agent_settings(
            async_session_maker,
            org_id,
            {'llm': {'model': 'openai/gpt-4o', 'api_key': 'org-active-key'}},
        )
        await save_profile(
            org_id=org_id,
            name='keyless',
            request=SaveProfileRequest(llm=StrictLLM(model='openai/gpt-4o')),
            user_id=str(ADMIN_USER_ID),
        )

        await save_profile(
            org_id=org_id,
            name='keyless',
            request=SaveProfileRequest(preserve_existing_api_key=True),
            user_id=str(ADMIN_USER_ID),
        )

        org = await _read_org(async_session_maker, org_id)
        assert _load_profiles(org).get('keyless').api_key is None

    @pytest.mark.asyncio
    async def test_explicit_llm_without_key_preserves_stored_key(
        self, async_session_maker, patch_route_db
    ):
        """GET→edit→POST round-trips null the key; the update must keep the
        stored one (parity with the personal profiles route)."""
        org_id = patch_route_db
        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(model='openai/gpt-4o', api_key='stored-key')
            ),
            user_id=str(ADMIN_USER_ID),
        )

        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(model='anthropic/claude-3-5-sonnet')
            ),
            user_id=str(ADMIN_USER_ID),
        )

        org = await _read_org(async_session_maker, org_id)
        saved = _load_profiles(org).get('work')
        assert saved.model == 'anthropic/claude-3-5-sonnet'
        assert saved.api_key.get_secret_value() == 'stored-key'

    @pytest.mark.asyncio
    async def test_explicit_llm_with_new_key_replaces_stored_key(
        self, async_session_maker, patch_route_db
    ):
        """Counter-test: an intentionally supplied key still wins."""
        org_id = patch_route_db
        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(model='openai/gpt-4o', api_key='old-key')
            ),
            user_id=str(ADMIN_USER_ID),
        )

        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(model='openai/gpt-4o', api_key='new-key')
            ),
            user_id=str(ADMIN_USER_ID),
        )

        org = await _read_org(async_session_maker, org_id)
        assert _load_profiles(org).get('work').api_key.get_secret_value() == 'new-key'


class TestActivateTransactionAtomicity:
    """Activate must commit the org marker and the member diff together."""

    @pytest.mark.asyncio
    async def test_writes_org_active_and_member_diff_together(
        self, async_session_maker, patch_route_db
    ):
        org_id = patch_route_db
        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(
                llm=StrictLLM(
                    model='anthropic/claude-3-5-sonnet',
                    base_url='https://api.anthropic.com/v1',
                    api_key='activate-key',
                )
            ),
            user_id=str(ADMIN_USER_ID),
        )

        await activate_profile(org_id=org_id, name='work', user_id=str(ADMIN_USER_ID))

        org = await _read_org(async_session_maker, org_id)
        assert org is not None
        assert _load_profiles(org).active == 'work'

        member = await _read_member(async_session_maker, org_id, ADMIN_USER_ID)
        assert member is not None
        assert (
            member.agent_settings_diff['llm']['model'] == 'anthropic/claude-3-5-sonnet'
        )

    @pytest.mark.asyncio
    async def test_member_lookup_failure_rolls_back_org_active(
        self, async_session_maker, patch_route_db
    ):
        """If the member row vanishes between perm-check and the same-session
        member read, the helper must roll back the org-side update so we
        don't end up with the org marker advanced without the member diff
        applied.
        """
        org_id = patch_route_db
        await save_profile(
            org_id=org_id,
            name='work',
            request=SaveProfileRequest(llm=StrictLLM(model='openai/gpt-4o')),
            user_id=str(ADMIN_USER_ID),
        )
        # Sanity: no active profile yet.
        org_before = await _read_org(async_session_maker, org_id)
        assert _load_profiles(org_before).active is None

        # Drop the member row so the in-transaction lookup raises 404.
        async with async_session_maker() as session:
            result = await session.execute(
                select(OrgMember).where(OrgMember.org_id == org_id)
            )
            member = result.scalars().first()
            await session.delete(member)
            await session.commit()

        with pytest.raises(HTTPException) as exc:
            await activate_profile(
                org_id=org_id, name='work', user_id=str(ADMIN_USER_ID)
            )
        assert exc.value.status_code == 404

        org_after = await _read_org(async_session_maker, org_id)
        # No commit happened → org.active stays None.
        assert _load_profiles(org_after).active is None

    @pytest.mark.asyncio
    async def test_activate_masks_key_in_diff_and_persists_real_key_encrypted(
        self, async_session_maker, patch_route_db
    ):
        """A per-profile key must never be stored raw in the plain-JSON
        ``agent_settings_diff`` column, and it must actually take effect: the
        effective key resolves from the encrypted ``_llm_api_key`` column (via
        ``has_custom_llm_api_key``), not from the diff.
        """
        org_id = patch_route_db
        await save_profile(
            org_id=org_id,
            name='byor',
            request=SaveProfileRequest(
                llm=StrictLLM(
                    model='anthropic/claude-3-5-sonnet',
                    base_url='https://api.anthropic.com/v1',
                    api_key='byor-secret',
                )
            ),
            user_id=str(ADMIN_USER_ID),
        )

        await activate_profile(org_id=org_id, name='byor', user_id=str(ADMIN_USER_ID))

        member = await _read_member(async_session_maker, org_id, ADMIN_USER_ID)
        assert member is not None
        # The raw key never lands in the unencrypted diff column.
        assert member.agent_settings_diff['llm']['api_key'] == MASKED_API_KEY
        assert 'byor-secret' not in str(member.agent_settings_diff)
        # A non-managed (BYOR) key takes effect via the encrypted member store.
        assert member.has_custom_llm_api_key is True
        assert member.llm_api_key.get_secret_value() == 'byor-secret'
