"""Unit tests for SaasUserAuth.get_effective_org_id().

Validates the precedence rules:

1. ``api_key_org_id`` (API key binding) — cannot be overridden; an
   ``X-Org-Id`` header that disagrees produces a 403.
2. ``X-Org-Id`` header — validated against the user's org memberships;
   non-member raises 403, malformed UUID raises 400.
3. ``user.current_org_id`` — fallback when neither of the above is set.
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role
from storage.user import User


@pytest.fixture
async def async_engine():
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
    )
    return engine


@pytest.fixture
async def async_session_maker(async_engine):
    session_maker = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return session_maker


@pytest.fixture
def user_id():
    return str(uuid.uuid4())


@pytest.fixture
def org_id():
    return uuid.uuid4()


@pytest.fixture
def other_org_id():
    return uuid.uuid4()


async def _seed_minimal(
    session_maker,
    user_id_str,
    current_org_id,
    *extra_org_ids,
    super_role_name: str | None = None,
):
    """Create a role, org(s), user, and org_member rows in the in-memory DB.

    When ``super_role_name`` is provided, also seed a separate role row
    with that name and assign it to the user via ``User.role_id``. This
    is what ``get_user_super_role`` consults; any non-None value flips
    the user into a "super" role for tests that exercise the super-role
    bypass in :meth:`SaasUserAuth._resolve_org_id`.
    """
    async with session_maker() as session:
        role = Role(name='member', rank=3)
        session.add(role)
        await session.flush()

        super_role_id: int | None = None
        if super_role_name is not None:
            super_role = Role(name=super_role_name, rank=0)
            session.add(super_role)
            await session.flush()
            super_role_id = super_role.id

        all_org_ids = [current_org_id, *extra_org_ids]
        for o in all_org_ids:
            session.add(
                Org(
                    id=o,
                    name=f'Org {o}',
                    org_version=1,
                    enable_proactive_conversation_starters=True,
                )
            )
        await session.flush()

        session.add(
            User(
                id=uuid.UUID(user_id_str),
                current_org_id=current_org_id,
                user_consents_to_analytics=True,
                role_id=super_role_id,
            )
        )
        await session.flush()

        for o in all_org_ids:
            session.add(
                OrgMember(
                    org_id=o,
                    user_id=uuid.UUID(user_id_str),
                    role_id=role.id,
                    status='active',
                    llm_api_key='test-api-key',
                )
            )
        await session.commit()


def _stores_patched(async_session_maker):
    """Return the standard set of patches used by SaasUserAuth helpers."""
    return (
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
        patch('storage.org_member_store.a_session_maker', async_session_maker),
        patch('storage.role_store.a_session_maker', async_session_maker),
    )


class TestGetEffectiveOrgId:
    @pytest.mark.asyncio
    async def test_no_header_falls_back_to_current_org_id(
        self, async_session_maker, user_id, org_id
    ):
        await _seed_minimal(async_session_maker, user_id, org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
        )
        with (
            _stores_patched(async_session_maker)[0],
            _stores_patched(async_session_maker)[2],
        ):
            effective = await user_auth.get_effective_org_id()

        assert effective == org_id

    @pytest.mark.asyncio
    async def test_header_overrides_when_user_is_member(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        await _seed_minimal(async_session_maker, user_id, org_id, other_org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header=str(other_org_id),
        )
        with (
            _stores_patched(async_session_maker)[0],
            _stores_patched(async_session_maker)[2],
        ):
            effective = await user_auth.get_effective_org_id()

        assert effective == other_org_id

    @pytest.mark.asyncio
    async def test_server_side_override_wins_when_user_is_member(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        await _seed_minimal(async_session_maker, user_id, org_id, other_org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            effective_org_id_override=other_org_id,
        )
        with _stores_patched(async_session_maker)[2]:
            effective = await user_auth.get_effective_org_id()

        assert effective == other_org_id

    @pytest.mark.asyncio
    async def test_server_side_override_with_non_member_org_raises_403(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        await _seed_minimal(async_session_maker, user_id, org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            effective_org_id_override=other_org_id,
        )
        with _stores_patched(async_session_maker)[2]:
            with pytest.raises(HTTPException) as exc_info:
                await user_auth.get_effective_org_id()

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_server_side_override_non_member_403_even_for_super_user(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        """The override path is for trusted server-side resolver code.

        Its membership check is defense-in-depth against buggy resolver
        contexts setting a wrong override; the super-role bypass only
        applies to the client-driven ``X-Org-Id`` header. This test
        pins that design so the two paths can't drift.
        """
        await _seed_minimal(
            async_session_maker,
            user_id,
            org_id,
            super_role_name='owner',
        )
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            effective_org_id_override=other_org_id,
        )
        with _stores_patched(async_session_maker)[2]:
            with pytest.raises(HTTPException) as exc_info:
                await user_auth.get_effective_org_id()

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_server_side_override_api_key_org_mismatch_raises_403(
        self, user_id, org_id, other_org_id
    ):
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            api_key_org_id=org_id,
            effective_org_id_override=other_org_id,
        )
        with pytest.raises(HTTPException) as exc_info:
            await user_auth.get_effective_org_id()

        assert exc_info.value.status_code == 403

    def test_set_effective_org_id_override_clears_org_scoped_caches(
        self, user_id, org_id
    ):
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
        )
        user_auth._effective_org_id = uuid.uuid4()
        user_auth._effective_org_id_resolved = True
        user_auth.settings_store = object()
        user_auth.secrets_store = object()
        user_auth._settings = object()
        user_auth._resolved_settings = object()
        user_auth._secrets = object()
        user_auth.provider_tokens = {}
        user_auth._org_id = 'old-org'
        user_auth._org_name = 'Old Org'
        user_auth._role = 'member'
        user_auth._permissions = ['read']
        user_auth._org_info_loaded = True

        user_auth.set_effective_org_id_override(org_id)

        assert user_auth.effective_org_id_override == org_id
        assert user_auth._effective_org_id is None
        assert user_auth._effective_org_id_resolved is False
        assert user_auth.settings_store is None
        assert user_auth.secrets_store is None
        assert user_auth._settings is None
        # The resolved launch view is org-scoped (referenced LLM key +
        # ref-filtered mcp_config) and must not survive a re-scope to another
        # org, or org B would read org A's resolved credentials.
        assert user_auth._resolved_settings is None
        assert user_auth._secrets is None
        assert user_auth.provider_tokens is None
        assert user_auth._org_id is None
        assert user_auth._org_name is None
        assert user_auth._role is None
        assert user_auth._permissions is None
        assert user_auth._org_info_loaded is False

    @pytest.mark.asyncio
    async def test_header_with_non_member_org_raises_403(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        # Seed the user as a member of `org_id` only, not `other_org_id`.
        await _seed_minimal(async_session_maker, user_id, org_id)
        # `other_org_id` org exists but the user isn't a member.
        async with async_session_maker() as session:
            session.add(
                Org(
                    id=other_org_id,
                    name='Outside Org',
                    org_version=1,
                    enable_proactive_conversation_starters=True,
                )
            )
            await session.commit()

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header=str(other_org_id),
        )
        with (
            _stores_patched(async_session_maker)[0],
            _stores_patched(async_session_maker)[2],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await user_auth.get_effective_org_id()

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_header_non_member_super_role_bypasses_membership_check(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        """Regression for the super-role / EFFECTIVE_ORG_ID inconsistency.

        A user with a super role (``user.role_id`` set) that targets a
        non-member org via ``X-Org-Id`` must get the org id back. The
        route's ``require_permission`` dependency is the authoritative
        check for the actual permission; this resolver should not 403
        such requests at the coarse membership gate.
        """
        # Seed user as member of ``org_id`` and as a super-role holder.
        await _seed_minimal(
            async_session_maker,
            user_id,
            org_id,
            super_role_name='owner',
        )
        # Add ``other_org_id`` as a real org the user is *not* a member of.
        async with async_session_maker() as session:
            session.add(
                Org(
                    id=other_org_id,
                    name='Outside Org',
                    org_version=1,
                    enable_proactive_conversation_starters=True,
                )
            )
            await session.commit()

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header=str(other_org_id),
        )
        with (
            _stores_patched(async_session_maker)[0],
            _stores_patched(async_session_maker)[2],
            _stores_patched(async_session_maker)[3],
        ):
            effective = await user_auth.get_effective_org_id()

        assert effective == other_org_id

    @pytest.mark.asyncio
    async def test_header_non_member_without_super_role_still_403(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        """Ensure the super-role bypass is opt-in: regular users still 403.

        Mirrors :meth:`test_header_with_non_member_org_raises_403` but
        is named to make the intent of the bypass explicit -- the gate
        is only relaxed when ``user.role_id`` is set.
        """
        await _seed_minimal(async_session_maker, user_id, org_id)
        async with async_session_maker() as session:
            session.add(
                Org(
                    id=other_org_id,
                    name='Outside Org',
                    org_version=1,
                    enable_proactive_conversation_starters=True,
                )
            )
            await session.commit()

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header=str(other_org_id),
        )
        with (
            _stores_patched(async_session_maker)[0],
            _stores_patched(async_session_maker)[2],
            _stores_patched(async_session_maker)[3],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await user_auth.get_effective_org_id()

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_malformed_header_raises_400(self, async_session_maker, user_id):
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header='not-a-uuid',
        )
        with pytest.raises(HTTPException) as exc_info:
            await user_auth.get_effective_org_id()

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_api_key_org_id_wins_over_user_current_org(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        # User's persisted current_org is `org_id`, but API key is pinned
        # to `other_org_id`. Effective org must be `other_org_id`.
        await _seed_minimal(async_session_maker, user_id, org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            api_key_org_id=other_org_id,
        )
        effective = await user_auth.get_effective_org_id()
        assert effective == other_org_id

    @pytest.mark.asyncio
    async def test_api_key_org_id_mismatch_with_header_raises_403(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            api_key_org_id=org_id,
            _x_org_id_header=str(other_org_id),
        )
        with pytest.raises(HTTPException) as exc_info:
            await user_auth.get_effective_org_id()

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_api_key_org_id_matching_header_is_allowed(
        self, async_session_maker, user_id, org_id
    ):
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            api_key_org_id=org_id,
            _x_org_id_header=str(org_id),
        )
        effective = await user_auth.get_effective_org_id()
        assert effective == org_id

    @pytest.mark.asyncio
    async def test_result_is_cached_across_calls(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        await _seed_minimal(async_session_maker, user_id, org_id, other_org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header=str(other_org_id),
        )
        with (
            _stores_patched(async_session_maker)[0],
            _stores_patched(async_session_maker)[2],
        ):
            first = await user_auth.get_effective_org_id()

        # Drop the patches; if cache works the second call must not touch DB.
        second = await user_auth.get_effective_org_id()
        assert first == second == other_org_id
        assert user_auth._effective_org_id_resolved is True


class TestGetTargetOrgIdForPermissionCheck:
    """Tests for ``SaasUserAuth.get_target_org_id_for_permission_check``.

    Resolution must mirror ``get_effective_org_id`` for API-key binding
    and header parsing, but **must not** verify that the user is an
    ``org_member`` of the resolved org -- super-role users need to be
    able to target orgs they have not joined.
    """

    @pytest.mark.asyncio
    async def test_header_targets_non_member_org_without_403(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        # Seed the user as a member of `org_id` only, not `other_org_id`.
        await _seed_minimal(async_session_maker, user_id, org_id)
        async with async_session_maker() as session:
            session.add(
                Org(
                    id=other_org_id,
                    name='Outside Org',
                    org_version=1,
                    enable_proactive_conversation_starters=True,
                )
            )
            await session.commit()

        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header=str(other_org_id),
        )
        with (
            _stores_patched(async_session_maker)[0],
            _stores_patched(async_session_maker)[2],
        ):
            target = await user_auth.get_target_org_id_for_permission_check()

        # Non-member header value must still be returned -- the caller
        # (require_permission) decides via the org/super role fallback.
        assert target == other_org_id

    @pytest.mark.asyncio
    async def test_no_header_falls_back_to_current_org(
        self, async_session_maker, user_id, org_id
    ):
        await _seed_minimal(async_session_maker, user_id, org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
        )
        with _stores_patched(async_session_maker)[0]:
            target = await user_auth.get_target_org_id_for_permission_check()

        assert target == org_id

    @pytest.mark.asyncio
    async def test_malformed_header_still_raises_400(self, user_id):
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            _x_org_id_header='not-a-uuid',
        )
        with pytest.raises(HTTPException) as exc_info:
            await user_auth.get_target_org_id_for_permission_check()
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_api_key_header_mismatch_still_raises_403(
        self, user_id, org_id, other_org_id
    ):
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            api_key_org_id=org_id,
            _x_org_id_header=str(other_org_id),
        )
        with pytest.raises(HTTPException) as exc_info:
            await user_auth.get_target_org_id_for_permission_check()
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_api_key_pins_org_when_no_header(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        await _seed_minimal(async_session_maker, user_id, org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            api_key_org_id=other_org_id,
        )
        target = await user_auth.get_target_org_id_for_permission_check()
        assert target == other_org_id

    @pytest.mark.asyncio
    async def test_server_side_override_still_requires_membership(
        self, async_session_maker, user_id, org_id, other_org_id
    ):
        # ``effective_org_id_override`` is a trusted server-side
        # mechanism with its own membership check -- super-role bypass
        # only applies to client-supplied X-Org-Id headers, not to
        # the override path.
        await _seed_minimal(async_session_maker, user_id, org_id)
        user_auth = SaasUserAuth(
            user_id=user_id,
            refresh_token=SecretStr('mock'),
            effective_org_id_override=other_org_id,
        )
        with _stores_patched(async_session_maker)[2]:
            with pytest.raises(HTTPException) as exc_info:
                await user_auth.get_target_org_id_for_permission_check()
        assert exc_info.value.status_code == 403
