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


async def _seed_minimal(session_maker, user_id_str, current_org_id, *extra_org_ids):
    """Create a role, org(s), user, and org_member rows in the in-memory DB."""
    async with session_maker() as session:
        role = Role(name='member', rank=3)
        session.add(role)
        await session.flush()

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
