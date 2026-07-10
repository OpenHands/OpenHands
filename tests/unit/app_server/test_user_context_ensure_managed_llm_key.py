"""Unit tests for ``UserContext.ensure_managed_llm_key`` (APP-2678).

The read-path self-heal for stale managed (OpenHands-proxy) LLM API keys.
The base implementation is a no-op; ``AuthUserContext`` delegates to
``UserAuth.ensure_managed_llm_key`` so the SaaS override there can talk
to LiteLLM and mint a fresh key. Admin contexts must stay a no-op since
they never start a conversation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.app_server.user.specifiy_user_context import (
    ADMIN,
    SpecifyUserContext,
)
from openhands.app_server.user.user_models import UserInfo


def _make_user_info() -> UserInfo:
    """Build a minimal UserInfo for the test path; only the LLM block is
    inspected by the implementations under test."""
    return UserInfo(
        id='test-user-id',
        agent_settings_diff={'llm': {'model': 'openhands/gpt-4'}},
    )


class TestSpecifyUserContextNoop:
    """Admin contexts never start a conversation; nothing to heal."""

    @pytest.mark.asyncio
    async def test_admin_noop(self):
        user = _make_user_info()
        result = await ADMIN.ensure_managed_llm_key(user)
        assert result is user

    @pytest.mark.asyncio
    async def test_arbitrary_user_id_noop(self):
        ctx = SpecifyUserContext(user_id='some-uuid')
        user = _make_user_info()
        result = await ctx.ensure_managed_llm_key(user)
        assert result is user


class TestAuthUserContextDelegation:
    """``AuthUserContext.ensure_managed_llm_key`` delegates to the
    underlying ``UserAuth.ensure_managed_llm_key`` so the SaaS-specific
    verify-and-fix logic can run."""

    @pytest.mark.asyncio
    async def test_delegates_to_user_auth(self):
        user_auth = MagicMock()
        user_auth.ensure_managed_llm_key = AsyncMock()
        ctx = AuthUserContext(user_auth=user_auth)

        user = _make_user_info()
        await ctx.ensure_managed_llm_key(user)

        user_auth.ensure_managed_llm_key.assert_awaited_once_with(user)

    @pytest.mark.asyncio
    async def test_returns_user_info_after_underlying_mutation(self):
        """``UserAuth.ensure_managed_llm_key`` is contractually a
        pass-through (mutates in place) — the context must surface the same
        object so the caller's reference is still the healed one."""
        user = _make_user_info()
        user_auth = MagicMock()
        user_auth.ensure_managed_llm_key = AsyncMock(
            side_effect=lambda settings: settings  # pass-through, like the real impl
        )
        ctx = AuthUserContext(user_auth=user_auth)

        result = await ctx.ensure_managed_llm_key(user)

        assert result is user
