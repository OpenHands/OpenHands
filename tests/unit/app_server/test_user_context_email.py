"""Unit tests for ``UserContext.get_user_email()`` implementations.

These tests cover the change that uses the user's email (when available) as
the Laminar trace ``user_id`` so traces are immediately attributable in the
Laminar UI, while still falling back to the internal user id (e.g. in OSS
mode or for admin-scoped contexts).
"""

from unittest.mock import AsyncMock

import pytest

from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.app_server.user.specifiy_user_context import (
    ADMIN,
    SpecifyUserContext,
)


class TestAuthUserContextEmail:
    """``AuthUserContext.get_user_email()`` delegates to ``UserAuth``."""

    @pytest.mark.asyncio
    async def test_returns_email_when_user_auth_has_one(self):
        """SaaS mode: Keycloak-backed UserAuth returns the user's email."""
        user_auth = AsyncMock()
        user_auth.get_user_email = AsyncMock(return_value='alice@example.com')
        ctx = AuthUserContext(user_auth=user_auth)

        email = await ctx.get_user_email()

        assert email == 'alice@example.com'
        user_auth.get_user_email.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_user_auth_has_no_email(self):
        """OSS mode: DefaultUserAuth returns None and we propagate that."""
        user_auth = AsyncMock()
        user_auth.get_user_email = AsyncMock(return_value=None)
        ctx = AuthUserContext(user_auth=user_auth)

        email = await ctx.get_user_email()

        assert email is None
        user_auth.get_user_email.assert_awaited_once()


class TestSpecifyUserContextEmail:
    """Admin contexts have no associated end-user email."""

    @pytest.mark.asyncio
    async def test_returns_none_for_admin_context(self):
        assert await ADMIN.get_user_email() is None

    @pytest.mark.asyncio
    async def test_returns_none_for_arbitrary_user_id(self):
        ctx = SpecifyUserContext(user_id='some-uuid')
        assert await ctx.get_user_email() is None


class TestLaminarUserIdFallback:
    """The exact ``email or user_id`` expression used at the three callsites
    inside ``LiveStatusAppConversationService``. We verify the semantics here
    rather than reach into the large async _start_app_conversation flow.
    """

    @pytest.mark.asyncio
    async def test_prefers_email_when_available(self):
        ctx = AuthUserContext(user_auth=AsyncMock())
        ctx.user_auth.get_user_email = AsyncMock(return_value='alice@example.com')
        user_id = 'internal-uuid-123'

        laminar_user_id = await ctx.get_user_email() or user_id

        assert laminar_user_id == 'alice@example.com'

    @pytest.mark.asyncio
    async def test_falls_back_to_user_id_when_email_is_none(self):
        ctx = AuthUserContext(user_auth=AsyncMock())
        ctx.user_auth.get_user_email = AsyncMock(return_value=None)
        user_id = 'internal-uuid-123'

        laminar_user_id = await ctx.get_user_email() or user_id

        assert laminar_user_id == 'internal-uuid-123'

    @pytest.mark.asyncio
    async def test_falls_back_to_user_id_when_email_is_empty_string(self):
        """``or`` also treats '' as falsy, which is the desired behavior:
        an empty email is not useful as a Laminar trace id."""
        ctx = AuthUserContext(user_auth=AsyncMock())
        ctx.user_auth.get_user_email = AsyncMock(return_value='')
        user_id = 'internal-uuid-123'

        laminar_user_id = await ctx.get_user_email() or user_id

        assert laminar_user_id == 'internal-uuid-123'

    @pytest.mark.asyncio
    async def test_admin_context_always_falls_back_to_user_id(self):
        user_id = 'admin-target-user-id'
        laminar_user_id = await ADMIN.get_user_email() or user_id
        assert laminar_user_id == 'admin-target-user-id'
