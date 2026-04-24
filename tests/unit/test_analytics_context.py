"""Tests for AnalyticsContext dataclass and resolve_analytics_context factory."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.analytics.analytics_context import (
    AnalyticsContext,
    resolve_analytics_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_store_module(get_user_by_id_mock: AsyncMock) -> dict[str, ModuleType]:
    """Build a fake ``storage.user_store`` module containing a mock UserStore.

    Returns a dict suitable for ``patch.dict(sys.modules, ...)``.
    """
    mock_user_store_class = MagicMock()
    mock_user_store_class.get_user_by_id = get_user_by_id_mock

    user_store_mod = ModuleType('storage.user_store')
    user_store_mod.UserStore = mock_user_store_class  # type: ignore[attr-defined]

    # Ensure 'storage' parent package exists in sys.modules too.
    storage_mod = sys.modules.get('storage') or ModuleType('storage')

    return {'storage': storage_mod, 'storage.user_store': user_store_mod}


# ---------------------------------------------------------------------------
# AnalyticsContext dataclass tests
# ---------------------------------------------------------------------------


class TestAnalyticsContext:
    """Tests for AnalyticsContext dataclass construction and field storage."""

    def test_context_stores_all_fields_correctly(self):
        """AnalyticsContext constructed with explicit values stores user_id, consented, org_id, user fields correctly."""
        mock_user = MagicMock()
        ctx = AnalyticsContext(
            user_id='user-123',
            consented=True,
            org_id='org-456',
            user=mock_user,
        )
        assert ctx.user_id == 'user-123'
        assert ctx.consented is True
        assert ctx.org_id == 'org-456'
        assert ctx.user is mock_user

    def test_context_default_safe_values(self):
        """AnalyticsContext can be created with safe defaults (consented=False, org_id=None, user=None)."""
        ctx = AnalyticsContext(
            user_id='user-123',
            consented=False,
            org_id=None,
            user=None,
        )
        assert ctx.user_id == 'user-123'
        assert ctx.consented is False
        assert ctx.org_id is None
        assert ctx.user is None


# ---------------------------------------------------------------------------
# resolve_analytics_context factory tests
# ---------------------------------------------------------------------------


class TestResolveContext:
    """Tests for resolve_analytics_context async factory function."""

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_with_valid_user(self):
        """resolve_analytics_context with valid user_id returns AnalyticsContext with consented from user, org_id from user."""
        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = True
        mock_user.current_org_id = 'org-abc-123'

        modules = _make_user_store_module(AsyncMock(return_value=mock_user))
        with patch.dict(sys.modules, modules):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.user_id == 'user-42'
        assert ctx.consented is True
        assert ctx.org_id == 'org-abc-123'
        assert ctx.user is mock_user

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_consent_none_means_false(self):
        """resolve_analytics_context with user.user_consents_to_analytics=None returns consented=False."""
        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = None
        mock_user.current_org_id = 'org-1'

        modules = _make_user_store_module(AsyncMock(return_value=mock_user))
        with patch.dict(sys.modules, modules):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.consented is False

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_org_id_none(self):
        """resolve_analytics_context with user.current_org_id=None returns org_id=None."""
        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = True
        mock_user.current_org_id = None

        modules = _make_user_store_module(AsyncMock(return_value=mock_user))
        with patch.dict(sys.modules, modules):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.org_id is None

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_user_not_found(self):
        """resolve_analytics_context when UserStore returns None returns safe default."""
        modules = _make_user_store_module(AsyncMock(return_value=None))
        with patch.dict(sys.modules, modules):
            ctx = await resolve_analytics_context('nonexistent-user')

        assert ctx.user_id == 'nonexistent-user'
        assert ctx.consented is False
        assert ctx.org_id is None
        assert ctx.user is None

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_user_store_raises_exception(self):
        """resolve_analytics_context when UserStore raises Exception returns safe default (no exception leaks)."""
        modules = _make_user_store_module(
            AsyncMock(side_effect=RuntimeError('DB connection failed'))
        )
        with patch.dict(sys.modules, modules):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.user_id == 'user-42'
        assert ctx.consented is False
        assert ctx.org_id is None
        assert ctx.user is None

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_logs_warning_on_failure(self):
        """resolve_analytics_context logs a warning when user lookup fails."""
        modules = _make_user_store_module(
            AsyncMock(side_effect=RuntimeError('DB error'))
        )
        with (
            patch.dict(sys.modules, modules),
            patch('openhands.analytics.analytics_context.logger') as mock_logger,
        ):
            await resolve_analytics_context('user-42')

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert 'user-42' in str(call_args)
