"""Tests for AnalyticsContext dataclass and resolve_analytics_context factory."""

from unittest.mock import MagicMock, patch

import pytest

from openhands.analytics.analytics_context import (
    AnalyticsContext,
    resolve_analytics_context,
)
from openhands.analytics.user_provider import (
    AnalyticsUserProvider,
    DefaultAnalyticsUserProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockAnalyticsUserProvider(AnalyticsUserProvider):
    """Mock provider for testing that returns a configurable user."""

    def __init__(self, user=None, raise_exception=None):
        self._user = user
        self._raise_exception = raise_exception

    async def get_user_by_id(self, user_id: str):
        if self._raise_exception:
            raise self._raise_exception
        return self._user


def _patch_user_provider(provider: AnalyticsUserProvider):
    """Create a patch context that makes _get_user_provider return the given provider."""
    return patch(
        'openhands.analytics.analytics_context._get_user_provider',
        return_value=provider,
    )


# ---------------------------------------------------------------------------
# AnalyticsUserProvider tests
# ---------------------------------------------------------------------------


class TestAnalyticsUserProvider:
    """Tests for AnalyticsUserProvider base class and default implementation."""

    @pytest.mark.asyncio
    async def test_default_provider_returns_none(self):
        """DefaultAnalyticsUserProvider.get_user_by_id returns None for any user_id."""
        provider = DefaultAnalyticsUserProvider()
        result = await provider.get_user_by_id('any-user-id')
        assert result is None


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

        provider = MockAnalyticsUserProvider(user=mock_user)
        with _patch_user_provider(provider):
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

        provider = MockAnalyticsUserProvider(user=mock_user)
        with _patch_user_provider(provider):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.consented is False

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_org_id_none(self):
        """resolve_analytics_context with user.current_org_id=None returns org_id=None."""
        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = True
        mock_user.current_org_id = None

        provider = MockAnalyticsUserProvider(user=mock_user)
        with _patch_user_provider(provider):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.org_id is None

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_user_not_found(self):
        """resolve_analytics_context when provider returns None returns safe default."""
        provider = MockAnalyticsUserProvider(user=None)
        with _patch_user_provider(provider):
            ctx = await resolve_analytics_context('nonexistent-user')

        assert ctx.user_id == 'nonexistent-user'
        assert ctx.consented is False
        assert ctx.org_id is None
        assert ctx.user is None

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_provider_raises_exception(self):
        """resolve_analytics_context when provider raises Exception returns safe default (no exception leaks)."""
        provider = MockAnalyticsUserProvider(
            raise_exception=RuntimeError('DB connection failed')
        )
        with _patch_user_provider(provider):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.user_id == 'user-42'
        assert ctx.consented is False
        assert ctx.org_id is None
        assert ctx.user is None

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_logs_warning_on_failure(self):
        """resolve_analytics_context logs a warning when user lookup fails."""
        provider = MockAnalyticsUserProvider(raise_exception=RuntimeError('DB error'))
        with (
            _patch_user_provider(provider),
            patch('openhands.analytics.analytics_context.logger') as mock_logger,
        ):
            await resolve_analytics_context('user-42')

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert 'user-42' in str(call_args)

    @pytest.mark.asyncio
    async def test_resolve_analytics_context_with_default_provider(self):
        """resolve_analytics_context with DefaultAnalyticsUserProvider returns safe defaults."""
        provider = DefaultAnalyticsUserProvider()
        with _patch_user_provider(provider):
            ctx = await resolve_analytics_context('user-42')

        assert ctx.user_id == 'user-42'
        assert ctx.consented is False
        assert ctx.org_id is None
        assert ctx.user is None
