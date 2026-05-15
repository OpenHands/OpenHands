"""AnalyticsContext: resolution helper for analytics call sites.

Provides a dataclass that bundles user_id, consent status, org_id, and the
full user object into a single value.  The async ``resolve_analytics_context`` factory
performs user lookup with full error isolation so callers never need
try/except around user resolution.

User lookup is performed via the AnalyticsUserProvider abstraction, which follows
the get_impl() pattern used throughout OpenHands. Enterprise deployments provide
their own implementation (e.g., SaasAnalyticsUserProvider) that queries the actual
user store, while the default implementation returns None for local/OSS mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openhands.analytics.user_base import UserBase
from openhands.analytics.user_provider import AnalyticsUserProvider
from openhands.app_server.utils.import_utils import get_impl
from openhands.app_server.utils.logger import openhands_logger as logger

# Sentinel reused by resolve_analytics_context for the safe-default path.
_SAFE_DEFAULT_KWARGS: dict[str, Any] = {
    'consented': False,
    'org_id': None,
    'user': None,
}


@dataclass
class AnalyticsContext:
    """Resolved analytics context for a single user.

    Attributes:
        user_id:   Raw user ID string (always set).
        consented: Whether the user opted in to analytics.  ``False`` is the
                   safe default (None / missing / error all map to False).
        org_id:    String org_id derived from ``user.current_org_id``, or
                   ``None`` when unavailable.
        user:      The UserBase instance, or ``None`` when lookup failed.
    """

    user_id: str
    consented: bool
    org_id: str | None
    user: UserBase | None


def _get_user_provider() -> AnalyticsUserProvider:
    """Get the configured AnalyticsUserProvider implementation.

    Uses get_impl() to load the implementation class specified in
    server_config.analytics_user_provider_class. Defaults to
    DefaultAnalyticsUserProvider if not configured.
    """
    from openhands.app_server.shared import server_config

    impl_class = get_impl(
        AnalyticsUserProvider, server_config.analytics_user_provider_class
    )
    return impl_class()


async def resolve_analytics_context(user_id: str) -> AnalyticsContext:
    """Resolve a user_id into a fully-populated :class:`AnalyticsContext`.

    Performs user lookup via the configured AnalyticsUserProvider, extracts
    consent and org_id, and wraps everything in try/except so no exception
    ever leaks to the caller.

    Returns a safe default (consented=False, org_id=None, user=None) when the
    user is not found or any error occurs.
    """
    try:
        provider = _get_user_provider()
        user = await provider.get_user_by_id(user_id)

        if user is None:
            return AnalyticsContext(user_id=user_id, **_SAFE_DEFAULT_KWARGS)

        # None = undecided = not consented (same logic as auth.py)
        consented = user.user_consents_to_analytics is True
        org_id = str(user.current_org_id) if user.current_org_id else None

        return AnalyticsContext(
            user_id=user_id,
            consented=consented,
            org_id=org_id,
            user=user,
        )
    except Exception:
        logger.warning(
            'resolve_analytics_context failed for user_id=%s, returning safe default',
            user_id,
        )
        return AnalyticsContext(user_id=user_id, **_SAFE_DEFAULT_KWARGS)
