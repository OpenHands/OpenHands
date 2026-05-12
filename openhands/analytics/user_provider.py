"""Abstract base class for analytics user lookup.

This module provides an extension point for user lookup in analytics contexts.
The default implementation returns None, which results in safe defaults being
used. Enterprise deployments can provide their own implementation that queries
the actual user store.

The implementation class is configured via server_config.analytics_user_provider_class
and loaded via get_impl() at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from openhands.analytics.user_base import UserBase


class AnalyticsUserProvider(ABC):
    """Abstract base class for looking up users for analytics purposes.

    This is an extension point that allows enterprise deployments to provide
    user lookup functionality without the open-source codebase depending on
    enterprise modules.

    The default implementation (DefaultAnalyticsUserProvider) returns None,
    which causes resolve_analytics_context to use safe defaults.
    """

    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> UserBase | None:
        """Look up a user by their ID.

        Args:
            user_id: The user's unique identifier.

        Returns:
            A UserBase instance with `user_consents_to_analytics` and
            `current_org_id` attributes, or None if the user cannot be found
            or user lookup is not available.
        """


class DefaultAnalyticsUserProvider(AnalyticsUserProvider):
    """Default implementation that returns None for all lookups.

    This is used in open-source/local deployments where no user store is
    available. All analytics calls will use safe defaults (consented=False,
    org_id=None, user=None).
    """

    async def get_user_by_id(self, user_id: str) -> UserBase | None:
        """Return None since no user store is available in local mode."""
        return None
