"""SaaS implementation of AnalyticsUserProvider.

This module provides the enterprise implementation of AnalyticsUserProvider
that queries the actual UserStore to look up user information for analytics.
"""

from __future__ import annotations

from storage.user_store import UserStore

from openhands.analytics.user_base import UserBase
from openhands.analytics.user_provider import AnalyticsUserProvider


class SaasAnalyticsUserProvider(AnalyticsUserProvider):
    """Enterprise implementation that queries the UserStore.

    This implementation is used in SaaS deployments where user information
    is stored in the database and accessible via UserStore.
    """

    async def get_user_by_id(self, user_id: str) -> UserBase | None:
        """Look up a user by their ID using the UserStore.

        Args:
            user_id: The user's unique identifier.

        Returns:
            The User object from the database, or None if not found.
        """
        return await UserStore.get_user_by_id(user_id)
