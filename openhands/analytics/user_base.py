"""Protocol for User models across OSS and Enterprise.

This module provides a Protocol that defines the minimal interface required
for analytics user lookup. Any object with matching attributes satisfies
the protocol — no inheritance required (structural typing).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class UserBase(Protocol):
    """Protocol defining the user interface for analytics.

    This protocol defines the minimal set of attributes required for analytics
    functionality. Any object with these attributes satisfies the protocol,
    including SQLAlchemy models and mock objects in tests.

    Attributes:
        id: The user's unique identifier (UUID or string).
        user_consents_to_analytics: Whether the user has consented to analytics.
            None means undecided (treated as not consented).
        current_org_id: The user's current organization ID, or None if not
            applicable (e.g., in OSS mode).
        accepted_tos: Timestamp when user accepted terms of service, used for
            tracking time-to-activate metrics.
    """

    id: Any
    user_consents_to_analytics: bool | None
    current_org_id: Any | None
    accepted_tos: datetime | None
