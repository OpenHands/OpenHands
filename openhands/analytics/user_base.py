"""Protocol for User models across OSS and Enterprise.

This module provides a Protocol that defines the minimal interface required
for analytics user lookup. Any object with matching attributes satisfies
the protocol — no inheritance required (structural typing).

Uses Any types for all attributes to ensure compatibility with SQLAlchemy's
Mapped descriptors, which mypy sees as Mapped[T] rather than T without
the SQLAlchemy mypy plugin.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class UserBase(Protocol):
    """Protocol defining the user interface for analytics.

    This protocol defines the minimal set of attributes required for analytics
    functionality. Any object with these attributes satisfies the protocol,
    including SQLAlchemy models and mock objects in tests.

    All attributes use `Any` type to be compatible with both plain Python
    objects and SQLAlchemy Mapped descriptors. The actual type semantics:
        id: User's unique identifier (UUID or string).
        user_consents_to_analytics: bool | None - whether user consented.
        current_org_id: Organization ID or None (e.g., in OSS mode).
        accepted_tos: datetime | None - when user accepted terms of service.
    """

    id: Any
    user_consents_to_analytics: Any
    current_org_id: Any
    accepted_tos: Any
