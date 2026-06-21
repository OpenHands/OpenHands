"""Protocol for User models across OSS and Enterprise.

This module provides a Protocol that defines the minimal interface required
for analytics user lookup. Any object with matching attributes satisfies
the protocol — no inheritance required (structural typing).

Uses Any types for compatibility with SQLAlchemy's Mapped descriptors,
which mypy sees as Mapped[T] rather than T without the SQLAlchemy mypy plugin.
The actual expected types are documented in the docstring.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class UserBase(Protocol):
    """Protocol defining the user interface for analytics.

    This protocol defines the minimal set of attributes required for analytics
    functionality. Any object with these attributes satisfies the protocol,
    including SQLAlchemy models and mock objects in tests.

    All attributes use ``Any`` type for compatibility with SQLAlchemy Mapped
    descriptors. The expected runtime types are:

    Attributes:
        id: UUID | str - User's unique identifier.
        user_consents_to_analytics: bool | None - Whether user consented to
            analytics, or None if undecided (treated as False).
        current_org_id: UUID | str | None - Organization ID, or None.
        accepted_tos: datetime | None - When user accepted terms of service.
    """

    id: Any
    user_consents_to_analytics: Any
    current_org_id: Any
    accepted_tos: Any
