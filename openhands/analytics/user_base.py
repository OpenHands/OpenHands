"""Base class for User models across OSS and Enterprise.

This module provides a shared abstract base class that defines the minimal
interface required for analytics user lookup. Both the enterprise SQLAlchemy
User model and any OSS user representations should inherit from this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class UserBase(ABC):
    """Abstract base class defining the user interface for analytics.

    This class defines the minimal set of attributes required for analytics
    functionality. Enterprise User models (SQLAlchemy) and OSS user models
    should inherit from this class.

    Attributes:
        id: The user's unique identifier (UUID or string).
        user_consents_to_analytics: Whether the user has consented to analytics.
            None means undecided (treated as not consented).
        current_org_id: The user's current organization ID, or None if not
            applicable (e.g., in OSS mode).
    """

    @property
    @abstractmethod
    def id(self) -> Any:
        """The user's unique identifier."""

    @property
    @abstractmethod
    def user_consents_to_analytics(self) -> bool | None:
        """Whether the user has consented to analytics."""

    @property
    @abstractmethod
    def current_org_id(self) -> Any | None:
        """The user's current organization ID."""
