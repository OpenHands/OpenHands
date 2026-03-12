"""User authorization model for managing email/provider based access control."""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import Column, DateTime, Identity, Integer, String
from storage.base import Base


class UserAuthorizationType(str, Enum):
    """Type of user authorization rule."""

    WHITELIST = 'whitelist'
    BLACKLIST = 'blacklist'


class UserAuthorization(Base):  # type: ignore
    """Stores user authorization rules based on email patterns and provider types.

    Supports:
    - Email pattern matching using SQL LIKE (e.g., '%@openhands.dev')
    - Provider type filtering (e.g., 'github', 'gitlab')
    - Whitelist/Blacklist rules

    When email_pattern is NULL, the rule matches all emails.
    When provider_type is NULL, the rule matches all providers.
    """

    __tablename__ = 'user_authorizations'

    id = Column(Integer, Identity(), primary_key=True)
    email_pattern = Column(String, nullable=True)
    provider_type = Column(String, nullable=True)
    type = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
