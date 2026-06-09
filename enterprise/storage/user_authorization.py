"""User authorization model for managing email/provider based access control."""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import DateTime, Identity, String
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class UserAuthorizationType(str, Enum):
    """Type of user authorization rule."""

    WHITELIST = 'whitelist'
    BLACKLIST = 'blacklist'


class UserAuthorization(Base):
    """Stores user authorization rules based on email patterns and provider types.

    Supports:
    - Email pattern matching using SQL LIKE (e.g., '%@openhands.dev')
    - Provider type filtering (e.g., 'github', 'gitlab')
    - Whitelist/Blacklist rules

    When email_pattern is NULL, the rule matches all emails.
    When provider_type is NULL, the rule matches all providers.
    """

    __tablename__ = 'user_authorizations'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    email_pattern: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_type: Mapped[str | None] = mapped_column(String, nullable=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
