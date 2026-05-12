"""SQLAlchemy model for tracking users synced to Resend audiences."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class ResendSyncedUser(Base):
    """Tracks users that have been synced to a Resend audience.

    This table ensures that once a user is synced to a Resend audience,
    they won't be re-added even if they are later deleted from the
    Resend UI. This respects manual deletions/unsubscribes.
    """

    __tablename__ = 'resend_synced_users'

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String, nullable=False, index=True)
    audience_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    keycloak_user_id: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            'email', 'audience_id', name='uq_resend_synced_email_audience'
        ),
    )
