"""SQLAlchemy model for tracking users synced to Resend audiences."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from storage.base import Base


class ResendSyncedUser(Base):  # type: ignore
    """Tracks users that have been synced to a Resend audience.

    This table ensures that once a user is synced to a Resend audience,
    they won't be re-added even if they are later deleted from the
    Resend UI. This respects manual deletions/unsubscribes.
    """

    __tablename__ = 'resend_synced_users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String, nullable=False, index=True)
    audience_id = Column(String, nullable=False, index=True)
    synced_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    keycloak_user_id = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            'email', 'audience_id', name='uq_resend_synced_email_audience'
        ),
    )
