"""
SQLAlchemy model for Organization Invitation.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org
    from storage.role import Role
    from storage.user import User


class OrgInvitation(Base):
    """Organization invitation model.

    Represents an invitation for a user to join an organization.
    Invitations are created by organization owners/admins and contain
    a secure token that can be used to accept the invitation.
    """

    __tablename__ = 'org_invitation'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    token: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    org_id: Mapped[UUID] = mapped_column(
        ForeignKey('org.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role_id: Mapped[int] = mapped_column(ForeignKey('role.id'), nullable=False)
    inviter_id: Mapped[UUID] = mapped_column(ForeignKey('user.id'), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'pending'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text('CURRENT_TIMESTAMP'),
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    accepted_by_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey('user.id'),
        nullable=True,
    )

    # Relationships
    org: Mapped['Org'] = relationship('Org', back_populates='invitations')
    role: Mapped['Role'] = relationship('Role')
    inviter: Mapped['User'] = relationship('User', foreign_keys=[inviter_id])
    accepted_by_user: Mapped['User | None'] = relationship(
        'User', foreign_keys=[accepted_by_user_id]
    )

    # Status constants
    STATUS_PENDING = 'pending'
    STATUS_ACCEPTED = 'accepted'
    STATUS_REVOKED = 'revoked'
    STATUS_EXPIRED = 'expired'
