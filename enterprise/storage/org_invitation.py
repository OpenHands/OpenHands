"""
SQLAlchemy model for Organization Invitation.
"""

from sqlalchemy import UUID, Column, DateTime, ForeignKey, Integer, String, text
from sqlalchemy.orm import relationship
from storage.base import Base


class OrgInvitation(Base):  # type: ignore
    """Organization invitation model.

    Represents an invitation for a user to join an organization.
    Invitations are created by organization owners/admins and contain
    a secure token that can be used to accept the invitation.
    """

    __tablename__ = 'org_invitation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    token = Column(String(64), nullable=False, unique=True, index=True)
    org_id = Column(
        UUID(as_uuid=True),
        ForeignKey('org.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    email = Column(String(255), nullable=False, index=True)
    role_id = Column(Integer, ForeignKey('role.id'), nullable=False)
    inviter_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False)
    status = Column(
        String(20),
        nullable=False,
        server_default=text("'pending'"),
    )
    created_at = Column(
        DateTime,
        nullable=False,
        server_default=text('CURRENT_TIMESTAMP'),
    )
    expires_at = Column(DateTime, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    accepted_by_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('user.id'),
        nullable=True,
    )

    # Relationships
    org = relationship('Org', back_populates='invitations')
    role = relationship('Role')
    inviter = relationship('User', foreign_keys=[inviter_id])
    accepted_by_user = relationship('User', foreign_keys=[accepted_by_user_id])

    # Status constants
    STATUS_PENDING = 'pending'
    STATUS_ACCEPTED = 'accepted'
    STATUS_REVOKED = 'revoked'
    STATUS_EXPIRED = 'expired'
