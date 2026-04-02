"""
SQLAlchemy model for Git Organization Claims.
"""

from uuid import uuid4

from sqlalchemy import UUID, Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship
from storage.base import Base


class OrgGitClaim(Base):  # type: ignore
    """Model for tracking which OpenHands org has claimed a Git organization."""

    __tablename__ = 'org_git_claim'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    org_id = Column(
        UUID(as_uuid=True), ForeignKey('org.id', ondelete='CASCADE'), nullable=False
    )
    provider = Column(String, nullable=False)
    git_organization = Column(String, nullable=False)
    claimed_by = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False)
    claimed_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint('provider', 'git_organization', name='uq_provider_git_org'),
    )

    org = relationship('Org', back_populates='git_claims')
