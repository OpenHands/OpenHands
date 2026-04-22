"""
SQLAlchemy model for Git Organization Claims.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org


class OrgGitClaim(Base):
    """Model for tracking which OpenHands org has claimed a Git organization."""

    __tablename__ = 'org_git_claim'

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    org_id: Mapped[UUID] = mapped_column(
        ForeignKey('org.id', ondelete='CASCADE'), nullable=False
    )
    provider: Mapped[str] = mapped_column(String, nullable=False)
    git_organization: Mapped[str] = mapped_column(String, nullable=False)
    claimed_by: Mapped[UUID] = mapped_column(ForeignKey('user.id'), nullable=False)
    claimed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        UniqueConstraint('provider', 'git_organization', name='uq_provider_git_org'),
    )

    org: Mapped['Org'] = relationship('Org', back_populates='git_claims')
