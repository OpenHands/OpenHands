"""
SQLAlchemy model for User.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas


class User(Base):
    """User model with organizational relationships."""

    __tablename__ = 'user'

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    current_org_id: Mapped[UUID] = mapped_column(ForeignKey('org.id'), nullable=False)
    role_id: Mapped[int | None] = mapped_column(ForeignKey('role.id'), nullable=True)
    accepted_tos: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    enable_sound_notifications: Mapped[bool | None] = mapped_column(nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    user_consents_to_analytics: Mapped[bool | None] = mapped_column(nullable=True)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    email_verified: Mapped[bool | None] = mapped_column(nullable=True)
    git_user_name: Mapped[str | None] = mapped_column(String, nullable=True)
    git_user_email: Mapped[str | None] = mapped_column(String, nullable=True)
    sandbox_grouping_strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    disabled_skills: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    onboarding_completed: Mapped[bool | None] = mapped_column(
        nullable=True, default=False
    )

    # Relationships
    role: Mapped['Role | None'] = relationship('Role', back_populates='users')
    org_members: Mapped[list['OrgMember']] = relationship(
        'OrgMember', back_populates='user'
    )
    current_org: Mapped['Org'] = relationship('Org', back_populates='current_users')
    stored_conversation_metadata_saas: Mapped[
        list['StoredConversationMetadataSaas']
    ] = relationship('StoredConversationMetadataSaas', back_populates='user')
