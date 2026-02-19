"""
SQLAlchemy model for User.
"""

from uuid import uuid4

from sqlalchemy import (
    UUID,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship
from storage.base import Base


class User(Base):  # type: ignore
    """User model with organizational relationships."""

    __tablename__ = 'user'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    current_org_id = Column(UUID(as_uuid=True), ForeignKey('org.id'), nullable=False)
    role_id = Column(Integer, ForeignKey('role.id'), nullable=True)
    accepted_tos = Column(DateTime, nullable=True)
    enable_sound_notifications = Column(Boolean, nullable=True)
    language = Column(String, nullable=True)
    user_consents_to_analytics = Column(Boolean, nullable=True)
    email = Column(String, nullable=True)
    email_verified = Column(Boolean, nullable=True)
    git_user_name = Column(String, nullable=True)
    git_user_email = Column(String, nullable=True)

    # Relationships
    role = relationship('Role', back_populates='users')
    org_members = relationship('OrgMember', back_populates='user')
    current_org = relationship('Org', back_populates='current_users')
    stored_conversation_metadata_saas = relationship(
        'StoredConversationMetadataSaas', back_populates='user'
    )
