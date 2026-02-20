"""
SQLAlchemy model for ConversationMetadataSaas.

This model stores the SaaS-specific metadata for conversations,
containing only the conversation_id, user_id, and org_id.
"""

from sqlalchemy import UUID as SQL_UUID
from sqlalchemy import Column, ForeignKey, String
from sqlalchemy.orm import relationship
from storage.base import Base


class StoredConversationMetadataSaas(Base):  # type: ignore
    """SaaS conversation metadata model containing user and org associations."""

    __tablename__ = 'conversation_metadata_saas'

    conversation_id = Column(String, primary_key=True)
    user_id = Column(SQL_UUID(as_uuid=True), ForeignKey('user.id'), nullable=False)
    org_id = Column(SQL_UUID(as_uuid=True), ForeignKey('org.id'), nullable=False)

    # Relationships
    user = relationship('User', back_populates='stored_conversation_metadata_saas')
    org = relationship('Org', back_populates='stored_conversation_metadata_saas')


__all__ = ['StoredConversationMetadataSaas']
