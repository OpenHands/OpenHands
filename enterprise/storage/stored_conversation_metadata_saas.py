"""
SQLAlchemy model for ConversationMetadataSaas.

This model stores the SaaS-specific metadata for conversations,
containing only the conversation_id, user_id, and org_id.
"""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org
    from storage.user import User


class StoredConversationMetadataSaas(Base):
    """SaaS conversation metadata model containing user and org associations."""

    __tablename__ = 'conversation_metadata_saas'

    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey('user.id'), nullable=False)
    org_id: Mapped[UUID] = mapped_column(ForeignKey('org.id'), nullable=False)

    # Relationships
    user: Mapped['User'] = relationship(
        'User', back_populates='stored_conversation_metadata_saas'
    )
    org: Mapped['Org'] = relationship(
        'Org', back_populates='stored_conversation_metadata_saas'
    )


__all__ = ['StoredConversationMetadataSaas']
