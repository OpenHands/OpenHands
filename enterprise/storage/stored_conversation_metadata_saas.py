"""
SQLAlchemy model for ConversationMetadataSaas.

This model stores the SaaS-specific metadata for conversations,
containing only the conversation_id, user_id, and org_id.
"""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org
    from storage.user import User


class StoredConversationMetadataSaas(Base):
    """SaaS conversation metadata model containing user and org associations."""

    __tablename__ = 'conversation_metadata_saas'
    __table_args__ = (
        # The only pre-existing index was the conversation_id PK, so every query
        # filtering by user_id or org_id did a full table scan (~680K seq scans /
        # ~42B rows read in prod). These cover those paths:
        #   (user_id, org_id): conversation listing filters by user_id and
        #     optionally org_id; the prefix also serves user_id-only deletes.
        #   (org_id): org-admin paths select/delete by org_id alone.
        Index('ix_conversation_metadata_saas_user_id_org_id', 'user_id', 'org_id'),
        Index('ix_conversation_metadata_saas_org_id', 'org_id'),
    )

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
