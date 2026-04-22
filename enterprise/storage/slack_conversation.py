from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Identity, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org


class SlackConversation(Base):
    __tablename__ = 'slack_conversation'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    channel_id: Mapped[str] = mapped_column(String, nullable=False)
    keycloak_user_id: Mapped[str] = mapped_column(String, nullable=False)
    org_id: Mapped[UUID | None] = mapped_column(ForeignKey('org.id'), nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    v1_enabled: Mapped[bool | None] = mapped_column(nullable=True)

    # Relationships
    org: Mapped['Org | None'] = relationship(
        'Org', back_populates='slack_conversations'
    )
