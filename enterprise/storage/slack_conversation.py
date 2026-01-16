from sqlalchemy import Boolean, Column, ForeignKey, Identity, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from storage.base import Base


class SlackConversation(Base):  # type: ignore
    __tablename__ = 'slack_conversation'
    id = Column(Integer, Identity(), primary_key=True)
    conversation_id = Column(String, nullable=False, index=True)
    channel_id = Column(String, nullable=False)
    keycloak_user_id = Column(String, nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey('org.id'), nullable=True)
    parent_id = Column(String, nullable=True, index=True)
    v1_enabled = Column(Boolean, nullable=True)

    # Relationships
    org = relationship('Org', back_populates='slack_conversations')
