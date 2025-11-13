from sqlalchemy import Column, DateTime, ForeignKey, Identity, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from storage.base import Base


class SlackUser(Base):  # type: ignore
    __tablename__ = 'slack_users'
    id = Column(Integer, Identity(), primary_key=True)
    keycloak_user_id = Column(String, nullable=False, index=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey('org.id'), nullable=True)
    slack_user_id = Column(String, nullable=False, index=True)
    slack_display_name = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )

    # Relationships
    org = relationship('Org', back_populates='slack_users')
