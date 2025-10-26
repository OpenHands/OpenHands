from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from storage.base import Base


class ApiKey(Base):
    """
    Represents an API key for a user.
    """

    __tablename__ = 'api_keys'
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey('org.id'), nullable=True)
    name = Column(String(255), nullable=True)
    created_at = Column(
        DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False
    )
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # Relationships
    org = relationship('Org', back_populates='api_keys')
