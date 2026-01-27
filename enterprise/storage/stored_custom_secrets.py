from sqlalchemy import Column, ForeignKey, Identity, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from storage.base import Base


class StoredCustomSecrets(Base):  # type: ignore
    __tablename__ = 'custom_secrets'
    id = Column(Integer, Identity(), primary_key=True)
    keycloak_user_id = Column(String, nullable=True, index=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey('org.id'), nullable=True)
    secret_name = Column(String, nullable=False)
    secret_value = Column(String, nullable=False)
    description = Column(String, nullable=True)

    # Relationships
    org = relationship('Org', back_populates='user_secrets')
