from sqlalchemy import Column, Identity, Integer, String
from storage.base import Base


class StoredCustomSecrets(Base):  # type: ignore
    __tablename__ = 'custom_secrets'
    id = Column(Integer, Identity(), primary_key=True)
    keycloak_user_id = Column(String, nullable=True, index=True)
    secret_name = Column(String, nullable=False)
    secret_value = Column(String, nullable=False)
    description = Column(String, nullable=True)
