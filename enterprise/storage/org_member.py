"""
SQLAlchemy model for Organization-Member relationship.
"""

from pydantic import SecretStr
from sqlalchemy import UUID, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from storage.base import Base
from storage.encrypt_utils import decrypt_value, encrypt_value


class OrgMember(Base):  # type: ignore
    """Junction table for organization-member relationships with roles."""

    __tablename__ = 'org_member'

    org_id = Column(UUID(as_uuid=True), ForeignKey('org.id'), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), primary_key=True)
    role_id = Column(Integer, ForeignKey('role.id'), nullable=False)
    _llm_api_key = Column(String, nullable=False)
    max_iterations = Column(Integer, nullable=True)
    llm_model = Column(String, nullable=True)
    _llm_api_key_for_byor = Column(String, nullable=True)
    llm_base_url = Column(String, nullable=True)
    status = Column(String, nullable=True)

    # Relationships
    org = relationship('Org', back_populates='org_members')
    user = relationship('User', back_populates='org_members')
    role = relationship('Role', back_populates='org_members')

    def __init__(self, **kwargs):
        # Handle known SQLAlchemy columns directly
        for key in list(kwargs):
            if hasattr(self.__class__, key):
                setattr(self, key, kwargs.pop(key))

        # Handle custom property-style fields
        if 'llm_api_key' in kwargs:
            self.llm_api_key = kwargs.pop('llm_api_key')
        if 'llm_api_key_for_byor' in kwargs:
            self.llm_api_key_for_byor = kwargs.pop('llm_api_key_for_byor')

        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {list(kwargs.keys())}')

    @property
    def llm_api_key(self) -> SecretStr:
        decrypted = decrypt_value(self._llm_api_key)
        return SecretStr(decrypted)

    @llm_api_key.setter
    def llm_api_key(self, value: str | SecretStr):
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        self._llm_api_key = encrypt_value(raw)

    @property
    def llm_api_key_for_byor(self) -> SecretStr | None:
        if self._llm_api_key_for_byor:
            decrypted = decrypt_value(self._llm_api_key_for_byor)
            return SecretStr(decrypted)
        return None

    @llm_api_key_for_byor.setter
    def llm_api_key_for_byor(self, value: str | SecretStr | None):
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        self._llm_api_key_for_byor = encrypt_value(raw) if raw else None
