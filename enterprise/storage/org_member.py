"""
SQLAlchemy model for Organization-Member relationship.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import SecretStr
from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base
from storage.encrypt_utils import decrypt_value, encrypt_value

if TYPE_CHECKING:
    from storage.org import Org
    from storage.role import Role
    from storage.user import User


class OrgMember(Base):
    """Junction table for organization-member relationships with roles."""

    __tablename__ = 'org_member'

    org_id: Mapped[UUID] = mapped_column(ForeignKey('org.id'), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey('user.id'), primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey('role.id'), nullable=False)
    _llm_api_key: Mapped[str] = mapped_column(String, nullable=False)
    _llm_api_key_for_byor: Mapped[str | None] = mapped_column(String, nullable=True)
    has_custom_llm_api_key: Mapped[bool] = mapped_column(nullable=False, default=False)
    agent_settings_diff: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    conversation_settings_diff: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    status: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationships
    org: Mapped['Org'] = relationship('Org', back_populates='org_members')
    user: Mapped['User'] = relationship('User', back_populates='org_members')
    role: Mapped['Role'] = relationship('Role', back_populates='org_members')

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
