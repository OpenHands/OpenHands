"""
SQLAlchemy model for Organization.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import SecretStr
from server.constants import DEFAULT_BILLING_MARGIN
from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base
from storage.encrypt_utils import decrypt_value, encrypt_value

if TYPE_CHECKING:
    from storage.api_key import ApiKey
    from storage.billing_session import BillingSession
    from storage.org_git_claim import OrgGitClaim
    from storage.org_invitation import OrgInvitation
    from storage.org_member import OrgMember
    from storage.slack_conversation import SlackConversation
    from storage.slack_user import SlackUser
    from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
    from storage.stored_custom_secrets import StoredCustomSecrets
    from storage.stripe_customer import StripeCustomer
    from storage.user import User


class Org(Base):
    """Organization model."""

    __tablename__ = 'org'

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    contact_name: Mapped[str | None] = mapped_column(String, nullable=True)
    contact_email: Mapped[str | None] = mapped_column(String, nullable=True)
    remote_runtime_resource_factor: Mapped[int | None] = mapped_column(nullable=True)
    billing_margin: Mapped[float | None] = mapped_column(
        nullable=True, default=DEFAULT_BILLING_MARGIN
    )
    enable_proactive_conversation_starters: Mapped[bool] = mapped_column(
        nullable=False, default=True
    )
    sandbox_base_container_image: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    sandbox_runtime_container_image: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    org_version: Mapped[int] = mapped_column(nullable=False, default=0)
    agent_settings: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    conversation_settings: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    # encrypted column, don't set directly, set without the underscore
    _llm_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    # encrypted column, don't set directly, set without the underscore
    _search_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    # encrypted column, don't set directly, set without the underscore
    _sandbox_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    max_budget_per_task: Mapped[float | None] = mapped_column(nullable=True)
    enable_solvability_analysis: Mapped[bool | None] = mapped_column(
        nullable=True, default=False
    )
    v1_enabled: Mapped[bool | None] = mapped_column(nullable=True)
    conversation_expiration: Mapped[int | None] = mapped_column(nullable=True)
    byor_export_enabled: Mapped[bool] = mapped_column(nullable=False, default=False)
    sandbox_grouping_strategy: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationships
    org_members: Mapped[list['OrgMember']] = relationship(
        'OrgMember', back_populates='org'
    )
    current_users: Mapped[list['User']] = relationship(
        'User', back_populates='current_org'
    )
    invitations: Mapped[list['OrgInvitation']] = relationship(
        'OrgInvitation', back_populates='org', passive_deletes=True
    )
    billing_sessions: Mapped[list['BillingSession']] = relationship(
        'BillingSession', back_populates='org'
    )
    stored_conversation_metadata_saas: Mapped[
        list['StoredConversationMetadataSaas']
    ] = relationship('StoredConversationMetadataSaas', back_populates='org')
    user_secrets: Mapped[list['StoredCustomSecrets']] = relationship(
        'StoredCustomSecrets', back_populates='org'
    )
    api_keys: Mapped[list['ApiKey']] = relationship('ApiKey', back_populates='org')
    slack_conversations: Mapped[list['SlackConversation']] = relationship(
        'SlackConversation', back_populates='org'
    )
    slack_users: Mapped[list['SlackUser']] = relationship(
        'SlackUser', back_populates='org'
    )
    stripe_customers: Mapped[list['StripeCustomer']] = relationship(
        'StripeCustomer', back_populates='org'
    )
    git_claims: Mapped[list['OrgGitClaim']] = relationship(
        'OrgGitClaim', back_populates='org'
    )

    def __init__(self, **kwargs):
        # Serialize Pydantic model objects to dicts for JSON columns.
        from pydantic import BaseModel

        for key in ('agent_settings', 'conversation_settings'):
            if key in kwargs and isinstance(kwargs[key], BaseModel):
                kwargs[key] = kwargs[key].model_dump(mode='json')

        # Handle known SQLAlchemy columns directly
        for key in list(kwargs):
            if hasattr(self.__class__, key):
                setattr(self, key, kwargs.pop(key))

        # Handle custom property-style fields
        if 'llm_api_key' in kwargs:
            self.llm_api_key = kwargs.pop('llm_api_key')
        if 'search_api_key' in kwargs:
            self.search_api_key = kwargs.pop('search_api_key')
        if 'sandbox_api_key' in kwargs:
            self.sandbox_api_key = kwargs.pop('sandbox_api_key')

        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {list(kwargs.keys())}')

    @property
    def llm_api_key(self) -> SecretStr | None:
        if self._llm_api_key:
            decrypted = decrypt_value(self._llm_api_key)
            return SecretStr(decrypted)
        return None

    @llm_api_key.setter
    def llm_api_key(self, value: str | SecretStr | None):
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        self._llm_api_key = encrypt_value(raw) if raw else None

    @property
    def search_api_key(self) -> SecretStr | None:
        if self._search_api_key:
            decrypted = decrypt_value(self._search_api_key)
            return SecretStr(decrypted)
        return None

    @search_api_key.setter
    def search_api_key(self, value: str | SecretStr | None):
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        self._search_api_key = encrypt_value(raw) if raw else None

    @property
    def sandbox_api_key(self) -> SecretStr | None:
        if self._sandbox_api_key:
            decrypted = decrypt_value(self._sandbox_api_key)
            return SecretStr(decrypted)
        return None

    @sandbox_api_key.setter
    def sandbox_api_key(self, value: str | SecretStr | None):
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        self._sandbox_api_key = encrypt_value(raw) if raw else None
