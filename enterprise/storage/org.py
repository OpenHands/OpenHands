"""
SQLAlchemy model for Organization.
"""

from uuid import uuid4

from pydantic import SecretStr
from server.constants import DEFAULT_BILLING_MARGIN
from sqlalchemy import JSON, UUID, Boolean, Column, Float, Integer, String
from sqlalchemy.orm import relationship
from storage.base import Base
from storage.encrypt_utils import decrypt_value, encrypt_value


class Org(Base):  # type: ignore
    """Organization model."""

    __tablename__ = 'org'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String, nullable=False, unique=True)
    contact_name = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    agent = Column(String, nullable=True)
    default_max_iterations = Column(Integer, nullable=True)
    security_analyzer = Column(String, nullable=True)
    confirmation_mode = Column(Boolean, nullable=True, default=False)
    default_llm_model = Column(String, nullable=True)
    default_llm_base_url = Column(String, nullable=True)
    remote_runtime_resource_factor = Column(Integer, nullable=True)
    enable_default_condenser = Column(Boolean, nullable=False, default=True)
    billing_margin = Column(Float, nullable=True, default=DEFAULT_BILLING_MARGIN)
    enable_proactive_conversation_starters = Column(
        Boolean, nullable=False, default=True
    )
    sandbox_base_container_image = Column(String, nullable=True)
    sandbox_runtime_container_image = Column(String, nullable=True)
    org_version = Column(Integer, nullable=False, default=0)
    mcp_config = Column(JSON, nullable=True)
    # encrypted column, don't set directly, set without the underscore
    _search_api_key = Column(String, nullable=True)
    # encrypted column, don't set directly, set without the underscore
    _sandbox_api_key = Column(String, nullable=True)
    max_budget_per_task = Column(Float, nullable=True)
    enable_solvability_analysis = Column(Boolean, nullable=True, default=False)
    v1_enabled = Column(Boolean, nullable=True)
    conversation_expiration = Column(Integer, nullable=True)
    condenser_max_size = Column(Integer, nullable=True)

    # Relationships
    org_members = relationship('OrgMember', back_populates='org')
    current_users = relationship('User', back_populates='current_org')
    billing_sessions = relationship('BillingSession', back_populates='org')
    stored_conversation_metadata_saas = relationship(
        'StoredConversationMetadataSaas', back_populates='org'
    )
    user_secrets = relationship('StoredCustomSecrets', back_populates='org')
    api_keys = relationship('ApiKey', back_populates='org')
    slack_conversations = relationship('SlackConversation', back_populates='org')
    slack_users = relationship('SlackUser', back_populates='org')
    stripe_customers = relationship('StripeCustomer', back_populates='org')

    def __init__(self, **kwargs):
        # Handle known SQLAlchemy columns directly
        for key in list(kwargs):
            if hasattr(self.__class__, key):
                setattr(self, key, kwargs.pop(key))

        # Handle custom property-style fields
        if 'search_api_key' in kwargs:
            self.search_api_key = kwargs.pop('search_api_key')
        if 'sandbox_api_key' in kwargs:
            self.sandbox_api_key = kwargs.pop('sandbox_api_key')

        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {list(kwargs.keys())}')

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
