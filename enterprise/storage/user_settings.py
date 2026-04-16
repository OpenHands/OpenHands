from __future__ import annotations

from pydantic import SecretStr
from server.constants import DEFAULT_BILLING_MARGIN
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Identity, Integer, String
from storage.base import Base
from storage.encrypt_utils import decrypt_legacy_value, encrypt_legacy_value


class UserSettings(Base):  # type: ignore
    __tablename__ = 'user_settings'
    id = Column(Integer, Identity(), primary_key=True)
    keycloak_user_id = Column(String, nullable=True, index=True)
    language = Column(String, nullable=True)
    # Deprecated (v0): API keys now live on Org / OrgMember.
    # Kept for backward-compat during migration; do not use in new code.
    llm_api_key = Column(String, nullable=True)
    llm_api_key_for_byor = Column(String, nullable=True)
    remote_runtime_resource_factor = Column(Integer, nullable=True)
    user_consents_to_analytics = Column(Boolean, nullable=True)
    billing_margin = Column(Float, nullable=True, default=DEFAULT_BILLING_MARGIN)
    enable_sound_notifications = Column(Boolean, nullable=True, default=False)
    enable_proactive_conversation_starters = Column(
        Boolean, nullable=False, default=True
    )
    sandbox_base_container_image = Column(String, nullable=True)
    sandbox_runtime_container_image = Column(String, nullable=True)
    sandbox_grouping_strategy = Column(String, nullable=True)
    user_version = Column(Integer, nullable=False, default=0)
    accepted_tos = Column(DateTime, nullable=True)
    # Deprecated (v0): mcp_config now lives inside AgentSettings on Org / OrgMember.
    mcp_config = Column(JSON, nullable=True)
    disabled_skills = Column(JSON, nullable=True)
    search_api_key = Column(String, nullable=True)
    sandbox_api_key = Column(String, nullable=True)
    max_budget_per_task = Column(Float, nullable=True)
    enable_solvability_analysis = Column(Boolean, nullable=True, default=False)
    email = Column(String, nullable=True)
    email_verified = Column(Boolean, nullable=True)
    git_user_name = Column(String, nullable=True)
    git_user_email = Column(String, nullable=True)
    v1_enabled = Column(Boolean, nullable=True)
    agent_settings = Column(JSON, nullable=False, default=dict)
    conversation_settings = Column(JSON, nullable=False, default=dict)

    @property
    def llm_api_key_for_byor_secret(self) -> SecretStr | None:
        raw = self.llm_api_key_for_byor
        if not raw:
            return None
        try:
            return SecretStr(decrypt_legacy_value(raw))
        except Exception:
            return SecretStr(raw)

    @llm_api_key_for_byor_secret.setter
    def llm_api_key_for_byor_secret(self, value: str | SecretStr | None) -> None:
        if value is None:
            self.llm_api_key_for_byor = None
            return
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        self.llm_api_key_for_byor = encrypt_legacy_value(raw)

    already_migrated = Column(
        Boolean, nullable=True, default=False
    )  # False = not migrated, True = migrated

    def to_settings(self):
        from openhands.sdk.settings import AgentSettings, ConversationSettings
        from openhands.storage.data_models.settings import Settings

        return Settings(
            agent_settings=AgentSettings.model_validate(self.agent_settings or {}),
            conversation_settings=ConversationSettings.model_validate(
                self.conversation_settings or {}
            ),
        )
