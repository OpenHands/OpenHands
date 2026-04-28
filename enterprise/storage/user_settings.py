from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import SecretStr
from server.constants import DEFAULT_BILLING_MARGIN
from sqlalchemy import DateTime, Identity, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base
from storage.encrypt_utils import decrypt_legacy_value, encrypt_legacy_value


class UserSettings(Base):
    __tablename__ = 'user_settings'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    keycloak_user_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    # Deprecated (v0): API keys now live on Org / OrgMember.
    # Kept for backward-compat during migration; do not use in new code.
    llm_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_api_key_for_byor: Mapped[str | None] = mapped_column(String, nullable=True)
    remote_runtime_resource_factor: Mapped[int | None] = mapped_column(nullable=True)
    user_consents_to_analytics: Mapped[bool | None] = mapped_column(nullable=True)
    billing_margin: Mapped[float | None] = mapped_column(
        nullable=True, default=DEFAULT_BILLING_MARGIN
    )
    enable_sound_notifications: Mapped[bool | None] = mapped_column(
        nullable=True, default=False
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
    sandbox_grouping_strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    user_version: Mapped[int] = mapped_column(nullable=False, default=0)
    accepted_tos: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Deprecated (v0): mcp_config now lives inside AgentSettings on Org / OrgMember.
    mcp_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    disabled_skills: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    search_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    sandbox_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    max_budget_per_task: Mapped[float | None] = mapped_column(nullable=True)
    enable_solvability_analysis: Mapped[bool | None] = mapped_column(
        nullable=True, default=False
    )
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    email_verified: Mapped[bool | None] = mapped_column(nullable=True)
    git_user_name: Mapped[str | None] = mapped_column(String, nullable=True)
    git_user_email: Mapped[str | None] = mapped_column(String, nullable=True)
    v1_enabled: Mapped[bool | None] = mapped_column(nullable=True)
    agent_settings: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    conversation_settings: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

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

    already_migrated: Mapped[bool | None] = mapped_column(
        nullable=True, default=False
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
