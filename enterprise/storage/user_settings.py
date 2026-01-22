from server.constants import DEFAULT_BILLING_MARGIN
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Identity, Integer, String
from storage.base import Base


class UserSettings(Base):  # type: ignore
    __tablename__ = 'user_settings'
    id = Column(Integer, Identity(), primary_key=True)
    keycloak_user_id = Column(String, nullable=True, index=True)
    language = Column(String, nullable=True)
    agent = Column(String, nullable=True)
    max_iterations = Column(Integer, nullable=True)
    security_analyzer = Column(String, nullable=True)
    confirmation_mode = Column(Boolean, nullable=True, default=False)
    llm_model = Column(String, nullable=True)
    llm_api_key = Column(String, nullable=True)
    llm_api_key_for_byor = Column(String, nullable=True)
    llm_base_url = Column(String, nullable=True)
    remote_runtime_resource_factor = Column(Integer, nullable=True)
    enable_default_condenser = Column(Boolean, nullable=False, default=True)
    condenser_max_size = Column(Integer, nullable=True)
    user_consents_to_analytics = Column(Boolean, nullable=True)
    billing_margin = Column(Float, nullable=True, default=DEFAULT_BILLING_MARGIN)
    enable_sound_notifications = Column(Boolean, nullable=True, default=False)
    enable_proactive_conversation_starters = Column(
        Boolean, nullable=False, default=True
    )
    sandbox_base_container_image = Column(String, nullable=True)
    sandbox_runtime_container_image = Column(String, nullable=True)
    user_version = Column(Integer, nullable=False, default=0)
    accepted_tos = Column(DateTime, nullable=True)
    mcp_config = Column(JSON, nullable=True)
    search_api_key = Column(String, nullable=True)
    sandbox_api_key = Column(String, nullable=True)
    max_budget_per_task = Column(Float, nullable=True)
    enable_solvability_analysis = Column(Boolean, nullable=True, default=False)
    email = Column(String, nullable=True)
    email_verified = Column(Boolean, nullable=True)
    git_user_name = Column(String, nullable=True)
    git_user_email = Column(String, nullable=True)
    v1_enabled = Column(Boolean, nullable=True)
    already_migrated = Column(
        Boolean, nullable=True, default=False
    )  # False = not migrated, True = migrated
