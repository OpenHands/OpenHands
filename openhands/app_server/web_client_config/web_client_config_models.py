

from datetime import datetime
from pydantic import BaseModel, Field

from openhands.agent_server.env_parser import DiscriminatedUnionMixin
from openhands.server.types import AppMode


class WebClientFeatureFlags(BaseModel):
    enable_billing: bool = False
    hide_llm_settings: bool = False
    enable_jira: bool = False
    enable_jira_dc: bool = False
    enable_linear: bool = False


class WebClientConfig(DiscriminatedUnionMixin):
    app_mode: AppMode = AppMode.OPENHANDS
    app_slug: str | None = None
    github_client_id: str | None = None
    posthog_client_key: str | None = "phc_3ESMmY9SgqEAGBB6sMGK5ayYHkeUuknH2vP6FmWH9RA"
    feature_flags: WebClientFeatureFlags = Field(default_factory=WebClientFeatureFlags)
    provider_configured: list[str] = Field(default_factory=list)
    maintenance_start_time: datetime | None = None
    auth_url: str | None = None
    recaptcha_site_key: str | None = None
