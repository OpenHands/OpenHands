

from datetime import datetime
from pydantic import BaseModel

from openhands.agent_server.env_parser import DiscriminatedUnionMixin
from openhands.server.types import AppMode


class WebClientFeatureFlags(BaseModel):
    enable_billing: bool = False
    hide_llm_settings: bool = False
    enable_jira: bool = False
    enable_jira_dc: bool = False
    enable_linear: bool = False


class WebClientConfig(DiscriminatedUnionMixin):
    app_mode: AppMode
    app_slug: str | None
    github_client_id: str | None
    posthog_client_key: str | None
    feature_flags: WebClientFeatureFlags
    providers_configured: list[str]
    maintenance_start_time: datetime | None
    auth_url: str | None
    recaptcha_site_key: str | None
    faulty_models: list[str]
