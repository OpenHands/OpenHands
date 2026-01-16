from datetime import datetime

from pydantic import BaseModel

from openhands.agent_server.env_parser import DiscriminatedUnionMixin
from openhands.integrations.service_types import ProviderType
from openhands.server.types import AppMode


class WebClientFeatureFlags(BaseModel):
    enable_billing: bool = False
    hide_llm_settings: bool = False
    enable_jira: bool = False
    enable_jira_dc: bool = False
    enable_linear: bool = False


class WebClientConfig(DiscriminatedUnionMixin):
    app_mode: AppMode
    posthog_client_key: str | None
    feature_flags: WebClientFeatureFlags
    providers_configured: list[ProviderType]
    maintenance_start_time: datetime | None
    auth_url: str | None
    recaptcha_site_key: str | None
    faulty_models: list[str]
    error_message: str | None
