from datetime import datetime

from pydantic import BaseModel, model_validator

from openhands.agent_server.env_parser import DiscriminatedUnionMixin
from openhands.app_server.web_client.web_client_deployment_mode import (
    DeploymentMode,
    get_deployment_mode,
)
from openhands.integrations.service_types import ProviderType
from openhands.server.types import AppMode


class WebClientFeatureFlags(BaseModel):
    enable_billing: bool = False
    hide_llm_settings: bool = False
    enable_jira: bool = False
    enable_jira_dc: bool = False
    enable_linear: bool = False
    hide_users_page: bool = False
    hide_billing_page: bool = False
    hide_integrations_page: bool = False
    deployment_mode: DeploymentMode | None = None

    # This can be removed / replaced when a DeploymentMode (or similar) env var is created.
    @model_validator(mode='after')
    def set_deployment_mode(self) -> 'WebClientFeatureFlags':
        if self.deployment_mode is None:
            self.deployment_mode = get_deployment_mode()
        return self


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
    updated_at: datetime
    github_app_slug: str | None
