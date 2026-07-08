from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from openhands.agent_server.env_parser import DiscriminatedUnionMixin
from openhands.app_server.config_api.config_models import AppMode
from openhands.app_server.integrations.service_types import ProviderType
from openhands.app_server.web_client.web_client_deployment_mode import (
    DeploymentMode,
    get_deployment_mode,
)


class WebClientFeatureFlags(BaseModel):
    enable_billing: bool = False
    hide_llm_settings: bool = False
    enable_jira: bool = False
    enable_jira_dc: bool = False
    enable_linear: bool = False
    hide_users_page: bool = False
    hide_billing_page: bool = False
    hide_integrations_page: bool = False
    # When true, the web client hides personal workspaces from the org list
    # and selector for users who belong to at least one team org. Used by
    # OHE installs that bootstrap a default org and want it to be the only
    # workspace users see. UI-level only — the orgs API still returns
    # personal orgs, and disabling the flag restores them.
    hide_personal_workspaces: bool = False
    # When false, the web client hides the BYOK editing UI (custom model
    # string, base URL, and API key inputs) in LLM settings, leaving only the
    # managed model dropdown. Used by OHE installs where admins curate the
    # model list on the bundled LiteLLM proxy. Defaults to True so SaaS and
    # existing installs are unaffected. UI-level only — previously saved BYOK
    # settings keep working at runtime.
    allow_user_llm_configuration: bool = True
    # Defaults to True so the ACP agent configuration UI (Settings > Agent) is
    # visible on SaaS and existing installs, matching Agent Canvas. Set
    # ENABLE_ACP=false to hide it. UI-level only.
    enable_acp: bool = True
    deployment_mode: DeploymentMode | None = None
    enable_onboarding: bool = False
    enable_automations: bool = True

    # This can be removed / replaced when a DeploymentMode (or similar) env var is created.
    @model_validator(mode='after')
    def set_deployment_mode(self) -> 'WebClientFeatureFlags':
        if self.deployment_mode is None:
            self.deployment_mode = get_deployment_mode()
        return self


class ACPModelOption(BaseModel):
    id: str
    label: str


class ACPProviderConfig(BaseModel):
    key: str
    display_name: str
    default_command: list[str]
    default_model: str | None = None
    available_models: list[ACPModelOption] = Field(default_factory=list)
    api_key_env_var: str | None = None
    base_url_env_var: str | None = None


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
    gitlab_enabled: bool = False
    provider_default_hosts: dict[str, str] = Field(default_factory=dict)
    slack_enabled: bool = False
    acp_providers: list[ACPProviderConfig] = Field(default_factory=list)
    # Hostname of the Jira Data Center server when DC OAuth is configured, so the
    # configure form can pre-fill and lock the host field (the OAuth callback only
    # accepts this exact host). None in email-match mode / when DC isn't configured.
    jira_dc_oauth_host: str | None = None
    # Optional OpenHands Enterprise/KOTS-managed Jira DC service account. When
    # configured, the frontend hides the in-app service-account secret entry and
    # the backend always prefers the env credentials at runtime.
    jira_dc_service_account_managed: bool = False
    jira_dc_service_account_email: str | None = None
    jira_dc_service_account_config_error: str | None = None
