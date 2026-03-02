import os
from datetime import datetime

from pydantic import Field

from openhands.app_server.web_client.web_client_config_injector import (
    WebClientConfigInjector,
)
from openhands.app_server.web_client.web_client_models import (
    WebClientConfig,
    WebClientFeatureFlags,
)
from openhands.integrations.service_types import ProviderType


def _get_recaptcha_site_key() -> str | None:
    """Get reCAPTCHA site key from environment variable."""
    key = os.getenv('RECAPTCHA_SITE_KEY', '').strip()
    return key if key else None


# OSS default PostHog key - used when no environment variable is configured
_OSS_POSTHOG_KEY = 'phc_3ESMmY9SgqEAGBB6sMGK5ayYHkeUuknH2vP6FmWH9RA'


def _get_posthog_client_key() -> str:
    """Get PostHog client key from environment variable.

    Reads POSTHOG_CLIENT_KEY from environment. If not set or empty,
    returns the OSS default key for backwards compatibility.
    """
    key = os.getenv('POSTHOG_CLIENT_KEY', '').strip()
    return key if key else _OSS_POSTHOG_KEY


def _get_auth_url() -> str | None:
    """Get authentication service URL from environment variable.

    Reads AUTH_URL from environment. If not set or empty, returns None.
    """
    url = os.getenv('AUTH_URL', '').strip()
    return url if url else None


def _get_maintenance_start_time() -> datetime | None:
    """Get maintenance start time from environment variable.

    Reads MAINTENANCE_START_TIME from environment. If set to a valid ISO 8601
    timestamp, returns the parsed datetime. If empty, unset, or invalid,
    returns None (graceful fallback).
    """
    value = os.getenv('MAINTENANCE_START_TIME', '').strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _get_providers_configured() -> list[ProviderType]:
    """Get configured OAuth providers from environment variables.

    Checks for presence of OAuth client ID env vars and returns a list of
    configured providers. Mirrors legacy logic from SaaSServerConfig.
    """
    providers: list[ProviderType] = []

    if os.getenv('GITHUB_APP_CLIENT_ID', '').strip():
        providers.append(ProviderType.GITHUB)

    if os.getenv('GITLAB_APP_CLIENT_ID', '').strip():
        providers.append(ProviderType.GITLAB)

    if os.getenv('BITBUCKET_APP_CLIENT_ID', '').strip():
        providers.append(ProviderType.BITBUCKET)

    if os.getenv('ENABLE_ENTERPRISE_SSO', '').strip():
        providers.append(ProviderType.ENTERPRISE_SSO)

    return providers


def _get_github_app_slug() -> str | None:
    """Get GitHub app slug from environment variable.

    Reads GITHUB_APP_SLUG from environment. If set, returns the value.
    If empty or unset, returns None.
    """
    slug = os.getenv('GITHUB_APP_SLUG', '').strip()
    return slug if slug else None


def _get_feature_flags() -> WebClientFeatureFlags:
    """Get feature flags from environment variables.

    Reads ENABLE_BILLING, HIDE_LLM_SETTINGS, ENABLE_JIRA, ENABLE_JIRA_DC,
    and ENABLE_LINEAR from environment. Each flag is True only if the
    corresponding env var is exactly 'true', otherwise False.
    """
    return WebClientFeatureFlags(
        enable_billing=os.getenv('ENABLE_BILLING', 'false') == 'true',
        hide_llm_settings=os.getenv('HIDE_LLM_SETTINGS', 'false') == 'true',
        enable_jira=os.getenv('ENABLE_JIRA', 'false') == 'true',
        enable_jira_dc=os.getenv('ENABLE_JIRA_DC', 'false') == 'true',
        enable_linear=os.getenv('ENABLE_LINEAR', 'false') == 'true',
    )


class DefaultWebClientConfigInjector(WebClientConfigInjector):
    posthog_client_key: str = Field(default_factory=_get_posthog_client_key)
    feature_flags: WebClientFeatureFlags = Field(default_factory=_get_feature_flags)
    providers_configured: list[ProviderType] = Field(
        default_factory=_get_providers_configured
    )
    maintenance_start_time: datetime | None = Field(
        default_factory=_get_maintenance_start_time
    )
    auth_url: str | None = Field(default_factory=_get_auth_url)
    recaptcha_site_key: str | None = Field(default_factory=_get_recaptcha_site_key)
    faulty_models: list[str] = Field(default_factory=list)
    error_message: str | None = None
    updated_at: datetime = Field(
        default=datetime.fromisoformat('2026-01-01T00:00:00Z'),
        description=(
            'The timestamp when error messages and faulty models were last updated. '
            'The frontend uses this value to determine whether error messages are '
            'new and should be displayed. (Default to start of 2026)'
        ),
    )
    github_app_slug: str | None = Field(default_factory=_get_github_app_slug)

    async def get_web_client_config(self) -> WebClientConfig:
        from openhands.app_server.config import get_global_config

        config = get_global_config()
        result = WebClientConfig(
            app_mode=config.app_mode,
            posthog_client_key=self.posthog_client_key,
            feature_flags=self.feature_flags,
            providers_configured=self.providers_configured,
            maintenance_start_time=self.maintenance_start_time,
            auth_url=self.auth_url,
            recaptcha_site_key=self.recaptcha_site_key,
            faulty_models=self.faulty_models,
            error_message=self.error_message,
            updated_at=self.updated_at,
            github_app_slug=self.github_app_slug,
        )
        return result
