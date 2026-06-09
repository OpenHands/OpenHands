"""Service-account resolution for Jira Data Center integrations."""

from dataclasses import dataclass

from server.auth.token_manager import TokenManager
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.app_server.integrations.jira_dc.config import (
    get_jira_dc_service_account_env_config,
)


@dataclass(frozen=True)
class JiraDcServiceAccount:
    email: str
    api_key: str
    managed_by_env: bool


class JiraDcServiceAccountError(ValueError):
    """Base error for Jira DC service-account resolution failures."""


class JiraDcServiceAccountConfigError(JiraDcServiceAccountError):
    """Raised when env-managed Jira DC service-account config is invalid."""


class JiraDcServiceAccountResolutionError(JiraDcServiceAccountError):
    """Raised when no usable Jira DC service account can be resolved."""


def get_jira_dc_service_account_config_error() -> str | None:
    """Return a human-readable KOTS/env service-account config error, if any."""
    return get_jira_dc_service_account_env_config().error


def get_jira_dc_managed_service_account() -> JiraDcServiceAccount | None:
    """Return the env-managed service account, or None when not configured."""
    config = get_jira_dc_service_account_env_config()
    if config.error:
        raise JiraDcServiceAccountConfigError(config.error)

    if not config.is_managed:
        return None

    return JiraDcServiceAccount(
        email=config.email,
        api_key=config.api_key,
        managed_by_env=True,
    )


def is_jira_dc_service_account_managed() -> bool:
    """Return True when Jira DC service-account credentials are env-managed."""
    return get_jira_dc_service_account_env_config().is_managed


def get_jira_dc_managed_service_account_email() -> str | None:
    """Return the env-managed service-account email, if fully configured."""
    config = get_jira_dc_service_account_env_config()
    if not config.is_managed:
        return None
    return config.email


def resolve_jira_dc_service_account(
    workspace: JiraDcWorkspace,
    token_manager: TokenManager,
) -> JiraDcServiceAccount:
    """Resolve the effective Jira DC service account for runtime API calls.

    KOTS/env values are authoritative when both are set. Otherwise the existing
    per-workspace encrypted values are used for SaaS and non-managed installs.
    """
    managed_service_account = get_jira_dc_managed_service_account()
    if managed_service_account:
        return managed_service_account

    email = (workspace.svc_acc_email or '').strip()
    if not email:
        raise JiraDcServiceAccountResolutionError(
            'Jira DC workspace is missing a service account email.'
        )

    encrypted_api_key = workspace.svc_acc_api_key
    if not encrypted_api_key:
        raise JiraDcServiceAccountResolutionError(
            'Jira DC workspace is missing a service account PAT.'
        )

    return JiraDcServiceAccount(
        email=email,
        api_key=token_manager.decrypt_text(encrypted_api_key),
        managed_by_env=False,
    )
