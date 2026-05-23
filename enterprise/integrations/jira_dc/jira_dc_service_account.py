"""Service-account resolution for Jira Data Center integrations."""

import re
from dataclasses import dataclass

from server.auth.constants import (
    JIRA_DC_SERVICE_ACCOUNT_EMAIL,
    JIRA_DC_SERVICE_ACCOUNT_PAT,
)
from server.auth.token_manager import TokenManager
from storage.jira_dc_workspace import JiraDcWorkspace

_EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


@dataclass(frozen=True)
class JiraDcServiceAccount:
    email: str
    api_key: str
    managed_by_env: bool


def get_jira_dc_service_account_config_error() -> str | None:
    """Return a human-readable KOTS/env service-account config error, if any."""
    email = JIRA_DC_SERVICE_ACCOUNT_EMAIL
    api_key = JIRA_DC_SERVICE_ACCOUNT_PAT

    if bool(email) != bool(api_key):
        return (
            'Jira DC service account is partially configured. Set both '
            'JIRA_DC_SERVICE_ACCOUNT_EMAIL and JIRA_DC_SERVICE_ACCOUNT_PAT, '
            'or clear both to configure service accounts in OpenHands.'
        )

    if email and not _EMAIL_RE.match(email):
        return 'JIRA_DC_SERVICE_ACCOUNT_EMAIL must be a valid email address.'

    if api_key and any(char.isspace() for char in api_key):
        return 'JIRA_DC_SERVICE_ACCOUNT_PAT cannot contain whitespace.'

    return None


def get_jira_dc_managed_service_account() -> JiraDcServiceAccount | None:
    """Return the env-managed service account, or None when not configured."""
    config_error = get_jira_dc_service_account_config_error()
    if config_error:
        raise ValueError(config_error)

    if not JIRA_DC_SERVICE_ACCOUNT_EMAIL or not JIRA_DC_SERVICE_ACCOUNT_PAT:
        return None

    return JiraDcServiceAccount(
        email=JIRA_DC_SERVICE_ACCOUNT_EMAIL,
        api_key=JIRA_DC_SERVICE_ACCOUNT_PAT,
        managed_by_env=True,
    )


def is_jira_dc_service_account_managed() -> bool:
    """Return True when Jira DC service-account credentials are env-managed."""
    return (
        get_jira_dc_service_account_config_error() is None
        and bool(JIRA_DC_SERVICE_ACCOUNT_EMAIL)
        and bool(JIRA_DC_SERVICE_ACCOUNT_PAT)
    )


def get_jira_dc_managed_service_account_email() -> str | None:
    """Return the env-managed service-account email, if fully configured."""
    if not is_jira_dc_service_account_managed():
        return None
    return JIRA_DC_SERVICE_ACCOUNT_EMAIL


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
        raise ValueError('Jira DC workspace is missing a service account email.')

    encrypted_api_key = workspace.svc_acc_api_key
    if not encrypted_api_key:
        raise ValueError('Jira DC workspace is missing a service account PAT.')

    return JiraDcServiceAccount(
        email=email,
        api_key=token_manager.decrypt_text(encrypted_api_key),
        managed_by_env=False,
    )
