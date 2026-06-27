"""Shared Jira Data Center environment configuration helpers."""

import os
import re
from dataclasses import dataclass

_EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


@dataclass(frozen=True)
class JiraDcServiceAccountEnvConfig:
    email: str
    api_key: str
    error: str | None

    @property
    def is_managed(self) -> bool:
        return bool(self.email and self.api_key) and self.error is None


def get_jira_dc_service_account_env_config() -> JiraDcServiceAccountEnvConfig:
    """Return env-managed Jira DC service-account config and validation state."""
    email = os.getenv('JIRA_DC_SERVICE_ACCOUNT_EMAIL', '').strip()
    api_key = os.getenv('JIRA_DC_SERVICE_ACCOUNT_PAT', '').strip()
    error = _validate_jira_dc_service_account_env(email, api_key)
    return JiraDcServiceAccountEnvConfig(email=email, api_key=api_key, error=error)


def _validate_jira_dc_service_account_env(
    email: str,
    api_key: str,
) -> str | None:
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
