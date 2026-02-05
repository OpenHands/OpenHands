import hashlib
import hmac
import os
import time
import typing

import jwt
import requests  # type: ignore
from fastapi import HTTPException
from server.auth.constants import (
    BITBUCKET_APP_CLIENT_ID,
    ENABLE_ENTERPRISE_SSO,
    ENABLE_JIRA,
    ENABLE_JIRA_DC,
    ENABLE_LINEAR,
    GITHUB_APP_CLIENT_ID,
    GITHUB_APP_PRIVATE_KEY,
    GITHUB_APP_WEBHOOK_SECRET,
    GITLAB_APP_CLIENT_ID,
    RECAPTCHA_SITE_KEY,
)

from openhands.core.config.utils import load_openhands_config
from openhands.integrations.service_types import ProviderType
from openhands.server.config.server_config import ServerConfig
from openhands.server.types import AppMode

# Create a function to get config to avoid circular imports
_config = None


def get_config():
    global _config
    if _config is None:
        _config = load_openhands_config()
    return _config


def sign_token(payload: dict[str, object], jwt_secret: str, algorithm='HS256') -> str:
    """Signs a JWT token."""
    return jwt.encode(payload, jwt_secret, algorithm=algorithm)


def verify_signature(payload: bytes, signature: str):
    if not signature:
        raise HTTPException(
            status_code=403, detail='x-hub-signature-256 header is missing!'
        )

    expected_signature = (
        'sha256='
        + hmac.new(
            GITHUB_APP_WEBHOOK_SECRET.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
    )

    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=403, detail="Request signatures didn't match!")


class SaaSServerConfig(ServerConfig):
    config_cls: str = os.environ.get('OPENHANDS_CONFIG_CLS', '')
    app_mode: AppMode = AppMode.SAAS
    posthog_client_key: str = os.environ.get('POSTHOG_CLIENT_KEY', '')
    github_client_id: str = os.environ.get('GITHUB_APP_CLIENT_ID', '')
    enable_billing = os.environ.get('ENABLE_BILLING', 'false') == 'true'
    hide_llm_settings = os.environ.get('HIDE_LLM_SETTINGS', 'false') == 'true'
    auth_url: str | None = os.environ.get('AUTH_URL')
    settings_store_class: str = 'storage.saas_settings_store.SaasSettingsStore'
    secret_store_class: str = 'storage.saas_secrets_store.SaasSecretsStore'
    conversation_store_class: str = (
        'storage.saas_conversation_store.SaasConversationStore'
    )
    conversation_manager_class: str = os.environ.get(
        'CONVERSATION_MANAGER_CLASS',
        'server.clustered_conversation_manager.ClusteredConversationManager',
    )
    monitoring_listener_class: str = (
        'server.saas_monitoring_listener.SaaSMonitoringListener'
    )
    user_auth_class: str = 'server.auth.saas_user_auth.SaasUserAuth'
    # Maintenance window configuration
    maintenance_start_time: str = os.environ.get(
        'MAINTENANCE_START_TIME', ''
    )  # Timestamp in EST e.g 2025-07-29T14:18:01.219616-04:00
    enable_jira = ENABLE_JIRA
    enable_jira_dc = ENABLE_JIRA_DC
    enable_linear = ENABLE_LINEAR

    app_slug: None | str = None

    def __init__(self) -> None:
        self._get_app_slug()

    def _get_app_slug(self):
        """Retrieves the GitHub App slug using the GitHub API's /app endpoint by generating a JWT for the app

        Raises:
            HTTPException: If the request to the GitHub API fails.
        """
        if not GITHUB_APP_CLIENT_ID or not GITHUB_APP_PRIVATE_KEY:
            return

        # Generate a JWT for the GitHub App
        now = int(time.time())
        payload = {
            'iat': now - 60,  # Issued at time (backdate 60 seconds for clock skew)
            'exp': now
            + (
                9 * 60
            ),  # Expiration time (set to 9 minutes as 10 was causing error if there is time drift)
            'iss': GITHUB_APP_CLIENT_ID,  # GitHub App ID
        }

        encoded_jwt = sign_token(payload, GITHUB_APP_PRIVATE_KEY, algorithm='RS256')  # type: ignore

        # Define the headers for the GitHub API request
        headers = {
            'Authorization': f'Bearer {encoded_jwt}',
            'Accept': 'application/vnd.github+json',
        }

        # Make a request to the GitHub API /app endpoint
        response = requests.get('https://api.github.com/app', headers=headers)

        # Check if the response is successful
        if response.status_code != 200:
            raise ValueError(
                f'Failed to retrieve app info, status code:{response.status_code}, message:{response.content.decode('utf-8')}'
            )

        # Extract the app slug from the response
        app_data = response.json()
        self.app_slug = app_data.get('slug')

        if not self.app_slug:
            raise ValueError("GitHub app slug is missing in the API response.'")

    def verify_config(self):
        if not self.config_cls:
            raise ValueError('Config path not provided!')

        if not self.posthog_client_key:
            raise ValueError('Missing posthog client key in env')

        if GITHUB_APP_CLIENT_ID and not self.github_client_id:
            raise ValueError('Missing Github client id')

    def get_config(self):
        # These providers are configurable via helm charts for self hosted deployments
        # The FE should have this info so that the login buttons reflect the supported IDPs
        providers_configured = []
        if GITHUB_APP_CLIENT_ID:
            providers_configured.append(ProviderType.GITHUB)

        if GITLAB_APP_CLIENT_ID:
            providers_configured.append(ProviderType.GITLAB)

        if BITBUCKET_APP_CLIENT_ID:
            providers_configured.append(ProviderType.BITBUCKET)

        if ENABLE_ENTERPRISE_SSO:
            providers_configured.append(ProviderType.ENTERPRISE_SSO)

        config: dict[str, typing.Any] = {
            'APP_MODE': self.app_mode,
            'APP_SLUG': self.app_slug,
            'GITHUB_CLIENT_ID': self.github_client_id,
            'POSTHOG_CLIENT_KEY': self.posthog_client_key,
            'FEATURE_FLAGS': {
                'ENABLE_BILLING': self.enable_billing,
                'HIDE_LLM_SETTINGS': self.hide_llm_settings,
                'ENABLE_JIRA': self.enable_jira,
                'ENABLE_JIRA_DC': self.enable_jira_dc,
                'ENABLE_LINEAR': self.enable_linear,
            },
            'PROVIDERS_CONFIGURED': providers_configured,
        }

        # Add maintenance window if configured
        if self.maintenance_start_time:
            config['MAINTENANCE'] = {
                'startTime': self.maintenance_start_time,
            }

        if self.auth_url:
            config['AUTH_URL'] = self.auth_url

        if RECAPTCHA_SITE_KEY:
            config['RECAPTCHA_SITE_KEY'] = RECAPTCHA_SITE_KEY

        return config
