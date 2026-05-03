# TODO: Merge this module with openhands.app_server.config
import os

from openhands.app_server.types import AppMode, ServerConfigInterface
from openhands.app_server.utils.import_utils import get_impl
from openhands.app_server.utils.logger import openhands_logger as logger


class ServerConfig(ServerConfigInterface):
    config_cls = os.environ.get('OPENHANDS_CONFIG_CLS', None)
    app_mode = AppMode.OPENHANDS
    posthog_client_key = 'phc_3ESMmY9SgqEAGBB6sMGK5ayYHkeUuknH2vP6FmWH9RA'
    github_client_id = os.environ.get('GITHUB_APP_CLIENT_ID', '')
    enable_billing = os.environ.get('ENABLE_BILLING', 'false') == 'true'
    hide_llm_settings = os.environ.get('HIDE_LLM_SETTINGS', 'false') == 'true'
    # This config is used to hide the microagent management page from the users for now. We will remove this once we release the new microagent management page.
    settings_store_class: str = (
        'openhands.app_server.settings.file_settings_store.FileSettingsStore'
    )
    secret_store_class: str = (
        'openhands.app_server.secrets.file_secrets_store.FileSecretsStore'
    )
    user_auth_class: str = (
        'openhands.app_server.user_auth.default_user_auth.DefaultUserAuth'
    )
    enable_v1: bool = os.getenv('ENABLE_V1') != '0'

    def verify_config(self):
        if self.config_cls:
            raise ValueError('Unexpected config path provided')

    def get_config(self):
        config = {
            'APP_MODE': self.app_mode,
            'GITHUB_CLIENT_ID': self.github_client_id,
            'POSTHOG_CLIENT_KEY': self.posthog_client_key,
            'FEATURE_FLAGS': {
                'ENABLE_BILLING': self.enable_billing,
                'HIDE_LLM_SETTINGS': self.hide_llm_settings,
            },
        }

        return config


def load_server_config() -> ServerConfig:
    config_cls = os.environ.get('OPENHANDS_CONFIG_CLS', None)
    logger.info(f'Using config class {config_cls}')

    server_config_cls = get_impl(ServerConfig, config_cls)
    server_config: ServerConfig = server_config_cls()
    server_config.verify_config()

    return server_config
