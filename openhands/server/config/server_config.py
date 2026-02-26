# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import os

from openhands.core.logger import openhands_logger as logger
from openhands.server.types import AppMode, ServerConfigInterface
from openhands.utils.import_utils import get_impl


class ServerConfig(ServerConfigInterface):
    config_cls = os.environ.get('OPENHANDS_CONFIG_CLS', None)
    app_mode = AppMode.OPENHANDS
    posthog_client_key = 'phc_3ESMmY9SgqEAGBB6sMGK5ayYHkeUuknH2vP6FmWH9RA'
    github_client_id = os.environ.get('GITHUB_APP_CLIENT_ID', '')
    enable_billing = os.environ.get('ENABLE_BILLING', 'false') == 'true'
    hide_llm_settings = os.environ.get('HIDE_LLM_SETTINGS', 'false') == 'true'
    # This config is used to hide the microagent management page from the users for now. We will remove this once we release the new microagent management page.
    settings_store_class: str = (
        'openhands.storage.settings.file_settings_store.FileSettingsStore'
    )
    secret_store_class: str = (
        'openhands.storage.secrets.file_secrets_store.FileSecretsStore'
    )
    conversation_store_class: str = (
        'openhands.storage.conversation.file_conversation_store.FileConversationStore'
    )
    conversation_manager_class: str = os.environ.get(
        'CONVERSATION_MANAGER_CLASS',
        'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager',
    )
    monitoring_listener_class: str = 'openhands.server.monitoring.MonitoringListener'
    user_auth_class: str = (
        'openhands.server.user_auth.default_user_auth.DefaultUserAuth'
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
