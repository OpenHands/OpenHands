import os
from copy import deepcopy

from pydantic import SecretStr

from openhands.app_server.settings.settings_models import Settings
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.utils.environment import get_effective_llm_base_url


def setup_llm_config(config: OpenHandsConfig, settings: Settings) -> OpenHandsConfig:
    # Copying this means that when we update variables they are not applied to the shared global configuration!
    config = deepcopy(config)

    agent_settings = settings.agent_settings
    llm_config = config.get_llm_config()
    llm_config.model = agent_settings.llm.model
    raw_key = settings.agent_settings.llm.api_key
    if isinstance(raw_key, str):
        llm_config.api_key = SecretStr(raw_key)
    else:
        llm_config.api_key = raw_key
    env_base_url = os.environ.get('LLM_BASE_URL')
    settings_base_url = agent_settings.llm.base_url

    # Use env_base_url if available, otherwise fall back to settings_base_url
    base_url_to_use = (
        env_base_url if env_base_url not in (None, '') else settings_base_url
    )

    llm_config.base_url = get_effective_llm_base_url(
        llm_config.model,
        base_url_to_use,
        llm_config.custom_llm_provider,
    )
    config.set_llm_config(llm_config)
    return config
