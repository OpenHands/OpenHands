import os
from copy import deepcopy

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.llm.llm_registry import LLMRegistry
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage import get_file_store
from openhands.storage.data_models.settings import Settings
from openhands.utils.environment import get_effective_llm_base_url


def setup_llm_config(config: OpenHandsConfig, settings: Settings) -> OpenHandsConfig:
    # Copying this means that when we update variables they are not applied to the shared global configuration!
    config = deepcopy(config)

    llm_config = config.get_llm_config()
    llm_config.model = settings.llm_model or ''
    llm_config.api_key = settings.llm_api_key
    env_base_url = os.environ.get('LLM_BASE_URL')
    settings_base_url = settings.llm_base_url

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


def create_registry_and_conversation_stats(
    config: OpenHandsConfig,
    sid: str,
    user_id: str | None,
    user_settings: Settings | None = None,
) -> tuple[LLMRegistry, ConversationStats, OpenHandsConfig]:
    user_config = config
    if user_settings:
        user_config = setup_llm_config(config, user_settings)

    agent_cls = user_settings.agent if user_settings else None
    llm_registry = LLMRegistry(user_config, agent_cls)
    file_store = get_file_store(
        file_store_type=config.file_store,
        file_store_path=config.file_store_path,
        file_store_web_hook_url=config.file_store_web_hook_url,
        file_store_web_hook_headers=config.file_store_web_hook_headers,
        file_store_web_hook_batch=config.file_store_web_hook_batch,
    )
    conversation_stats = ConversationStats(file_store, sid, user_id)
    llm_registry.subscribe(conversation_stats.register_llm)
    return llm_registry, conversation_stats, user_config
