# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import uuid
from types import MappingProxyType
from typing import Any

from openhands.core.config.mcp_config import MCPConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.message import MessageAction
from openhands.experiments.experiment_manager import ExperimentManagerImpl
from openhands.integrations.provider import (
    CUSTOM_SECRETS_TYPE,
    PROVIDER_TOKEN_TYPE,
    ProviderToken,
)
from openhands.integrations.service_types import ProviderType
from openhands.server.data_models.agent_loop_info import AgentLoopInfo
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.server.shared import (
    ConversationStoreImpl,
    SecretsStoreImpl,
    SettingsStoreImpl,
    config,
    conversation_manager,
    server_config,
)
from openhands.server.types import AppMode, LLMAuthenticationError, MissingSettingsError
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.storage.data_models.secrets import Secrets
from openhands.utils.conversation_summary import get_default_conversation_title


async def initialize_conversation(
    user_id: str | None,
    conversation_id: str | None,
    selected_repository: str | None,
    selected_branch: str | None,
    conversation_trigger: ConversationTrigger = ConversationTrigger.GUI,
    git_provider: ProviderType | None = None,
) -> ConversationMetadata:
    if conversation_id is None:
        conversation_id = uuid.uuid4().hex

    conversation_store = await ConversationStoreImpl.get_instance(config, user_id)

    if not await conversation_store.exists(conversation_id):
        logger.info(
            f'New conversation ID: {conversation_id}',
            extra={'user_id': user_id, 'session_id': conversation_id},
        )

        conversation_title = get_default_conversation_title(conversation_id)

        logger.info(f'Saving metadata for conversation {conversation_id}')
        conversation_metadata = ConversationMetadata(
            trigger=conversation_trigger,
            conversation_id=conversation_id,
            title=conversation_title,
            user_id=user_id,
            selected_repository=selected_repository,
            selected_branch=selected_branch,
            git_provider=git_provider,
        )

        await conversation_store.save_metadata(conversation_metadata)
        return conversation_metadata

    conversation_metadata = await conversation_store.get_metadata(conversation_id)
    return conversation_metadata


async def start_conversation(
    user_id: str | None,
    git_provider_tokens: PROVIDER_TOKEN_TYPE | None,
    custom_secrets: CUSTOM_SECRETS_TYPE | None,
    initial_user_msg: str | None,
    image_urls: list[str] | None,
    replay_json: str | None,
    conversation_id: str,
    conversation_metadata: ConversationMetadata,
    conversation_instructions: str | None,
    mcp_config: MCPConfig | None = None,
) -> AgentLoopInfo:
    logger.info(
        'Creating conversation',
        extra={
            'signal': 'create_conversation',
            'user_id': user_id,
            'trigger': conversation_metadata.trigger,
        },
    )
    logger.info('Loading settings')
    settings_store = await SettingsStoreImpl.get_instance(config, user_id)
    settings = await settings_store.load()
    logger.info('Settings loaded')

    session_init_args: dict[str, Any] = {}
    if settings:
        session_init_args = {**settings.__dict__, **session_init_args}
        # We could use litellm.check_valid_key for a more accurate check,
        # but that would run a tiny inference.
        model_name = settings.llm_model or ''
        is_bedrock_model = model_name.startswith('bedrock/')
        is_lemonade_model = model_name.startswith('lemonade/')

        if (
            not is_bedrock_model
            and not is_lemonade_model
            and (
                not settings.llm_api_key
                or settings.llm_api_key.get_secret_value().isspace()
            )
        ):
            logger.warning(f'Missing api key for model {settings.llm_model}')
            raise LLMAuthenticationError(
                'Error authenticating with the LLM provider. Please check your API key'
            )
        elif is_bedrock_model:
            logger.info(f'Bedrock model detected ({model_name}), API key not required')

    else:
        logger.warning('Settings not present, not starting conversation')
        raise MissingSettingsError('Settings not found')

    session_init_args['git_provider_tokens'] = git_provider_tokens
    session_init_args['selected_repository'] = conversation_metadata.selected_repository
    session_init_args['custom_secrets'] = custom_secrets
    session_init_args['selected_branch'] = conversation_metadata.selected_branch
    session_init_args['git_provider'] = conversation_metadata.git_provider
    session_init_args['conversation_instructions'] = conversation_instructions
    if mcp_config:
        session_init_args['mcp_config'] = mcp_config

    conversation_init_data = ConversationInitData(**session_init_args)

    conversation_init_data = ExperimentManagerImpl.run_conversation_variant_test(
        user_id, conversation_id, conversation_init_data
    )

    logger.info(
        f'Starting agent loop for conversation {conversation_id}',
        extra={'user_id': user_id, 'session_id': conversation_id},
    )

    initial_message_action = None
    if initial_user_msg or image_urls:
        initial_message_action = MessageAction(
            content=initial_user_msg or '',
            image_urls=image_urls or [],
        )

    agent_loop_info = await conversation_manager.maybe_start_agent_loop(
        conversation_id,
        conversation_init_data,
        user_id,
        initial_user_msg=initial_message_action,
        replay_json=replay_json,
    )
    logger.info(f'Finished initializing conversation {agent_loop_info.conversation_id}')
    return agent_loop_info


async def create_new_conversation(
    user_id: str | None,
    git_provider_tokens: PROVIDER_TOKEN_TYPE | None,
    custom_secrets: CUSTOM_SECRETS_TYPE | None,
    selected_repository: str | None,
    selected_branch: str | None,
    initial_user_msg: str | None,
    image_urls: list[str] | None,
    replay_json: str | None,
    conversation_instructions: str | None = None,
    conversation_trigger: ConversationTrigger = ConversationTrigger.GUI,
    git_provider: ProviderType | None = None,
    conversation_id: str | None = None,
    mcp_config: MCPConfig | None = None,
) -> AgentLoopInfo:
    conversation_metadata = await initialize_conversation(
        user_id,
        conversation_id,
        selected_repository,
        selected_branch,
        conversation_trigger,
        git_provider,
    )

    return await start_conversation(
        user_id,
        git_provider_tokens,
        custom_secrets,
        initial_user_msg,
        image_urls,
        replay_json,
        conversation_metadata.conversation_id,
        conversation_metadata,
        conversation_instructions,
        mcp_config,
    )


def create_provider_tokens_object(
    providers_set: list[ProviderType],
) -> PROVIDER_TOKEN_TYPE:
    """Create provider tokens object for the given providers."""
    provider_information: dict[ProviderType, ProviderToken] = {}

    for provider in providers_set:
        provider_information[provider] = ProviderToken(token=None, user_id=None)

    return MappingProxyType(provider_information)


async def setup_init_conversation_settings(
    user_id: str | None,
    conversation_id: str,
    providers_set: list[ProviderType],
    provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
) -> ConversationInitData:
    """Set up conversation initialization data with provider tokens.

    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        providers_set: List of provider types to set up tokens for
        provider_tokens: Optional provider tokens to use (for SAAS mode resume)

    Returns:
        ConversationInitData with provider tokens configured
    """
    settings_store = await SettingsStoreImpl.get_instance(config, user_id)
    settings = await settings_store.load()

    secrets_store = await SecretsStoreImpl.get_instance(config, user_id)
    user_secrets: Secrets | None = await secrets_store.load()

    if not settings:
        from socketio.exceptions import ConnectionRefusedError

        raise ConnectionRefusedError(
            'Settings not found', {'msg_id': 'CONFIGURATION$SETTINGS_NOT_FOUND'}
        )

    session_init_args: dict = {}
    session_init_args = {**settings.__dict__, **session_init_args}

    # Use provided tokens if available (for SAAS resume), otherwise create scaffold
    if provider_tokens:
        logger.info(
            f'Using provided provider_tokens: {list(provider_tokens.keys())}',
            extra={'session_id': conversation_id},
        )
        git_provider_tokens = provider_tokens
    else:
        logger.info(
            f'No provider_tokens provided, creating scaffold for: {providers_set}',
            extra={'session_id': conversation_id},
        )
        git_provider_tokens = create_provider_tokens_object(providers_set)
        logger.info(
            f'Git provider scaffold: {git_provider_tokens}',
            extra={'session_id': conversation_id},
        )

        if server_config.app_mode != AppMode.SAAS and user_secrets:
            logger.info(
                f'Non-SaaS mode: Overriding with user_secrets provider tokens: {list(user_secrets.provider_tokens.keys())}',
                extra={'session_id': conversation_id},
            )
            git_provider_tokens = user_secrets.provider_tokens

    session_init_args['git_provider_tokens'] = git_provider_tokens
    if user_secrets:
        session_init_args['custom_secrets'] = user_secrets.custom_secrets

    conversation_init_data = ConversationInitData(**session_init_args)
    # We should recreate the same experiment conditions when restarting a conversation
    return ExperimentManagerImpl.run_conversation_variant_test(
        user_id, conversation_id, conversation_init_data
    )
