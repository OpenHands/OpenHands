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

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderToken,
)
from openhands.integrations.service_types import ProviderType
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.server.shared import (
    ConversationStoreImpl,
    SecretsStoreImpl,
    SettingsStoreImpl,
    config,
    server_config,
)
from openhands.server.types import AppMode
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

    session_init_args: dict = settings.model_dump()

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

    return ConversationInitData(**session_init_args)
