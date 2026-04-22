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
from openhands.server.shared import ConversationStoreImpl, config
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
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
