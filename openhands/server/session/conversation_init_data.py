# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ConfigDict, Field, field_validator

from openhands.integrations.provider import CUSTOM_SECRETS_TYPE, PROVIDER_TOKEN_TYPE
from openhands.integrations.service_types import ProviderType
from openhands.storage.data_models.settings import Settings


class ConversationInitData(Settings):
    """Session initialization data for the web environment - a deep copy of the global config is made and then overridden with this data."""

    git_provider_tokens: PROVIDER_TOKEN_TYPE | None = Field(default=None)
    custom_secrets: CUSTOM_SECRETS_TYPE | None = Field(default=None)
    selected_repository: str | None = Field(default=None)
    replay_json: str | None = Field(default=None)
    selected_branch: str | None = Field(default=None)
    conversation_instructions: str | None = Field(default=None)
    git_provider: ProviderType | None = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    @field_validator('git_provider_tokens', 'custom_secrets')
    @classmethod
    def immutable_validator(cls, value: Mapping | None) -> MappingProxyType | None:
        """Ensure git_provider_tokens and custom_secrets are always MappingProxyType.

        This validator converts any Mapping (including dict) to MappingProxyType,
        ensuring type safety and immutability. If the value is None, it returns None.
        """
        if value is None:
            return None
        if isinstance(value, MappingProxyType):
            return value
        return MappingProxyType(value)
