# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openhands.core.config.llm_config import LLMConfig


class ModelRoutingConfig(BaseModel):
    """Configuration for model routing.

    Attributes:
        router_name (str): The name of the router to use. Default is 'noop_router'.
        llms_for_routing (dict[str, LLMConfig]): A dictionary mapping config names of LLMs for routing to their configurations.
    """

    router_name: str = Field(default='noop_router')
    llms_for_routing: dict[str, LLMConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra='forbid')

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, 'ModelRoutingConfig']:
        """
        Create a mapping of ModelRoutingConfig instances from a toml dictionary representing the [model_routing] section.

        The configuration is built from all keys in data.

        Returns:
            dict[str, ModelRoutingConfig]: A mapping where the key "model_routing" corresponds to the [model_routing] configuration
        """

        # Initialize the result mapping
        model_routing_mapping: dict[str, ModelRoutingConfig] = {}

        # Try to create the configuration instance
        try:
            model_routing_mapping['model_routing'] = cls.model_validate(data)
        except ValidationError as e:
            raise ValueError(f'Invalid model routing configuration: {e}')

        return model_routing_mapping
