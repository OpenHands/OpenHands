# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.llm.router.base import ROUTER_LLM_REGISTRY, RouterLLM


class MultimodalRouter(RouterLLM):
    SECONDARY_MODEL_CONFIG_NAME = 'secondary_model'
    ROUTER_NAME = 'multimodal_router'

    def __init__(
        self,
        agent_config: AgentConfig,
        llm_registry: LLMRegistry,
        **kwargs,
    ):
        super().__init__(agent_config, llm_registry, **kwargs)

        self._validate_model_routing_config(self.llms_for_routing)

        # States
        self.max_token_exceeded = False

    def _select_llm(self, messages: list[Message]) -> str:
        """Select LLM based on multimodal content and token limits."""
        route_to_primary = False

        # Check for multimodal content in messages
        for message in messages:
            if message.contains_image:
                logger.info(
                    'Multimodal content detected in messages. Routing to the primary model.'
                )
                route_to_primary = True

        if not route_to_primary and self.max_token_exceeded:
            route_to_primary = True

        # Check if `messages` exceeds context window of the secondary model
        # Assuming the secondary model has a lower context window limit compared to the primary model
        secondary_llm = self.available_llms.get(self.SECONDARY_MODEL_CONFIG_NAME)
        if secondary_llm and (
            secondary_llm.config.max_input_tokens
            and secondary_llm.get_token_count(messages)
            > secondary_llm.config.max_input_tokens
        ):
            logger.warning(
                f"Messages having {secondary_llm.get_token_count(messages)} tokens, exceed secondary model's max input tokens ({secondary_llm.config.max_input_tokens} tokens). "
                'Routing to the primary model.'
            )
            self.max_token_exceeded = True
            route_to_primary = True

        if route_to_primary:
            logger.info('Routing to the primary model...')
            return 'primary'
        else:
            logger.info('Routing to the secondary model...')
            return self.SECONDARY_MODEL_CONFIG_NAME

    def vision_is_active(self):
        return self.primary_llm.vision_is_active()

    def _validate_model_routing_config(self, llms_for_routing: dict[str, LLM]):
        if self.SECONDARY_MODEL_CONFIG_NAME not in llms_for_routing:
            raise ValueError(
                f'Secondary LLM config {self.SECONDARY_MODEL_CONFIG_NAME} not found.'
            )


# Register the router
ROUTER_LLM_REGISTRY[MultimodalRouter.ROUTER_NAME] = MultimodalRouter
