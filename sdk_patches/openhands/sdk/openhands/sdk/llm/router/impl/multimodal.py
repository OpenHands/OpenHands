from typing import ClassVar

from pydantic import model_validator

from openhands.sdk.llm.message import Message
from openhands.sdk.llm.router.base import RouterLLM
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class MultimodalRouter(RouterLLM):
    """
    A RouterLLM implementation that routes requests based on multimodal content
    (e.g., images) and token limits. If any message contains multimodal content
    or if the token limit of the secondary model is exceeded, it routes to the
    primary model. Otherwise, it routes to the secondary model.

    Note: The primary model is expected to support multimodal content, while
    the secondary model is typically a text-only model with a lower context window.
    """

    router_name: str = "multimodal_router"

    PRIMARY_MODEL_KEY: ClassVar[str] = "primary"
    SECONDARY_MODEL_KEY: ClassVar[str] = "secondary"

    def select_llm(self, messages: list[Message]) -> str:
        """Select LLM based on multimodal content and token limits."""
        route_to_primary = False

        # Check for multimodal content in messages
        for message in messages:
            if message.contains_image:
                logger.info(
                    "Multimodal content detected in messages. "
                    "Routing to the primary model."
                )
                route_to_primary = True

        # Check if `messages` exceeds context window of the secondary model
        # Assuming the secondary model has a lower context window limit
        # compared to the primary model
        secondary_llm = self.llms_for_routing.get(self.SECONDARY_MODEL_KEY)
        if secondary_llm and (
            secondary_llm.max_input_tokens
            and secondary_llm.get_token_count(messages) > secondary_llm.max_input_tokens
        ):
            logger.warning(
                f"Messages having {secondary_llm.get_token_count(messages)} tokens, exceeded secondary model's max input tokens ({secondary_llm.max_input_tokens} tokens). "  # noqa: E501
                "Routing to the primary model."
            )
            route_to_primary = True

        if route_to_primary:
            logger.info("Routing to the primary model...")
            return self.PRIMARY_MODEL_KEY
        else:
            logger.info("Routing to the secondary model...")
            return self.SECONDARY_MODEL_KEY

    @model_validator(mode="after")
    def _validate_llms_for_routing(self) -> "MultimodalRouter":
        """Ensure required models are present in llms_for_routing."""
        if self.PRIMARY_MODEL_KEY not in self.llms_for_routing:
            raise ValueError(
                f"Primary LLM key '{self.PRIMARY_MODEL_KEY}' not found"
                " in llms_for_routing."
            )
        if self.SECONDARY_MODEL_KEY not in self.llms_for_routing:
            raise ValueError(
                f"Secondary LLM key '{self.SECONDARY_MODEL_KEY}' not found"
                " in llms_for_routing."
            )
        return self
