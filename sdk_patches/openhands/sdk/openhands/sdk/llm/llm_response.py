"""LLMResponse type for LLM completion responses.

This module provides the LLMResponse type that wraps LLM completion responses
with OpenHands-native types, eliminating the need for consumers to work directly
with LiteLLM types.
"""

import warnings
from typing import ClassVar

from litellm import ResponsesAPIResponse
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, ConfigDict

from openhands.sdk.llm.message import Message
from openhands.sdk.llm.utils.metrics import MetricsSnapshot


# Suppress Pydantic serializer warnings from litellm
# These warnings occur when Pydantic serializes litellm's ModelResponse objects
# that have mismatched field counts, which is expected behavior in litellm
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


__all__ = ["LLMResponse"]


class LLMResponse(BaseModel):
    """Result of an LLM completion request.

    This type provides a clean interface for LLM completion results, exposing
    only OpenHands-native types to consumers while preserving access to the
    raw LiteLLM response for internal use.

    Attributes:
        message: The completion message converted to OpenHands Message type
        metrics: Snapshot of metrics from the completion request
        raw_response: The original LiteLLM response (ModelResponse or
            ResponsesAPIResponse) for internal use
    """

    message: Message
    metrics: MetricsSnapshot
    raw_response: ModelResponse | ResponsesAPIResponse

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @property
    def id(self) -> str:
        """Get the response ID from the underlying LLM response.

        This property provides a clean interface to access the response ID,
        supporting both completion mode (ModelResponse) and response API modes
        (ResponsesAPIResponse).

        Returns:
            The response ID from the LLM response
        """
        return self.raw_response.id
