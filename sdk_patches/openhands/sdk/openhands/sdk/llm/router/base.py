from abc import abstractmethod
from collections.abc import Sequence

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from openhands.sdk.llm.llm import LLM
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.message import Message
from openhands.sdk.llm.streaming import TokenCallbackType
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.tool import ToolDefinition


logger = get_logger(__name__)


class RouterLLM(LLM):
    """
    Base class for multiple LLM acting as a unified LLM.
    This class provides a foundation for implementing model routing by
    inheriting from LLM, allowing routers to work with multiple underlying
    LLM models while presenting a unified LLM interface to consumers.
    Key features:
    - Works with multiple LLMs configured via llms_for_routing
    - Delegates all other operations/properties to the selected LLM
    - Provides routing interface through select_llm() method
    """

    router_name: str = Field(default="base_router", description="Name of the router")
    llms_for_routing: dict[str, LLM] = Field(
        default_factory=dict
    )  # Mapping of LLM name to LLM instance for routing
    active_llm: LLM | None = Field(
        default=None, description="Currently selected LLM instance"
    )

    @field_validator("llms_for_routing")
    @classmethod
    def validate_llms_not_empty(cls, v):
        if not v:
            raise ValueError(
                "llms_for_routing cannot be empty - at least one LLM must be provided"
            )
        return v

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        This method intercepts completion calls and routes them to the appropriate
        underlying LLM based on the routing logic implemented in select_llm().
        """
        # Select appropriate LLM
        selected_model = self.select_llm(messages)
        self.active_llm = self.llms_for_routing[selected_model]

        logger.info(f"RouterLLM routing to {selected_model}...")

        # Delegate to selected LLM
        return self.active_llm.completion(
            messages=messages,
            tools=tools,
            _return_metrics=return_metrics,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token,
            **kwargs,
        )

    @abstractmethod
    def select_llm(self, messages: list[Message]) -> str:
        """Select which LLM to use based on messages and events.

        This method implements the core routing logic for the RouterLLM.
        Subclasses should analyze the provided messages to determine which
        LLM from llms_for_routing is most appropriate for handling the request.

        Args:
            messages: List of messages in the conversation that can be used
                     to inform the routing decision.

        Returns:
            The key/name of the LLM to use from llms_for_routing dictionary.
        """

    def __getattr__(self, name):
        """Delegate other attributes/methods to the active LLM."""
        fallback_llm = next(iter(self.llms_for_routing.values()))
        logger.info(f"RouterLLM: No active LLM, using first LLM for attribute '{name}'")
        return getattr(fallback_llm, name)

    def __str__(self) -> str:
        """String representation of the router."""
        return f"{self.__class__.__name__}(llms={list(self.llms_for_routing.keys())})"

    @model_validator(mode="before")
    @classmethod
    def set_placeholder_model(cls, data):
        """Guarantee `model` exists before LLM base validation runs."""
        if not isinstance(data, dict):
            return data
        d = dict(data)

        # In router, we don't need a model name to be specified
        if "model" not in d or not d["model"]:
            d["model"] = d.get("router_name", "router")

        return d
