# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import copy
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics

if TYPE_CHECKING:
    from openhands.llm.llm_registry import LLMRegistry

ROUTER_LLM_REGISTRY: dict[str, type['RouterLLM']] = {}


class RouterLLM(LLM):
    """
    Base class for multiple LLM acting as a unified LLM.

    This class provides a foundation for implementing model routing by inheriting from LLM,
    allowing routers to work with multiple underlying LLM models while presenting a unified
    LLM interface to consumers.

    Key features:
    - Works with multiple LLMs configured via llms_for_routing
    - Delegates all other operations/properties to the selected LLM
    - Provides routing interface through _select_llm() method
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        llm_registry: 'LLMRegistry',
        service_id: str = 'router_llm',
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ):
        """
        Initialize RouterLLM with multiple LLM support.
        """
        self.llm_registry = llm_registry
        self.model_routing_config = agent_config.model_routing

        # Get the primary agent LLM
        self.primary_llm = llm_registry.get_llm_from_agent_config('agent', agent_config)

        # Instantiate all the LLM instances for routing
        llms_for_routing_config = self.model_routing_config.llms_for_routing
        self.llms_for_routing = {
            config_name: self.llm_registry.get_llm(
                f'llm_for_routing.{config_name}', config=llm_config
            )
            for config_name, llm_config in llms_for_routing_config.items()
        }

        # All available LLMs for routing (set this BEFORE calling super().__init__)
        self.available_llms = {'primary': self.primary_llm, **self.llms_for_routing}

        # Create router config based on primary LLM
        router_config = copy.deepcopy(self.primary_llm.config)

        # Update model name to indicate this is a router
        llm_names = [self.primary_llm.config.model]
        if self.model_routing_config.llms_for_routing:
            llm_names.extend(
                config.model
                for config in self.model_routing_config.llms_for_routing.values()
            )
        router_config.model = f'router({",".join(llm_names)})'

        # Initialize parent LLM class
        super().__init__(
            config=router_config,
            service_id=service_id,
            metrics=metrics,
            retry_listener=retry_listener,
        )

        # Current LLM state
        self._current_llm = self.primary_llm  # Default to primary LLM
        self._last_routing_decision = 'primary'

        logger.info(
            f'RouterLLM initialized with {len(self.available_llms)} LLMs: {list(self.available_llms.keys())}'
        )

    @abstractmethod
    def _select_llm(self, messages: list[Message]) -> str:
        """
        Select which LLM to use based on messages and events.
        """
        pass

    def _get_llm_by_key(self, llm_key: str) -> LLM:
        """
        Get LLM instance by key.
        """
        if llm_key not in self.available_llms:
            raise ValueError(
                f'Unknown LLM key: {llm_key}. Available: {list(self.available_llms.keys())}'
            )
        return self.available_llms[llm_key]

    @property
    def completion(self) -> Callable:
        """
        Override completion to route to appropriate LLM.

        This method intercepts completion calls and routes them to the appropriate
        underlying LLM based on the routing logic implemented in _select_llm().
        """

        def router_completion(*args: Any, **kwargs: Any) -> Any:
            # Extract messages for routing decision
            messages = kwargs.get('messages', [])
            if args and not messages:
                messages = args[0] if args else []

            # Select appropriate LLM
            selected_llm_key = self._select_llm(messages)
            selected_llm = self._get_llm_by_key(selected_llm_key)

            # Update current state
            self._current_llm = selected_llm
            self._last_routing_decision = selected_llm_key

            logger.debug(
                f'RouterLLM routing to {selected_llm_key} ({selected_llm.config.model})'
            )

            # Delegate to selected LLM
            return selected_llm.completion(*args, **kwargs)

        return router_completion

    def __str__(self) -> str:
        """String representation of the router."""
        return f'{self.__class__.__name__}(llms={list(self.available_llms.keys())})'

    def __repr__(self) -> str:
        """Detailed string representation of the router."""
        return (
            f'{self.__class__.__name__}('
            f'primary={self.primary_llm.config.model}, '
            f'routing_llms={[llm.config.model for llm in self.llms_for_routing.values()]}, '
            f'current={self._last_routing_decision})'
        )

    def __getattr__(self, name):
        """Delegate other attributes/methods to the active LLM."""
        return getattr(self._current_llm, name)

    @classmethod
    def from_config(
        cls, llm_registry: 'LLMRegistry', agent_config: AgentConfig, **kwargs
    ) -> 'RouterLLM':
        """Factory method to create a RouterLLM instance from configuration."""
        router_cls = ROUTER_LLM_REGISTRY.get(agent_config.model_routing.router_name)
        if not router_cls:
            raise ValueError(
                f'Router LLM {agent_config.model_routing.router_name} not found.'
            )
        return router_cls(agent_config, llm_registry, **kwargs)
