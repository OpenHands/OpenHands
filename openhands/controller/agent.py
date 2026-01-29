# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# V1 replacement for this module lives in the Software Agent SDK.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.events.action import Action
    from openhands.events.action.message import SystemMessageAction
    from openhands.utils.prompt import PromptManager
from litellm import ChatCompletionToolParam

from openhands.core.config import AgentConfig
from openhands.core.exceptions import (
    AgentAlreadyRegisteredError,
    AgentNotRegisteredError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.event import EventSource
from openhands.runtime.plugins import PluginRequirement


class Agent(ABC):
    DEPRECATED = False
    """
    This abstract base class is an general interface for an agent dedicated to
    executing a specific instruction and allowing human interaction with the
    agent during execution.
    It tracks the execution status and maintains a history of interactions.
    """

    _registry: dict[str, type['Agent']] = {}
    sandbox_plugins: list[PluginRequirement] = []

    config_model: type[AgentConfig] = AgentConfig
    """Class field that specifies the config model to use for the agent. Subclasses may override with a derived config model if needed."""

    def __init__(
        self,
        config: AgentConfig,
        llm_registry: LLMRegistry,
    ):
        self.llm = llm_registry.get_llm_from_agent_config('agent', config)
        self.llm_registry = llm_registry
        self.config = config
        self._complete = False
        self._prompt_manager: 'PromptManager' | None = None
        self.mcp_tools: dict[str, ChatCompletionToolParam] = {}
        self.tools: list = []

    @property
    def prompt_manager(self) -> 'PromptManager':
        if self._prompt_manager is None:
            raise ValueError(f'Prompt manager not initialized for agent {self.name}')
        return self._prompt_manager

    def get_system_message(self) -> 'SystemMessageAction | None':
        """Returns a SystemMessageAction containing the system message and tools.
        This will be added to the event stream as the first message.

        Returns:
            SystemMessageAction: The system message action with content and tools
            None: If there was an error generating the system message
        """
        # Import here to avoid circular imports
        from openhands.events.action.message import SystemMessageAction

        try:
            if not self.prompt_manager:
                logger.warning(
                    f'[{self.name}] Prompt manager not initialized before getting system message'
                )
                return None

            system_message = self.prompt_manager.get_system_message(
                cli_mode=self.config.cli_mode
            )

            # Get tools if available
            tools = getattr(self, 'tools', None)

            system_message_action = SystemMessageAction(
                content=system_message, tools=tools, agent_class=self.name
            )
            # Set the source attribute
            system_message_action._source = EventSource.AGENT  # type: ignore

            return system_message_action
        except Exception as e:
            logger.warning(f'[{self.name}] Failed to generate system message: {e}')
            return None

    @property
    def complete(self) -> bool:
        """Indicates whether the current instruction execution is complete.

        Returns:
        - complete (bool): True if execution is complete; False otherwise.
        """
        return self._complete

    @abstractmethod
    def step(self, state: 'State') -> 'Action':
        """Starts the execution of the assigned instruction. This method should
        be implemented by subclasses to define the specific execution logic.
        """
        pass

    def reset(self) -> None:
        """Resets the agent's execution status."""
        # Only reset the completion status, not the LLM metrics
        self._complete = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def register(cls, name: str, agent_cls: type['Agent']) -> None:
        """Registers an agent class in the registry.

        Parameters:
        - name (str): The name to register the class under.
        - agent_cls (Type['Agent']): The class to register.

        Raises:
        - AgentAlreadyRegisteredError: If name already registered
        """
        if name in cls._registry:
            raise AgentAlreadyRegisteredError(name)
        cls._registry[name] = agent_cls

    @classmethod
    def get_cls(cls, name: str) -> type['Agent']:
        """Retrieves an agent class from the registry.

        Parameters:
        - name (str): The name of the class to retrieve

        Returns:
        - agent_cls (Type['Agent']): The class registered under the specified name.

        Raises:
        - AgentNotRegisteredError: If name not registered
        """
        if name not in cls._registry:
            raise AgentNotRegisteredError(name)
        return cls._registry[name]

    @classmethod
    def list_agents(cls) -> list[str]:
        """Retrieves the list of all agent names from the registry.

        Raises:
        - AgentNotRegisteredError: If no agent is registered
        """
        if not bool(cls._registry):
            raise AgentNotRegisteredError()
        return list(cls._registry.keys())

    def set_mcp_tools(self, mcp_tools: list[dict]) -> None:
        """Sets the list of MCP tools for the agent.

        Args:
        - mcp_tools (list[dict]): The list of MCP tools.
        """
        logger.info(
            f'Setting {len(mcp_tools)} MCP tools for agent {self.name}: {[tool["function"]["name"] for tool in mcp_tools]}'
        )
        for tool in mcp_tools:
            _tool = ChatCompletionToolParam(**tool)
            if _tool['function']['name'] in self.mcp_tools:
                logger.warning(
                    f'Tool {_tool["function"]["name"]} already exists, skipping'
                )
                continue
            self.mcp_tools[_tool['function']['name']] = _tool
            self.tools.append(_tool)
        logger.info(
            f'Tools updated for agent {self.name}, total {len(self.tools)}: {[tool["function"]["name"] for tool in self.tools]}'
        )
