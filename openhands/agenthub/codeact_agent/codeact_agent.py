import os
import sys
from collections import deque
from typing import TYPE_CHECKING
import random
import re
from dataclasses import dataclass

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
from openhands.agenthub.codeact_agent.tools.condensation_request import (
    CondensationRequestTool,
)
from openhands.agenthub.codeact_agent.tools.finish import FinishTool
from openhands.agenthub.codeact_agent.tools.ipython import IPythonTool
from openhands.agenthub.codeact_agent.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.codeact_agent.tools.str_replace_editor import (
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.task_tracker import (
    create_task_tracker_tool,
)
from openhands.agenthub.codeact_agent.tools.think import ThinkTool, ReflectionTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event, EventSource
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager
from openhands.events.action.agent import AgentThinkAction
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)



# Optional: in the long run, move this into AgentConfig
@dataclass
class AutoReflectionConfig:
    enabled: bool = True
    # Probabilistic trigger: after each observation event, fire with probability `prob`
    prob: float = 0.50
    # Reactive trigger: check last turn for "no tool" or "error observation"
    reactive_enabled: bool = True
    # How many past events to consider for the “last N steps” wording
    lookback_window: int = 3
    # The seeded thought text; {n} is formatted with lookback_window
    prompt: str = (
        "Look back at the last {n} steps. Are you making good progress, or should you "
        "step back and reconsider your approach? If you're off-track, propose "
        "concrete adjustments and the single next best action/tool to try."
    )


class CodeActAgent(Agent):
    VERSION = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    [Cerebras-Only] We want to trigger an autoreflection on 3 cases:
        (1) there is a tool parsing problem | This is in the Action space
        (2) the observation returned by tool call contains some errors | This is in the Observation space
        (3) general check: Probabilistically (say 10% chance) add it after N steps to “Look back at last N steps to see if you are making good progress or should take a step back and reconsider your approach”

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        # Override with router if needed
        self.llm = self.llm_registry.get_router(self.config)

        # NOTE: reflection
        self._num_steps = 0
        self._last_reflection_step = -1
        self.auto_reflect = AutoReflectionConfig()

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
                system_prompt_filename=self.config.resolved_system_prompt_filename,
            )

        return self._prompt_manager

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        # For these models, we use short tool descriptions ( < 1024 tokens)
        # to avoid hitting the OpenAI token limit for tool descriptions.
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-4', 'o3', 'o1', 'o4']

        use_short_tool_desc = False
        if self.llm is not None:
            # For historical reasons, previously OpenAI enforces max function description length of 1k characters
            # https://community.openai.com/t/function-call-description-max-length/529902
            # But it no longer seems to be an issue recently
            # https://community.openai.com/t/was-the-character-limit-for-schema-descriptions-upgraded/1225975
            # Tested on GPT-5 and longer description still works. But we still keep the logic to be safe for older models.
            use_short_tool_desc = any(
                model_substr in self.llm.config.model
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))
        if self.config.enable_think:
            tools.append(ThinkTool)
            tools.append(ReflectionTool)
        if self.config.enable_finish:
            tools.append(FinishTool)
        if self.config.enable_condensation_request:
            tools.append(CondensationRequestTool)
        if self.config.enable_browsing:
            if sys.platform == 'win32':
                logger.warning('Windows runtime does not support browsing yet')
            else:
                tools.append(BrowserTool)
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        if self.config.enable_plan_mode:
            # In plan mode, we use the task_tracker tool for task management
            tools.append(create_task_tracker_tool(use_short_tool_desc))
        if self.config.enable_llm_editor:
            tools.append(LLMBasedFileEditTool)
        elif self.config.enable_editor:
            tools.append(
                create_str_replace_editor_tool(
                    use_short_description=use_short_tool_desc
                )
            )
        return tools

    def reset(self) -> None:
        """Resets the CodeAct Agent's internal state."""
        super().reset()
        # Only clear pending actions, not LLM metrics
        self.pending_actions.clear()
        # NOTE: reflection
        self._num_steps = 0
        self._last_reflection_step = -1

    def step(self, state: State) -> 'Action':
        """Performs one step using the CodeAct Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        - CondensationAction(...) - condense conversation history by forgetting specified events and optionally providing a summary
        - FileReadAction(path, ...) - read file content from specified path
        - FileEditAction(path, ...) - edit file using LLM-based (deprecated) or ACI-based editing
        - AgentThinkAction(thought) - log agent's thought/reasoning process
        - CondensationRequestAction() - request condensation of conversation history
        - BrowseInteractiveAction(browser_actions) - interact with browser using specified actions
        - MCPAction(name, arguments) - interact with MCP server tools
        """
        # Condense the events from the state. If we get a view we'll pass those
        # to the conversation manager for processing, but if we get a condensation
        # event we'll just return that instead of an action. The controller will
        # immediately ask the agent to step again with the new view.
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        # NOTE: reflection case 1: Revisit last observation to catch if there is tool call failures
        if self.auto_reflect.enabled:
            if self.auto_reflect.reactive_enabled:
                is_last_turn_tool_error, tool_call_error_message =self._last_turn_has_tool_error(condensed_history)
                if is_last_turn_tool_error and not self._last_event_is_autoreflect(condensed_history):
                    self.pending_actions.clear() # clear the queue
                    self._last_reflection_step = self._num_steps
                    tool_call_reflection_prompt = f"I encountered a tool call with the following error: {tool_call_error_message}. \nI need to think about it and propose concrete adjustments plus the single next best action/tool."
                    return self._emit_reflection(tool_call_reflection_prompt)

            # NOTE: reflection case 2: Probablistically do general last N step reflection
            # Let's do not break any pending actions
            # Also make sure the last action was not think
            if random.random() < self.auto_reflect.prob and \
                not self.pending_actions and \
                self._last_reflection_step != self._num_steps and \
                not self._last_event_is_autoreflect(condensed_history):

                self._last_reflection_step = self._num_steps
                return self._emit_reflection(
                    self.auto_reflect.prompt.format(n=self.auto_reflect.lookback_window)
                )

        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)

        params: dict = {
            'messages': messages,
        }

        params['tools'] = check_tools(self.tools, self.llm.config)
        params['extra_body'] = {
            'metadata': state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }
        # if self._num_steps > 5:
        #     breakpoint()
        logger.debug(f'Last utterance input to LLM: {messages[-1]}')
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')
        # try:
        actions = self.response_to_actions(response)
        # NOTE: reflection case 3: cope with tool parse failures
        # except Exception as e:
            # # Wrong tool parsing will be raised by response_to_actions in function_calling.py
            # logger.warning(f"[AutoReflect] Tool-call parse error: {e}. Injecting reflection.")
            # self._last_reflection_step = self._num_steps
            # return self._emit_reflection(
            #     f"Malformed tool call (invalid JSON/args):\n```\n{e}\n```\n"
            #     "Reflect briefly and emit a valid tool call or a better-plan message."
            # )
        logger.debug(f'Actions after response_to_actions: {actions}')
        for action in actions:
            self.pending_actions.append(action)
        self._num_steps += 1
        return self.pending_actions.popleft()

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Finds the initial user message action from the full history."""
        initial_user_message: MessageAction | None = None
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                initial_user_message = event
                break

        if initial_user_message is None:
            # This should not happen in a valid conversation
            logger.error(
                f'CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history.'
            )
            # Depending on desired robustness, could raise error or create a dummy action
            # and log the error
            raise ValueError(
                'Initial user message not found in history. Please report this issue.'
            )
        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Checks for SystemMessageAction in events, adds one if missing (legacy support)
        2. Processes events (Actions and Observations) into messages, including SystemMessageAction
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            events: The list of events to convert to messages

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt (from SystemMessageAction)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Use ConversationMemory to process events (including SystemMessageAction)
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return codeact_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )


    def _last_turn_has_tool_error(self, events: list[Event])  -> tuple[bool, str | None]:
        '''
        We want to trigger a reflection on two cases related to tool calling:
        (1) there is a tool call parsing problem | This is in the Action space
        (2) the observation returned by tool call contains some errors | This is in the Observation space

        In this function we solve (2)
        '''
        # Edge case: no events
        if not events:
            return False, None

        # Make sure we only focus on Observations
        if not events[-1].__class__.__name__.lower().endswith("observation"):
            return False, None

        # For direct code execution or cmd execution observations
        if events[-1].__class__.__name__ in ["IPythonRunCellObservation", "CmdOutputObservation"]:
            if events[-1].error:
                return True, events[-1].__str__()

        # For other observations
        if events[-1].__class__.__name__ == "ErrorObservation":
            return True, events[-1].__str__()

        return False, None


    def _emit_reflection(self, text: str) -> MessageAction:
        msg = MessageAction(content=f"[AutoReflect]\n{text}\n Next step: Use `reflection` tool in next turn to do this reflection, no other tools are allowed!")
        msg._source = EventSource.AGENT   # <-- set source after init
        return msg

    def _last_event_is_autoreflect(self, events) -> bool:
        # Scan backwards, skip observations; stop on the last action/message.
        for ev in reversed(events):
            name = ev.__class__.__name__.lower()
            if name.endswith("observation"):
                continue
            if isinstance(ev, MessageAction) and getattr(ev, "_source", None) == EventSource.AGENT:
                text = (getattr(ev, "content", "") or "").strip().lower()
                return text.startswith("[autoreflect]")
            if name.endswith("action"):
                # last action but not our autoreflect
                return False
        return False
