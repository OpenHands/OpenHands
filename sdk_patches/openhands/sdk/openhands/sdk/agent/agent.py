import json

from pydantic import ValidationError, model_validator

import openhands.sdk.security.analyzer as analyzer
import openhands.sdk.security.risk as risk
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.agent.utils import (
    fix_malformed_tool_arguments,
    make_llm_completion,
    prepare_llm_messages,
)
from openhands.sdk.conversation import (
    ConversationCallbackType,
    ConversationState,
    ConversationTokenCallbackType,
    LocalConversation,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    LLMConvertibleEvent,
    MessageEvent,
    ObservationEvent,
    SystemPromptEvent,
    TokenEvent,
    UserRejectObservation,
)
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.llm import (
    LLMResponse,
    Message,
    MessageToolCall,
    ReasoningItemModel,
    RedactedThinkingBlock,
    TextContent,
    ThinkingBlock,
)
from openhands.sdk.llm.exceptions import (
    FunctionCallValidationError,
    LLMContextWindowExceedError,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.observability.laminar import (
    maybe_init_laminar,
    observe,
    should_enable_observability,
)
from openhands.sdk.observability.utils import extract_action_name
from openhands.sdk.security.llm_analyzer import LLMSecurityAnalyzer
from openhands.sdk.tool import (
    Action,
    Observation,
)
from openhands.sdk.tool.builtins import (
    FinishAction,
    FinishTool,
    ThinkAction,
)


logger = get_logger(__name__)
maybe_init_laminar()


class Agent(AgentBase):
    """Main agent implementation for OpenHands.

    The Agent class provides the core functionality for running AI agents that can
    interact with tools, process messages, and execute actions. It inherits from
    AgentBase and implements the agent execution logic.

    Example:
        >>> from openhands.sdk import LLM, Agent, Tool
        >>> llm = LLM(model="claude-sonnet-4-20250514", api_key=SecretStr("key"))
        >>> tools = [Tool(name="TerminalTool"), Tool(name="FileEditorTool")]
        >>> agent = Agent(llm=llm, tools=tools)
    """

    @model_validator(mode="before")
    @classmethod
    def _add_security_prompt_as_default(cls, data):
        """Ensure llm_security_analyzer=True is always set before initialization."""
        if not isinstance(data, dict):
            return data

        kwargs = data.get("system_prompt_kwargs") or {}
        if not isinstance(kwargs, dict):
            kwargs = {}

        kwargs.setdefault("llm_security_analyzer", True)
        data["system_prompt_kwargs"] = kwargs
        return data

    def init_state(
        self,
        state: ConversationState,
        on_event: ConversationCallbackType,
    ) -> None:
        super().init_state(state, on_event=on_event)
        # TODO(openhands): we should add test to test this init_state will actually
        # modify state in-place

        llm_convertible_messages = [
            event for event in state.events if isinstance(event, LLMConvertibleEvent)
        ]
        if len(llm_convertible_messages) == 0:
            # Prepare system message
            event = SystemPromptEvent(
                source="agent",
                system_prompt=TextContent(text=self.system_message),
                # Tools are stored as ToolDefinition objects and converted to
                # OpenAI format with security_risk parameter during LLM completion.
                # See make_llm_completion() in agent/utils.py for details.
                tools=list(self.tools_map.values()),
            )
            on_event(event)

    def _execute_actions(
        self,
        conversation: LocalConversation,
        action_events: list[ActionEvent],
        on_event: ConversationCallbackType,
    ):
        for action_event in action_events:
            self._execute_action_event(conversation, action_event, on_event=on_event)

    @observe(name="agent.step", ignore_inputs=["state", "on_event"])
    def step(
        self,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        on_token: ConversationTokenCallbackType | None = None,
    ) -> None:
        state = conversation.state
        # Check for pending actions (implicit confirmation)
        # and execute them before sampling new actions.
        pending_actions = ConversationState.get_unmatched_actions(state.events)
        if pending_actions:
            logger.info(
                "Confirmation mode: Executing %d pending action(s)",
                len(pending_actions),
            )
            self._execute_actions(conversation, pending_actions, on_event)
            return

        # Check if the last user message was blocked by a UserPromptSubmit hook
        # If so, skip processing and mark conversation as finished
        for event in reversed(list(state.events)):
            if isinstance(event, MessageEvent) and event.source == "user":
                reason = state.pop_blocked_message(event.id)
                if reason is not None:
                    logger.info(f"User message blocked by hook: {reason}")
                    state.execution_status = ConversationExecutionStatus.FINISHED
                    return
                break  # Only check the most recent user message

        # Prepare LLM messages using the utility function
        _messages_or_condensation = prepare_llm_messages(
            state.events, condenser=self.condenser, llm=self.llm
        )

        # Process condensation event before agent sampels another action
        if isinstance(_messages_or_condensation, Condensation):
            on_event(_messages_or_condensation)
            return

        _messages = _messages_or_condensation

        logger.debug(
            "Sending messages to LLM: "
            f"{json.dumps([m.model_dump() for m in _messages[1:]], indent=2)}"
        )

        try:
            llm_response = make_llm_completion(
                self.llm,
                _messages,
                tools=list(self.tools_map.values()),
                on_token=on_token,
            )
        except FunctionCallValidationError as e:
            logger.warning(f"LLM generated malformed function call: {e}")
            error_message = MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text=str(e))],
                ),
            )
            on_event(error_message)
            return
        except LLMContextWindowExceedError as e:
            # If condenser is available and handles requests, trigger condensation
            if (
                self.condenser is not None
                and self.condenser.handles_condensation_requests()
            ):
                logger.warning(
                    "LLM raised context window exceeded error, triggering condensation"
                )
                on_event(CondensationRequest())
                return
            # No condenser available or doesn't handle requests; log helpful warning
            self._log_context_window_exceeded_warning()
            raise e

        # LLMResponse already contains the converted message and metrics snapshot
        message: Message = llm_response.message

        # Check if this is a reasoning-only response (e.g., from reasoning models)
        # or a message-only response without tool calls
        has_reasoning = (
            message.responses_reasoning_item is not None
            or message.reasoning_content is not None
            or (message.thinking_blocks and len(message.thinking_blocks) > 0)
        )
        has_content = any(
            isinstance(c, TextContent) and c.text.strip() for c in message.content
        )

        if message.tool_calls and len(message.tool_calls) > 0:
            if not all(isinstance(c, TextContent) for c in message.content):
                logger.warning(
                    "LLM returned tool calls but message content is not all "
                    "TextContent - ignoring non-text content"
                )

            # Generate unique batch ID for this LLM response
            thought_content = [c for c in message.content if isinstance(c, TextContent)]

            action_events: list[ActionEvent] = []
            for i, tool_call in enumerate(message.tool_calls):
                action_event = self._get_action_event(
                    tool_call,
                    llm_response_id=llm_response.id,
                    on_event=on_event,
                    security_analyzer=state.security_analyzer,
                    thought=thought_content
                    if i == 0
                    else [],  # Only first gets thought
                    # Only first gets reasoning content
                    reasoning_content=message.reasoning_content if i == 0 else None,
                    # Only first gets thinking blocks
                    thinking_blocks=list(message.thinking_blocks) if i == 0 else [],
                    responses_reasoning_item=message.responses_reasoning_item
                    if i == 0
                    else None,
                )
                if action_event is None:
                    continue
                action_events.append(action_event)

            # Handle confirmation mode - exit early if actions need confirmation
            if self._requires_user_confirmation(state, action_events):
                return

            if action_events:
                self._execute_actions(conversation, action_events, on_event)

            # Emit VLLM token ids if enabled before returning
            self._maybe_emit_vllm_tokens(llm_response, on_event)
            return

        # No tool calls - emit message event for reasoning or content responses
        if not has_reasoning and not has_content:
            logger.warning("LLM produced empty response - continuing agent loop")

        msg_event = MessageEvent(
            source="agent",
            llm_message=message,
            llm_response_id=llm_response.id,
        )
        on_event(msg_event)

        # Emit VLLM token ids if enabled
        self._maybe_emit_vllm_tokens(llm_response, on_event)

        # Finish conversation if LLM produced content (awaits user input)
        # Continue if only reasoning without content (e.g., GPT-5 codex thinking)
        if has_content:
            logger.debug("LLM produced a message response - awaits user input")
            state.execution_status = ConversationExecutionStatus.FINISHED
            return

    def _requires_user_confirmation(
        self, state: ConversationState, action_events: list[ActionEvent]
    ) -> bool:
        """
        Decide whether user confirmation is needed to proceed.

        Rules:
            1. Confirmation mode is enabled
            2. Every action requires confirmation
            3. A single `FinishAction` never requires confirmation
            4. A single `ThinkAction` never requires confirmation
        """
        # A single `FinishAction` or `ThinkAction` never requires confirmation
        if len(action_events) == 1 and isinstance(
            action_events[0].action, (FinishAction, ThinkAction)
        ):
            return False

        # If there are no actions there is nothing to confirm
        if len(action_events) == 0:
            return False

        # If a security analyzer is registered, use it to grab the risks of the actions
        # involved. If not, we'll set the risks to UNKNOWN.
        if state.security_analyzer is not None:
            risks = [
                risk
                for _, risk in state.security_analyzer.analyze_pending_actions(
                    action_events
                )
            ]
        else:
            risks = [risk.SecurityRisk.UNKNOWN] * len(action_events)

        # Grab the confirmation policy from the state and pass in the risks.
        if any(state.confirmation_policy.should_confirm(risk) for risk in risks):
            state.execution_status = (
                ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
            )
            return True

        return False

    def _extract_security_risk(
        self,
        arguments: dict,
        tool_name: str,
        read_only_tool: bool,
        security_analyzer: analyzer.SecurityAnalyzerBase | None = None,
    ) -> risk.SecurityRisk:
        requires_sr = isinstance(security_analyzer, LLMSecurityAnalyzer)
        raw = arguments.pop("security_risk", None)

        # Default risk value for action event
        # Tool is marked as read-only so security risk can be ignored
        if read_only_tool:
            return risk.SecurityRisk.UNKNOWN

        # Raises exception if failed to pass risk field when expected
        # Exception will be sent back to agent as error event
        # Strong models like GPT-5 can correct itself by retrying
        if requires_sr and raw is None:
            raise ValueError(
                f"Failed to provide security_risk field in tool '{tool_name}'"
            )

        # When using weaker models without security analyzer
        # safely ignore missing security risk fields
        if not requires_sr and raw is None:
            return risk.SecurityRisk.UNKNOWN

        # Raises exception if invalid risk enum passed by LLM
        security_risk = risk.SecurityRisk(raw)
        return security_risk

    def _get_action_event(
        self,
        tool_call: MessageToolCall,
        llm_response_id: str,
        on_event: ConversationCallbackType,
        security_analyzer: analyzer.SecurityAnalyzerBase | None = None,
        thought: list[TextContent] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] | None = None,
        responses_reasoning_item: ReasoningItemModel | None = None,
    ) -> ActionEvent | None:
        """Converts a tool call into an ActionEvent, validating arguments.

        NOTE: state will be mutated in-place.
        """
        tool_name = tool_call.name
        tool = self.tools_map.get(tool_name, None)
        # Handle non-existing tools
        if tool is None:
            available = list(self.tools_map.keys())
            err = f"Tool '{tool_name}' not found. Available: {available}"
            logger.error(err)
            # Persist assistant function_call so next turn has matching call_id
            tc_event = ActionEvent(
                source="agent",
                thought=thought or [],
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks or [],
                responses_reasoning_item=responses_reasoning_item,
                tool_call=tool_call,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                llm_response_id=llm_response_id,
                action=None,
            )
            on_event(tc_event)
            event = AgentErrorEvent(
                error=err,
                tool_name=tool_name,
                tool_call_id=tool_call.id,
            )
            on_event(event)
            return

        # Validate arguments
        security_risk: risk.SecurityRisk = risk.SecurityRisk.UNKNOWN
        try:
            arguments = json.loads(tool_call.arguments)

            # Fix malformed arguments (e.g., JSON strings for list/dict fields)
            arguments = fix_malformed_tool_arguments(arguments, tool.action_type)
            security_risk = self._extract_security_risk(
                arguments,
                tool.name,
                tool.annotations.readOnlyHint if tool.annotations else False,
                security_analyzer,
            )
            assert "security_risk" not in arguments, (
                "Unexpected 'security_risk' key found in tool arguments"
            )

            action: Action = tool.action_from_arguments(arguments)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            err = (
                f"Error validating args {tool_call.arguments} for tool "
                f"'{tool.name}': {e}"
            )
            # Persist assistant function_call so next turn has matching call_id
            tc_event = ActionEvent(
                source="agent",
                thought=thought or [],
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks or [],
                responses_reasoning_item=responses_reasoning_item,
                tool_call=tool_call,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                llm_response_id=llm_response_id,
                action=None,
            )
            on_event(tc_event)
            event = AgentErrorEvent(
                error=err,
                tool_name=tool_name,
                tool_call_id=tool_call.id,
            )
            on_event(event)
            return

        action_event = ActionEvent(
            action=action,
            thought=thought or [],
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks or [],
            responses_reasoning_item=responses_reasoning_item,
            tool_name=tool.name,
            tool_call_id=tool_call.id,
            tool_call=tool_call,
            llm_response_id=llm_response_id,
            security_risk=security_risk,
        )
        on_event(action_event)
        return action_event

    @observe(ignore_inputs=["state", "on_event"])
    def _execute_action_event(
        self,
        conversation: LocalConversation,
        action_event: ActionEvent,
        on_event: ConversationCallbackType,
    ):
        """Execute an action event and update the conversation state.

        It will call the tool's executor and update the state & call callback fn
        with the observation.

        If the action was blocked by a PreToolUse hook (recorded in
        state.blocked_actions), a UserRejectObservation is emitted instead
        of executing the action.
        """
        state = conversation.state

        # Check if this action was blocked by a PreToolUse hook
        reason = state.pop_blocked_action(action_event.id)
        if reason is not None:
            logger.info(f"Action '{action_event.tool_name}' blocked by hook: {reason}")
            rejection = UserRejectObservation(
                action_id=action_event.id,
                tool_name=action_event.tool_name,
                tool_call_id=action_event.tool_call_id,
                rejection_reason=reason,
            )
            on_event(rejection)
            return rejection

        tool = self.tools_map.get(action_event.tool_name, None)
        if tool is None:
            raise RuntimeError(
                f"Tool '{action_event.tool_name}' not found. This should not happen "
                "as it was checked earlier."
            )

        # Execute actions!
        try:
            if should_enable_observability():
                tool_name = extract_action_name(action_event)
                observation: Observation = observe(name=tool_name, span_type="TOOL")(
                    tool
                )(action_event.action, conversation)
            else:
                observation = tool(action_event.action, conversation)
            assert isinstance(observation, Observation), (
                f"Tool '{tool.name}' executor must return an Observation"
            )
        except ValueError as e:
            # Tool execution raised a ValueError (e.g., invalid argument combination)
            # Convert to AgentErrorEvent so the agent can correct itself
            err = f"Error executing tool '{tool.name}': {e}"
            logger.warning(err)
            error_event = AgentErrorEvent(
                error=err,
                tool_name=tool.name,
                tool_call_id=action_event.tool_call.id,
            )
            on_event(error_event)
            return error_event

        obs_event = ObservationEvent(
            observation=observation,
            action_id=action_event.id,
            tool_name=tool.name,
            tool_call_id=action_event.tool_call.id,
        )
        on_event(obs_event)

        # Set conversation state
        if tool.name == FinishTool.name:
            state.execution_status = ConversationExecutionStatus.FINISHED
        return obs_event

    def _maybe_emit_vllm_tokens(
        self, llm_response: LLMResponse, on_event: ConversationCallbackType
    ) -> None:
        if (
            "return_token_ids" in self.llm.litellm_extra_body
        ) and self.llm.litellm_extra_body["return_token_ids"]:
            token_event = TokenEvent(
                source="agent",
                prompt_token_ids=llm_response.raw_response["prompt_token_ids"],
                response_token_ids=llm_response.raw_response["choices"][0][
                    "provider_specific_fields"
                ]["token_ids"],
            )
            on_event(token_event)

    def _log_context_window_exceeded_warning(self) -> None:
        """Log a helpful warning when context window is exceeded without a condenser."""
        if self.condenser is None:
            logger.warning(
                "\n"
                "=" * 80 + "\n"
                "⚠️  CONTEXT WINDOW EXCEEDED ERROR\n"
                "=" * 80 + "\n"
                "\n"
                "The LLM's context window has been exceeded, but no condenser is "
                "configured.\n"
                "\n"
                "Current configuration:\n"
                f"  • Condenser: None\n"
                f"  • LLM Model: {self.llm.model}\n"
                "\n"
                "To prevent this error, configure a condenser to automatically "
                "summarize\n"
                "conversation history when it gets too long.\n"
                "\n"
                "Example configuration:\n"
                "\n"
                "  from openhands.sdk import Agent, LLM\n"
                "  from openhands.sdk.context.condenser import "
                "LLMSummarizingCondenser\n"
                "\n"
                "  agent = Agent(\n"
                "      llm=LLM(model='your-model'),\n"
                "      condenser=LLMSummarizingCondenser(\n"
                "          llm=LLM(model='your-model'),  # Can use same or "
                "cheaper model\n"
                "          max_size=120,  # Maximum events before condensation\n"
                "          keep_first=4   # Number of initial events to preserve\n"
                "      )\n"
                "  )\n"
                "\n"
                "For more information, see: "
                "https://docs.openhands.dev/sdk/guides/context-condenser\n"
                "=" * 80
            )
        else:
            condenser_type = type(self.condenser).__name__
            handles_requests = self.condenser.handles_condensation_requests()
            condenser_config = self.condenser.model_dump(
                exclude={"llm"}, exclude_none=True
            )
            condenser_llm_obj = getattr(self.condenser, "llm", None)
            condenser_llm = (
                condenser_llm_obj.model if condenser_llm_obj is not None else "N/A"
            )

            logger.warning(
                "\n"
                "=" * 80 + "\n"
                "⚠️  CONTEXT WINDOW EXCEEDED ERROR\n"
                "=" * 80 + "\n"
                "\n"
                "The LLM's context window has been exceeded.\n"
                "\n"
                "Current configuration:\n"
                f"  • Condenser Type: {condenser_type}\n"
                f"  • Handles Condensation Requests: {handles_requests}\n"
                f"  • Condenser LLM: {condenser_llm}\n"
                f"  • Agent LLM Model: {self.llm.model}\n"
                f"  • Condenser Config: {json.dumps(condenser_config, indent=4)}\n"
                "\n"
                "Your condenser is configured but does not handle condensation "
                "requests\n"
                "(handles_condensation_requests() returned False).\n"
                "\n"
                "To fix this:\n"
                "  1. Use LLMSummarizingCondenser which handles condensation "
                "requests, OR\n"
                "  2. Implement handles_condensation_requests() in your custom "
                "condenser\n"
                "\n"
                "Example with LLMSummarizingCondenser:\n"
                "\n"
                "  from openhands.sdk.context.condenser import "
                "LLMSummarizingCondenser\n"
                "\n"
                "  agent = Agent(\n"
                "      llm=LLM(model='your-model'),\n"
                "      condenser=LLMSummarizingCondenser(\n"
                "          llm=LLM(model='your-model'),\n"
                "          max_size=120,\n"
                "          keep_first=4\n"
                "      )\n"
                "  )\n"
                "\n"
                "For more information, see: "
                "https://docs.openhands.dev/sdk/guides/context-condenser\n"
                "=" * 80
            )
