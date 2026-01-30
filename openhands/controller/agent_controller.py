# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# V1 replacement for this module lives in the Software Agent SDK.
from __future__ import annotations

import asyncio
import copy
import os
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from openhands.security.analyzer import SecurityAnalyzer

from litellm.exceptions import (  # noqa
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from openhands.controller.agent import Agent
from openhands.controller.replay import ReplayManager
from openhands.controller.state.state import State
from openhands.controller.state.state_tracker import StateTracker
from openhands.controller.stuck import StuckDetector
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.exceptions import (
    AgentStuckInLoopError,
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMContextWindowExceedError,
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
)
from openhands.core.logger import LOG_ALL_EVENTS
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events import (
    EventSource,
    EventStream,
    EventStreamSubscriber,
    RecallType,
)
from openhands.events.action import (
    Action,
    ActionConfirmationStatus,
    ActionSecurityRisk,
    AgentDelegateAction,
    AgentFinishAction,
    AgentRejectAction,
    BrowseInteractiveAction,
    ChangeAgentStateAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
    NullAction,
    SystemMessageAction,
    LoopRecoveryAction,
)
from openhands.events.action.agent import (
    CondensationAction,
    CondensationRequestAction,
    RecallAction,
)
from openhands.events.event import Event
from openhands.events.observation import (
    AgentDelegateObservation,
    AgentStateChangedObservation,
    ErrorObservation,
    NullObservation,
    Observation,
    LoopDetectionObservation,
)
from openhands.events.serialization.event import truncate_content
from openhands.llm.metrics import Metrics
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.files import FileStore

# note: RESUME is only available on web GUI
TRAFFIC_CONTROL_REMINDER = (
    "Please click on resume button if you'd like to continue, or start a new task."
)
ERROR_ACTION_NOT_EXECUTED_STOPPED_ID = 'AGENT_ERROR$ERROR_ACTION_NOT_EXECUTED_STOPPED'
ERROR_ACTION_NOT_EXECUTED_ERROR_ID = 'AGENT_ERROR$ERROR_ACTION_NOT_EXECUTED_ERROR'
ERROR_ACTION_NOT_EXECUTED_STOPPED = (
    'Stop button pressed. The action has not been executed.'
)
ERROR_ACTION_NOT_EXECUTED_ERROR = 'The action has not been executed due to a runtime error. The runtime system may have crashed and restarted due to resource constraints. Any previously established system state, dependencies, or environment variables may have been lost.'


class AgentController:
    id: str
    agent: Agent
    max_iterations: int
    event_stream: EventStream
    state: State
    confirmation_mode: bool
    agent_to_llm_config: dict[str, LLMConfig]
    agent_configs: dict[str, AgentConfig]
    parent: 'AgentController | None' = None
    delegate: 'AgentController | None' = None
    _pending_action_info: tuple[Action, float] | None = None  # (action, timestamp)
    _closed: bool = False
    _cached_first_user_message: MessageAction | None = None

    def __init__(
        self,
        agent: Agent,
        event_stream: EventStream,
        conversation_stats: ConversationStats,
        iteration_delta: int,
        budget_per_task_delta: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
        sid: str | None = None,
        file_store: FileStore | None = None,
        user_id: str | None = None,
        confirmation_mode: bool = False,
        initial_state: State | None = None,
        is_delegate: bool = False,
        headless_mode: bool = True,
        status_callback: Callable | None = None,
        replay_events: list[Event] | None = None,
        security_analyzer: 'SecurityAnalyzer | None' = None,
    ):
        """Initializes a new instance of the AgentController class.

        Args:
            agent: The agent instance to control.
            event_stream: The event stream to publish events to.
            max_iterations: The maximum number of iterations the agent can run.
            max_budget_per_task: The maximum budget (in USD) allowed per task, beyond which the agent will stop.
            agent_to_llm_config: A dictionary mapping agent names to LLM configurations in the case that
                we delegate to a different agent.
            agent_configs: A dictionary mapping agent names to agent configurations in the case that
                we delegate to a different agent.
            sid: The session ID of the agent.
            confirmation_mode: Whether to enable confirmation mode for agent actions.
            initial_state: The initial state of the controller.
            is_delegate: Whether this controller is a delegate.
            headless_mode: Whether the agent is run in headless mode.
            status_callback: Optional callback function to handle status updates.
            replay_events: A list of logs to replay.
        """
        self.id = sid or event_stream.sid
        self.user_id = user_id
        self.file_store = file_store
        self.agent = agent
        self.headless_mode = headless_mode
        self.is_delegate = is_delegate
        self.conversation_stats = conversation_stats

        # the event stream must be set before maybe subscribing to it
        self.event_stream = event_stream

        # subscribe to the event stream if this is not a delegate
        if not self.is_delegate:
            self.event_stream.subscribe(
                EventStreamSubscriber.AGENT_CONTROLLER, self.on_event, self.id
            )

        self.state_tracker = StateTracker(sid, file_store, user_id)

        # state from the previous session, state from a parent agent, or a fresh state
        self.set_initial_state(
            state=initial_state,
            conversation_stats=conversation_stats,
            max_iterations=iteration_delta,
            max_budget_per_task=budget_per_task_delta,
            confirmation_mode=confirmation_mode,
        )

        self.state = self.state_tracker.state  # TODO: share between manager and controller for backward compatability; we should ideally move all state related logic to the state manager

        self.agent_to_llm_config = agent_to_llm_config if agent_to_llm_config else {}
        self.agent_configs = agent_configs if agent_configs else {}
        self._initial_max_iterations = iteration_delta
        self._initial_max_budget_per_task = budget_per_task_delta

        # stuck helper
        self._stuck_detector = StuckDetector(self.state)
        self.status_callback = status_callback

        # replay-related
        self._replay_manager = ReplayManager(replay_events)

        self.confirmation_mode = confirmation_mode

        # security analyzer for direct access
        self.security_analyzer = security_analyzer

        # Add the system message to the event stream
        self._add_system_message()

    async def _handle_security_analyzer(self, action: Action) -> None:
        """Handle security risk analysis for an action.

        If a security analyzer is configured, use it to analyze the action.
        If no security analyzer is configured, set the risk to HIGH (fail-safe approach).

        Args:
            action: The action to analyze for security risks.
        """
        if self.security_analyzer:
            try:
                if (
                    hasattr(action, 'security_risk')
                    and action.security_risk is not None
                ):
                    logger.debug(
                        f'Original security risk for {action}: {action.security_risk})'
                    )
                if hasattr(action, 'security_risk'):
                    action.security_risk = await self.security_analyzer.security_risk(
                        action
                    )
                    logger.debug(
                        f'[Security Analyzer: {self.security_analyzer.__class__}] Override security risk for action {action}: {action.security_risk}'
                    )
            except Exception as e:
                logger.warning(
                    f'Failed to analyze security risk for action {action}: {e}'
                )
                if hasattr(action, 'security_risk'):
                    action.security_risk = ActionSecurityRisk.UNKNOWN
        else:
            # When no security analyzer is configured, treat all actions as UNKNOWN risk
            # This is a fail-safe approach that ensures confirmation is required
            logger.debug(
                f'No security analyzer configured, setting UNKNOWN risk for action: {action}'
            )
            if hasattr(action, 'security_risk'):
                action.security_risk = ActionSecurityRisk.UNKNOWN

    def _add_system_message(self):
        for event in self.event_stream.search_events(start_id=self.state.start_id):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                # FIXME: Remove this after 6/1/2025
                # Do not try to add a system message if we first run into
                # a user message -- this means the eventstream exits before
                # SystemMessageAction is introduced.
                # We expect *agent* to handle this case gracefully.
                return

            if isinstance(event, SystemMessageAction):
                # Do not try to add the system message if it already exists
                return

        # Add the system message to the event stream
        # This should be done for all agents, including delegates
        system_message = self.agent.get_system_message()
        if system_message and system_message.content:
            preview = (
                system_message.content[:50] + '...'
                if len(system_message.content) > 50
                else system_message.content
            )
            logger.debug(f'System message: {preview}')
            self.event_stream.add_event(system_message, EventSource.AGENT)

    async def close(self, set_stop_state: bool = True) -> None:
        """Closes the agent controller, canceling any ongoing tasks and unsubscribing from the event stream.

        Note that it's fairly important that this closes properly, otherwise the state is incomplete.
        """
        if set_stop_state:
            await self.set_agent_state_to(AgentState.STOPPED)

        self.state_tracker.close(self.event_stream)

        # unsubscribe from the event stream
        # only the root parent controller subscribes to the event stream
        if not self.is_delegate:
            self.event_stream.unsubscribe(
                EventStreamSubscriber.AGENT_CONTROLLER, self.id
            )
        self._closed = True

    def log(
        self,
        level: str,
        message: str,
        extra: dict | None = None,
        exc_info: bool = False,
    ) -> None:
        """Logs a message to the agent controller's logger.

        Args:
            level (str): The logging level to use (e.g., 'info', 'debug', 'error').
            message (str): The message to log.
            extra (dict | None, optional): Additional fields to log. Includes session_id by default.
            exc_info (bool, optional): Whether to include exception info. Defaults to False.
        """
        message = f'[Agent Controller {self.id}] {message}'
        if extra is None:
            extra = {}
        extra_merged = {'session_id': self.id, **extra}
        getattr(logger, level)(
            message, extra=extra_merged, exc_info=exc_info, stacklevel=2
        )

    async def _react_to_exception(
        self,
        e: Exception,
    ) -> None:
        """React to an exception by setting the agent state to error and sending a status message."""
        # Store the error reason before setting the agent state
        self.state.last_error = f'{type(e).__name__}: {str(e)}'

        if self.status_callback is not None:
            runtime_status = RuntimeStatus.ERROR
            if isinstance(e, AuthenticationError):
                runtime_status = RuntimeStatus.ERROR_LLM_AUTHENTICATION
                self.state.last_error = runtime_status.value
            elif isinstance(
                e,
                (
                    ServiceUnavailableError,
                    APIConnectionError,
                    APIError,
                ),
            ):
                runtime_status = RuntimeStatus.ERROR_LLM_SERVICE_UNAVAILABLE
                self.state.last_error = runtime_status.value
            elif isinstance(e, InternalServerError):
                runtime_status = RuntimeStatus.ERROR_LLM_INTERNAL_SERVER_ERROR
                self.state.last_error = runtime_status.value
            elif isinstance(e, BadRequestError) and 'ExceededBudget' in str(e):
                runtime_status = RuntimeStatus.ERROR_LLM_OUT_OF_CREDITS
                self.state.last_error = runtime_status.value
            elif isinstance(e, ContentPolicyViolationError) or (
                isinstance(e, BadRequestError)
                and 'ContentPolicyViolationError' in str(e)
            ):
                runtime_status = RuntimeStatus.ERROR_LLM_CONTENT_POLICY_VIOLATION
                self.state.last_error = runtime_status.value
            elif isinstance(e, RateLimitError):
                # Check if this is the final retry attempt
                if (
                    hasattr(e, 'retry_attempt')
                    and hasattr(e, 'max_retries')
                    and e.retry_attempt >= e.max_retries
                ):
                    # All retries exhausted, set to ERROR state with a special message
                    self.state.last_error = (
                        RuntimeStatus.AGENT_RATE_LIMITED_STOPPED_MESSAGE.value
                    )
                    await self.set_agent_state_to(AgentState.ERROR)
                else:
                    # Still retrying, set to RATE_LIMITED state
                    await self.set_agent_state_to(AgentState.RATE_LIMITED)
                return
            self.status_callback('error', runtime_status, self.state.last_error)

        # Set the agent state to ERROR after storing the reason
        await self.set_agent_state_to(AgentState.ERROR)

    def step(self) -> None:
        asyncio.create_task(self._step_with_exception_handling())

    async def _step_with_exception_handling(self) -> None:
        try:
            await self._step()
        except Exception as e:
            self.log(
                'error',
                f'Error while running the agent (session ID: {self.id}): {e}',
                exc_info=True,
            )
            reported = RuntimeError(
                f'There was an unexpected error while running the agent: {e.__class__.__name__}. You can refresh the page or ask the agent to try again.'
            )
            if (
                isinstance(e, Timeout)
                or isinstance(e, APIError)
                or isinstance(e, BadRequestError)
                or isinstance(e, NotFoundError)
                or isinstance(e, InternalServerError)
                or isinstance(e, AuthenticationError)
                or isinstance(e, RateLimitError)
                or isinstance(e, ContentPolicyViolationError)
                or isinstance(e, LLMContextWindowExceedError)
            ):
                reported = e
            else:
                self.log(
                    'warning',
                    f'Unknown exception type while running the agent: {type(e).__name__}.',
                )
            await self._react_to_exception(reported)

    def should_step(self, event: Event) -> bool:
        """Whether the agent should take a step based on an event.

        In general, the agent should take a step if it receives a message from the user,
        or observes something in the environment (after acting).
        """
        # it might be the delegate's day in the sun
        if self.delegate is not None:
            return False

        if isinstance(event, Action):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return True
            if (
                isinstance(event, MessageAction)
                and self.get_agent_state() != AgentState.AWAITING_USER_INPUT
            ):
                # TODO: this is fragile, but how else to check if eligible?
                return True
            if isinstance(event, AgentDelegateAction):
                return True
            if isinstance(event, CondensationAction):
                return True
            if isinstance(event, CondensationRequestAction):
                return True
            return False
        if isinstance(event, Observation):
            if (
                isinstance(event, NullObservation)
                and event.cause is not None
                and event.cause
                > 0  # NullObservation has cause > 0 (RecallAction), not 0 (user message)
            ):
                return True
            if isinstance(event, AgentStateChangedObservation) or isinstance(
                event, NullObservation
            ):
                return False
            return True
        return False

    def on_event(self, event: Event) -> None:
        """Callback from the event stream. Notifies the controller of incoming events.

        Args:
            event (Event): The incoming event to process.
        """
        # If we have a delegate that is not finished or errored, forward events to it
        if self.delegate is not None:
            delegate_state = self.delegate.get_agent_state()
            if (
                delegate_state
                not in (
                    AgentState.FINISHED,
                    AgentState.ERROR,
                    AgentState.REJECTED,
                )
                or 'RuntimeError: Agent reached maximum iteration.'
                in self.delegate.state.last_error
                or 'RuntimeError:Agent reached maximum budget for conversation'
                in self.delegate.state.last_error
            ):
                # Forward the event to delegate and skip parent processing
                asyncio.get_event_loop().run_until_complete(
                    self.delegate._on_event(event)
                )
                return
            else:
                # delegate is done or errored, so end it
                self.end_delegate()
                return

        # continue parent processing only if there's no active delegate
        asyncio.get_event_loop().run_until_complete(self._on_event(event))

    async def _on_event(self, event: Event) -> None:
        if hasattr(event, 'hidden') and event.hidden:
            return

        self.state_tracker.add_history(event)

        if isinstance(event, Action):
            await self._handle_action(event)
        elif isinstance(event, Observation):
            await self._handle_observation(event)

        should_step = self.should_step(event)
        if should_step:
            self.log(
                'debug',
                f'Stepping agent after event: {type(event).__name__}',
                extra={'msg_type': 'STEPPING_AGENT'},
            )
            await self._step_with_exception_handling()
        elif isinstance(event, MessageAction) and event.source == EventSource.USER:
            # If we received a user message but aren't stepping, log why
            self.log(
                'warning',
                f'Not stepping agent after user message. Current state: {self.get_agent_state()}',
                extra={'msg_type': 'NOT_STEPPING_AFTER_USER_MESSAGE'},
            )

    async def _handle_action(self, action: Action) -> None:
        """Handles an Action from the agent or delegate."""
        if isinstance(action, ChangeAgentStateAction):
            await self.set_agent_state_to(action.agent_state)  # type: ignore
        elif isinstance(action, MessageAction):
            await self._handle_message_action(action)
        elif isinstance(action, AgentDelegateAction):
            await self.start_delegate(action)
            assert self.delegate is not None
            # Post a MessageAction with the task for the delegate
            if 'task' in action.inputs:
                self.event_stream.add_event(
                    MessageAction(content='TASK: ' + action.inputs['task']),
                    EventSource.USER,
                )
                await self.delegate.set_agent_state_to(AgentState.RUNNING)
            return

        elif isinstance(action, AgentFinishAction):
            self.state.outputs = action.outputs
            await self.set_agent_state_to(AgentState.FINISHED)
        elif isinstance(action, AgentRejectAction):
            self.state.outputs = action.outputs
            await self.set_agent_state_to(AgentState.REJECTED)
        elif isinstance(action, LoopRecoveryAction):
            await self._handle_loop_recovery_action(action)

    async def _handle_observation(self, observation: Observation) -> None:
        """Handles observation from the event stream.

        Args:
            observation (observation): The observation to handle.
        """
        observation_to_print = copy.deepcopy(observation)
        if len(observation_to_print.content) > self.agent.llm.config.max_message_chars:
            observation_to_print.content = truncate_content(
                observation_to_print.content, self.agent.llm.config.max_message_chars
            )
        # Use info level if LOG_ALL_EVENTS is set
        log_level = 'info' if os.getenv('LOG_ALL_EVENTS') in ('true', '1') else 'debug'
        self.log(
            log_level, str(observation_to_print), extra={'msg_type': 'OBSERVATION'}
        )

        # this happens for runnable actions and microagent actions
        if self._pending_action and self._pending_action.id == observation.cause:
            if self.state.agent_state == AgentState.AWAITING_USER_CONFIRMATION:
                return

            self._pending_action = None

            if self.state.agent_state == AgentState.USER_CONFIRMED:
                await self.set_agent_state_to(AgentState.RUNNING)
            if self.state.agent_state == AgentState.USER_REJECTED:
                await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
            return

    async def _handle_message_action(self, action: MessageAction) -> None:
        """Handles message actions from the event stream.

        Args:
            action (MessageAction): The message action to handle.
        """
        if action.source == EventSource.USER:
            # Use info level if LOG_ALL_EVENTS is set
            log_level = (
                'info' if os.getenv('LOG_ALL_EVENTS') in ('true', '1') else 'debug'
            )
            self.log(
                log_level,
                str(action),
                extra={'msg_type': 'ACTION', 'event_source': EventSource.USER},
            )

            # if this is the first user message for this agent, matters for the microagent info type
            first_user_message = self._first_user_message()
            is_first_user_message = (
                action.id == first_user_message.id if first_user_message else False
            )
            recall_type = (
                RecallType.WORKSPACE_CONTEXT
                if is_first_user_message
                else RecallType.KNOWLEDGE
            )

            recall_action = RecallAction(query=action.content, recall_type=recall_type)
            self._pending_action = recall_action
            # this is source=USER because the user message is the trigger for the microagent retrieval
            self.event_stream.add_event(recall_action, EventSource.USER)

            if self.get_agent_state() != AgentState.RUNNING:
                await self.set_agent_state_to(AgentState.RUNNING)

        elif action.source == EventSource.AGENT:
            # If the agent is waiting for a response, set the appropriate state
            if action.wait_for_response:
                await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)

    async def _handle_loop_recovery_action(self, action: LoopRecoveryAction) -> None:
        # Check if this is a loop recovery option
        if self._stuck_detector.stuck_analysis:
            option = action.option

            # Handle the loop recovery option
            if option == 1:
                # Option 1: Restart from before loop
                await self._perform_loop_recovery(self._stuck_detector.stuck_analysis)
            elif option == 2:
                # Option 2: Restart with last user message
                await self._restart_with_last_user_message(
                    self._stuck_detector.stuck_analysis
                )
            elif option == 3:
                # Option 3: Stop agent completely
                await self.set_agent_state_to(AgentState.STOPPED)
            return

    def _reset(self) -> None:
        """Resets the agent controller."""
        # Runnable actions need an Observation
        # make sure there is an Observation with the tool call metadata to be recognized by the agent
        # otherwise the pending action is found in history, but it's incomplete without an obs with tool result
        if self._pending_action and hasattr(self._pending_action, 'tool_call_metadata'):
            # find out if there already is an observation with the same tool call metadata
            found_observation = False
            for event in self.state.history:
                if (
                    isinstance(event, Observation)
                    and event.tool_call_metadata
                    == self._pending_action.tool_call_metadata
                ):
                    found_observation = True
                    break

            # make a new ErrorObservation with the tool call metadata
            if not found_observation:
                # Use different messages and IDs based on whether the agent was stopped by user or due to error
                if self.state.agent_state == AgentState.STOPPED:
                    error_content = ERROR_ACTION_NOT_EXECUTED_STOPPED
                    error_id = ERROR_ACTION_NOT_EXECUTED_STOPPED_ID
                else:  # AgentState.ERROR
                    error_content = ERROR_ACTION_NOT_EXECUTED_ERROR
                    error_id = ERROR_ACTION_NOT_EXECUTED_ERROR_ID

                obs = ErrorObservation(
                    content=error_content,
                    error_id=error_id,
                )
                obs.tool_call_metadata = self._pending_action.tool_call_metadata
                obs._cause = self._pending_action.id  # type: ignore[attr-defined]
                self.event_stream.add_event(obs, EventSource.AGENT)

        # NOTE: RecallActions don't need an ErrorObservation upon reset, as long as they have no tool calls

        # reset the pending action, this will be called when the agent is STOPPED or ERROR
        self._pending_action = None
        self.agent.reset()

    async def set_agent_state_to(self, new_state: AgentState) -> None:
        """Updates the agent's state and handles side effects. Can emit events to the event stream.

        Args:
            new_state (AgentState): The new state to set for the agent.
        """
        self.log(
            'info',
            f'Setting agent({self.agent.name}) state from {self.state.agent_state} to {new_state}',
        )

        if new_state == self.state.agent_state:
            return

        # Store old state for control limits check
        old_state = self.state.agent_state

        # Update agent state BEFORE calling _reset() so _reset() sees the correct state
        self.state.agent_state = new_state

        if new_state in (AgentState.STOPPED, AgentState.ERROR):
            self._reset()

        # User is allowing to check control limits and expand them if applicable
        if old_state == AgentState.ERROR and new_state == AgentState.RUNNING:
            self.state_tracker.maybe_increase_control_flags_limits(self.headless_mode)

        if self._pending_action is not None and (
            new_state in (AgentState.USER_CONFIRMED, AgentState.USER_REJECTED)
        ):
            if hasattr(self._pending_action, 'thought'):
                self._pending_action.thought = ''  # type: ignore[union-attr]
            if new_state == AgentState.USER_CONFIRMED:
                confirmation_state = ActionConfirmationStatus.CONFIRMED
            else:
                confirmation_state = ActionConfirmationStatus.REJECTED
            self._pending_action.confirmation_state = confirmation_state  # type: ignore[attr-defined]
            self._pending_action._id = None  # type: ignore[attr-defined]
            self.event_stream.add_event(self._pending_action, EventSource.AGENT)

        # Create observation with reason field if it's an error state
        reason = ''
        if new_state == AgentState.ERROR:
            reason = self.state.last_error

        self.event_stream.add_event(
            AgentStateChangedObservation('', self.state.agent_state, reason),
            EventSource.ENVIRONMENT,
        )

        # Save state whenever agent state changes to ensure we don't lose state
        # in case of crashes or unexpected circumstances
        self.save_state()

    def get_agent_state(self) -> AgentState:
        """Returns the current state of the agent.

        Returns:
            AgentState: The current state of the agent.
        """
        return self.state.agent_state

    async def start_delegate(self, action: AgentDelegateAction) -> None:
        """Start a delegate agent to handle a subtask.

        OpenHands is a multi-agentic system. A `task` is a conversation between
        OpenHands (the whole system) and the user, which might involve one or more inputs
        from the user. It starts with an initial input (typically a task statement) from
        the user, and ends with either an `AgentFinishAction` initiated by the agent, a
        stop initiated by the user, or an error.

        A `subtask` is a conversation between an agent and the user, or another agent. If a `task`
        is conducted by a single agent, then it's also a `subtask`. Otherwise, a `task` consists of
        multiple `subtasks`, each executed by one agent.

        Args:
            action (AgentDelegateAction): The action containing information about the delegate agent to start.
        """
        agent_cls: type[Agent] = Agent.get_cls(action.agent)
        agent_config = self.agent_configs.get(action.agent, self.agent.config)
        # Make sure metrics are shared between parent and child for global accumulation
        delegate_agent = agent_cls(
            config=agent_config, llm_registry=self.agent.llm_registry
        )

        # Take a snapshot of the current metrics before starting the delegate
        state = State(
            session_id=self.id.removesuffix('-delegate'),
            user_id=self.user_id,
            inputs=action.inputs or {},
            iteration_flag=self.state.iteration_flag,
            budget_flag=self.state.budget_flag,
            delegate_level=self.state.delegate_level + 1,
            # global metrics should be shared between parent and child
            metrics=self.state.metrics,
            # start on top of the stream
            start_id=self.event_stream.get_latest_event_id() + 1,
            parent_metrics_snapshot=self.state_tracker.get_metrics_snapshot(),
            parent_iteration=self.state.iteration_flag.current_value,
        )
        self.log(
            'debug',
            f'start delegate, creating agent {delegate_agent.name}',
        )

        # Create the delegate with is_delegate=True so it does NOT subscribe directly
        self.delegate = AgentController(
            sid=self.id + '-delegate',
            file_store=self.file_store,
            user_id=self.user_id,
            agent=delegate_agent,
            event_stream=self.event_stream,
            conversation_stats=self.conversation_stats,
            iteration_delta=self._initial_max_iterations,
            budget_per_task_delta=self._initial_max_budget_per_task,
            agent_to_llm_config=self.agent_to_llm_config,
            agent_configs=self.agent_configs,
            initial_state=state,
            is_delegate=True,
            headless_mode=self.headless_mode,
            security_analyzer=self.security_analyzer,
        )

    def end_delegate(self) -> None:
        """Ends the currently active delegate (e.g., if it is finished or errored).

        so that this controller can resume normal operation.
        """
        if self.delegate is None:
            return

        delegate_state = self.delegate.get_agent_state()

        # update iteration that is shared across agents
        self.state.iteration_flag.current_value = (
            self.delegate.state.iteration_flag.current_value
        )

        # Calculate delegate-specific metrics before closing the delegate
        delegate_metrics = self.state.get_local_metrics()
        logger.info(f'Local metrics for delegate: {delegate_metrics}')

        # close the delegate controller before adding new events
        asyncio.get_event_loop().run_until_complete(self.delegate.close())

        if delegate_state in (AgentState.FINISHED, AgentState.REJECTED):
            # retrieve delegate result
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )

            # prepare delegate result observation
            # TODO: replace this with AI-generated summary (#2395)
            # Filter out metrics from the formatted output to avoid clutter
            display_outputs = {
                k: v for k, v in delegate_outputs.items() if k != 'metrics'
            }
            formatted_output = ', '.join(
                f'{key}: {value}' for key, value in display_outputs.items()
            )
            content = (
                f'{self.delegate.agent.name} finishes task with {formatted_output}'
            )
        else:
            # delegate state is ERROR
            # emit AgentDelegateObservation with error content
            delegate_outputs = (
                self.delegate.state.outputs if self.delegate.state else {}
            )
            content = (
                f'{self.delegate.agent.name} encountered an error during execution.'
            )

        content = f'Delegated agent finished with result:\n\n{content}'

        # emit the delegate result observation
        obs = AgentDelegateObservation(outputs=delegate_outputs, content=content)

        # associate the delegate action with the initiating tool call
        for event in reversed(self.state.history):
            if isinstance(event, AgentDelegateAction):
                delegate_action = event
                obs.tool_call_metadata = delegate_action.tool_call_metadata
                break

        self.event_stream.add_event(obs, EventSource.AGENT)

        # unset delegate so parent can resume normal handling
        self.delegate = None

    async def _step(self) -> None:
        """Executes a single step of the parent or delegate agent. Detects stuck agents and limits on the number of iterations and the task budget."""
        if self.get_agent_state() != AgentState.RUNNING:
            self.log(
                'debug',
                f'Agent not stepping because state is {self.get_agent_state()} (not RUNNING)',
                extra={'msg_type': 'STEP_BLOCKED_STATE'},
            )
            return

        if self._pending_action:
            action_id = getattr(self._pending_action, 'id', 'unknown')
            action_type = type(self._pending_action).__name__
            self.log(
                'debug',
                f'Agent not stepping because of pending action: {action_type} (id={action_id})',
                extra={'msg_type': 'STEP_BLOCKED_PENDING_ACTION'},
            )
            return

        self.log(
            'debug',
            f'LEVEL {self.state.delegate_level} LOCAL STEP {self.state.get_local_step()} GLOBAL STEP {self.state.iteration_flag.current_value}',
            extra={'msg_type': 'STEP'},
        )

        # Synchronize spend across all llm services with the budget flag
        self.state_tracker.sync_budget_flag_with_metrics()
        if self.agent.config.enable_stuck_detection and self._is_stuck():
            await self._react_to_exception(
                AgentStuckInLoopError('Agent got stuck in a loop')
            )
            return

        try:
            self.state_tracker.run_control_flags()
        except Exception as e:
            logger.warning('Control flag limits hit')
            await self._react_to_exception(e)
            return

        action: Action = NullAction()

        if self._replay_manager.should_replay():
            # in replay mode, we don't let the agent to proceed
            # instead, we replay the action from the replay trajectory
            action = self._replay_manager.step()
        else:
            try:
                action = self.agent.step(self.state)
                if action is None:
                    raise LLMNoActionError('No action was returned')
                action._source = EventSource.AGENT  # type: ignore [attr-defined]
            except (
                LLMMalformedActionError,
                LLMNoActionError,
                LLMResponseError,
                FunctionCallValidationError,
                FunctionCallNotExistsError,
            ) as e:
                self.event_stream.add_event(
                    ErrorObservation(
                        content=str(e),
                    ),
                    EventSource.AGENT,
                )
                return
            except (ContextWindowExceededError, BadRequestError, OpenAIError) as e:
                # FIXME: this is a hack until a litellm fix is confirmed
                # Check if this is a nested context window error
                # We have to rely on string-matching because LiteLLM doesn't consistently
                # wrap the failure in a ContextWindowExceededError
                error_str = str(e).lower()
                if (
                    'contextwindowexceedederror' in error_str
                    or 'prompt is too long' in error_str
                    or 'input length and `max_tokens` exceed context limit' in error_str
                    or 'please reduce the length of' in error_str
                    or 'the request exceeds the available context size' in error_str
                    or 'context length exceeded' in error_str
                    # For OpenRouter context window errors
                    or (
                        'sambanovaexception' in error_str
                        and 'maximum context length' in error_str
                    )
                    # For SambaNova context window errors - only match when both patterns are present
                    or isinstance(e, ContextWindowExceededError)
                ):
                    if self.agent.config.enable_history_truncation:
                        self.event_stream.add_event(
                            CondensationRequestAction(), EventSource.AGENT
                        )
                        return
                    else:
                        raise LLMContextWindowExceedError()
                # Check if this is a tool call validation error that should be recoverable
                elif (
                    isinstance(e, BadRequestError)
                    and 'tool call validation failed' in error_str
                    and (
                        'missing properties' in error_str
                        or 'missing required' in error_str
                    )
                ):
                    # Handle tool call validation errors from Groq as recoverable errors
                    self.event_stream.add_event(
                        ErrorObservation(
                            content=f'Tool call validation failed: {str(e)}. Please check the tool parameters and try again.',
                        ),
                        EventSource.AGENT,
                    )
                    return
                else:
                    raise e

        if action.runnable:
            if self.state.confirmation_mode and (
                type(action) is CmdRunAction
                or type(action) is IPythonRunCellAction
                or type(action) is BrowseInteractiveAction
                or type(action) is FileEditAction
                or type(action) is FileReadAction
            ):
                # Handle security risk analysis using the dedicated method
                await self._handle_security_analyzer(action)

                # Check if the action has a security_risk attribute set by the LLM or security analyzer
                security_risk = getattr(
                    action, 'security_risk', ActionSecurityRisk.UNKNOWN
                )

                is_high_security_risk = security_risk == ActionSecurityRisk.HIGH
                is_ask_for_every_action = (
                    security_risk == ActionSecurityRisk.UNKNOWN
                    and not self.security_analyzer
                )

                # If security_risk is HIGH, requires confirmation
                # UNLESS it is CLI which will handle action risks it itself
                if self.agent.config.cli_mode:
                    # TODO(refactor): this is not ideal to have CLI been an exception
                    # We should refactor agent controller to consider this in the future
                    # See issue: https://github.com/OpenHands/OpenHands/issues/10464
                    action.confirmation_state = (  # type: ignore[union-attr]
                        ActionConfirmationStatus.AWAITING_CONFIRMATION
                    )
                # Only HIGH security risk actions require confirmation
                elif (
                    is_high_security_risk or is_ask_for_every_action
                ) and self.confirmation_mode:
                    logger.debug(
                        f'[non-CLI mode] Detected HIGH security risk in action: {action}. Ask for confirmation'
                    )
                    action.confirmation_state = (  # type: ignore[union-attr]
                        ActionConfirmationStatus.AWAITING_CONFIRMATION
                    )
            self._pending_action = action

        if not isinstance(action, NullAction):
            if (
                hasattr(action, 'confirmation_state')
                and action.confirmation_state
                == ActionConfirmationStatus.AWAITING_CONFIRMATION
            ):
                await self.set_agent_state_to(AgentState.AWAITING_USER_CONFIRMATION)

            # Create and log metrics for frontend display
            self._prepare_metrics_for_frontend(action)

            self.event_stream.add_event(action, action._source)  # type: ignore [attr-defined]

        log_level = 'info' if LOG_ALL_EVENTS else 'debug'
        self.log(log_level, str(action), extra={'msg_type': 'ACTION'})

    @property
    def _pending_action(self) -> Action | None:
        """Get the current pending action with time tracking.

        Returns:
            Action | None: The current pending action, or None if there isn't one.
        """
        if self._pending_action_info is None:
            return None

        action, timestamp = self._pending_action_info
        current_time = time.time()
        elapsed_time = current_time - timestamp

        # Log if the pending action has been active for a long time (but don't clear it)
        if elapsed_time > 60.0:  # 1 minute - just for logging purposes
            action_id = getattr(action, 'id', 'unknown')
            action_type = type(action).__name__
            self.log(
                'info',
                f'Pending action active for {elapsed_time:.2f}s: {action_type} (id={action_id})',
                extra={'msg_type': 'PENDING_ACTION_TIMEOUT'},
            )

        return action

    @_pending_action.setter
    def _pending_action(self, action: Action | None) -> None:
        """Set or clear the pending action with timestamp and logging.

        Args:
            action: The action to set as pending, or None to clear.
        """
        if action is None:
            if self._pending_action_info is not None:
                prev_action, timestamp = self._pending_action_info
                action_id = getattr(prev_action, 'id', 'unknown')
                action_type = type(prev_action).__name__
                elapsed_time = time.time() - timestamp
                self.log(
                    'debug',
                    f'Cleared pending action after {elapsed_time:.2f}s: {action_type} (id={action_id})',
                    extra={'msg_type': 'PENDING_ACTION_CLEARED'},
                )
            self._pending_action_info = None
        else:
            action_id = getattr(action, 'id', 'unknown')
            action_type = type(action).__name__
            self.log(
                'debug',
                f'Set pending action: {action_type} (id={action_id})',
                extra={'msg_type': 'PENDING_ACTION_SET'},
            )
            self._pending_action_info = (action, time.time())

    def get_state(self) -> State:
        """Returns the current running state object.

        Returns:
            State: The current state object.
        """
        return self.state

    def set_initial_state(
        self,
        state: State | None,
        conversation_stats: ConversationStats,
        max_iterations: int,
        max_budget_per_task: float | None,
        confirmation_mode: bool = False,
    ):
        self.state_tracker.set_initial_state(
            self.id,
            state,
            conversation_stats,
            max_iterations,
            max_budget_per_task,
            confirmation_mode,
        )
        # Always load from the event stream to avoid losing history
        self.state_tracker._init_history(
            self.event_stream,
        )

    def get_trajectory(self, include_screenshots: bool = False) -> list[dict]:
        # state history could be partially hidden/truncated before controller is closed
        assert self._closed
        return self.state_tracker.get_trajectory(include_screenshots)

    def _is_stuck(self) -> bool:
        """Checks if the agent or its delegate is stuck in a loop.

        Returns:
            bool: True if the agent is stuck, False otherwise.
        """
        # check if delegate stuck
        if self.delegate and self.delegate._is_stuck():
            return True

        return self._stuck_detector.is_stuck(self.headless_mode)

    def attempt_loop_recovery(self) -> bool:
        """Attempts loop recovery when agent is stuck in a loop.
        Only supports CLI for now.

        Returns:
            bool: True if recovery was successful and agent should continue,
                  False if recovery failed or was not attempted.
        """
        # Check if we're in a loop
        if not self._stuck_detector.stuck_analysis:
            return False

        """Handle loop recovery in CLI mode by pausing the agent and presenting recovery options."""
        recovery_point = self._stuck_detector.stuck_analysis.loop_start_idx

        # Present loop detection message
        self.event_stream.add_event(
            LoopDetectionObservation(
                content=f"""⚠️  Agent detected in a loop!
Loop type: {self._stuck_detector.stuck_analysis.loop_type}
Loop detected at iteration {self.state.iteration_flag.current_value}
\nRecovery options:
/resume 1. Restart from before loop (preserves {recovery_point} events)
/resume 2. Restart with last user message (reuses your most recent instruction)
/exit. Quit directly
\nThe agent has been paused. Type '/resume 1', '/resume 2', or '/exit' to choose an option.
"""
            ),
            source=EventSource.ENVIRONMENT,
        )

        # Pause the agent using the same mechanism as Ctrl+P
        # This ensures consistent behavior and avoids event loop conflicts
        self.event_stream.add_event(
            ChangeAgentStateAction(AgentState.PAUSED),
            EventSource.ENVIRONMENT,  # Use ENVIRONMENT source to distinguish from user pause
        )
        return True

    def _prepare_metrics_for_frontend(self, action: Action) -> None:
        """Create a minimal metrics object for frontend display and log it.

        To avoid performance issues with long conversations, we only keep:
        - accumulated_cost: The current total cost
        - accumulated_token_usage: Accumulated token statistics across all API calls
        - max_budget_per_task: The maximum budget allowed for the task

        This includes metrics from both the agent's LLM and the condenser's LLM if it exists.

        Args:
            action: The action to attach metrics to
        """
        # Get metrics from agent LLM
        metrics = self.conversation_stats.get_combined_metrics()

        # Create a clean copy with only the fields we want to keep
        clean_metrics = Metrics()
        clean_metrics.accumulated_cost = metrics.accumulated_cost
        clean_metrics._accumulated_token_usage = copy.deepcopy(
            metrics.accumulated_token_usage
        )

        # Add max_budget_per_task to metrics
        if self.state.budget_flag:
            clean_metrics.max_budget_per_task = self.state.budget_flag.max_value

        action.llm_metrics = clean_metrics

        # Log the metrics information for debugging
        # Get the latest usage directly from the agent's metrics
        latest_usage = None
        if self.state.metrics.token_usages:
            latest_usage = self.state.metrics.token_usages[-1]

        accumulated_usage = self.state.metrics.accumulated_token_usage
        self.log(
            'debug',
            f'Action metrics - accumulated_cost: {metrics.accumulated_cost}, max_budget: {metrics.max_budget_per_task}, '
            f'latest tokens (prompt/completion/cache_read/cache_write): '
            f'{latest_usage.prompt_tokens if latest_usage else 0}/'
            f'{latest_usage.completion_tokens if latest_usage else 0}/'
            f'{latest_usage.cache_read_tokens if latest_usage else 0}/'
            f'{latest_usage.cache_write_tokens if latest_usage else 0}, '
            f'accumulated tokens (prompt/completion): '
            f'{accumulated_usage.prompt_tokens}/'
            f'{accumulated_usage.completion_tokens}',
            extra={'msg_type': 'METRICS'},
        )

    def __repr__(self) -> str:
        pending_action_info = '<none>'
        if (
            hasattr(self, '_pending_action_info')
            and self._pending_action_info is not None
        ):
            action, timestamp = self._pending_action_info
            action_id = getattr(action, 'id', 'unknown')
            action_type = type(action).__name__
            elapsed_time = time.time() - timestamp
            pending_action_info = (
                f'{action_type}(id={action_id}, elapsed={elapsed_time:.2f}s)'
            )

        return (
            f'AgentController(id={getattr(self, "id", "<uninitialized>")}, '
            f'agent={getattr(self, "agent", "<uninitialized>")!r}, '
            f'event_stream={getattr(self, "event_stream", "<uninitialized>")!r}, '
            f'state={getattr(self, "state", "<uninitialized>")!r}, '
            f'delegate={getattr(self, "delegate", "<uninitialized>")!r}, '
            f'_pending_action={pending_action_info})'
        )

    def _is_awaiting_observation(self) -> bool:
        events = self.event_stream.search_events(reverse=True)
        for event in events:
            if isinstance(event, AgentStateChangedObservation):
                result = event.agent_state == AgentState.RUNNING
                return result
        return False

    def _first_user_message(
        self, events: list[Event] | None = None
    ) -> MessageAction | None:
        """Get the first user message for this agent.

        For regular agents, this is the first user message from the beginning (start_id=0).
        For delegate agents, this is the first user message after the delegate's start_id.

        Args:
            events: Optional list of events to search through. If None, uses the event stream.

        Returns:
            MessageAction | None: The first user message, or None if no user message found
        """
        # If events list is provided, search through it
        if events is not None:
            return next(
                (
                    e
                    for e in events
                    if isinstance(e, MessageAction) and e.source == EventSource.USER
                ),
                None,
            )

        # Otherwise, use the original event stream logic with caching
        # Return cached message if any
        if self._cached_first_user_message is not None:
            return self._cached_first_user_message

        # Find the first user message
        self._cached_first_user_message = next(
            (
                e
                for e in self.event_stream.search_events(
                    start_id=self.state.start_id,
                )
                if isinstance(e, MessageAction) and e.source == EventSource.USER
            ),
            None,
        )
        return self._cached_first_user_message

    async def _perform_loop_recovery(
        self, stuck_analysis: StuckDetector.StuckAnalysis
    ) -> None:
        """Perform loop recovery by truncating memory and restarting from before the loop."""
        recovery_point = stuck_analysis.loop_start_idx

        # Truncate memory to the recovery point
        await self._truncate_memory_to_point(recovery_point)

        # Set agent state to AWAITING_USER_INPUT to allow user to provide new instructions
        await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)

        self.event_stream.add_event(
            LoopDetectionObservation(
                content="""✅ Loop recovery completed. Agent has been reset to before the loop.
You can now provide new instructions to continue.
"""
            ),
            source=EventSource.ENVIRONMENT,
        )

    async def _truncate_memory_to_point(self, recovery_point: int) -> None:
        """Truncate memory to the specified recovery point."""
        # Get all events from state history
        all_events = self.state.history

        if recovery_point >= len(all_events):
            return

        # Keep only events up to the recovery point
        events_to_keep = all_events[:recovery_point]

        # Update state history
        self.state.history = events_to_keep

        # Update end_id to reflect the truncation
        if events_to_keep:
            self.state.end_id = events_to_keep[-1].id
        else:
            self.state.end_id = -1

        # Clear any cached messages
        self._cached_first_user_message = None

    async def _restart_with_last_user_message(
        self, stuck_analysis: StuckDetector.StuckAnalysis
    ) -> None:
        """Restart the agent using the last user message as the new instruction."""

        # Find the last user message in the history
        last_user_message = None
        for event in reversed(self.state.history):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                last_user_message = event
                break

        if last_user_message:
            # Truncate memory to just before the loop started
            recovery_point = stuck_analysis.loop_start_idx
            await self._truncate_memory_to_point(recovery_point)

            # Set agent state to RUNNING and re-use the last user message
            await self.set_agent_state_to(AgentState.RUNNING)

            # Re-use the last user message as the new instruction
            self.event_stream.add_event(
                LoopDetectionObservation(
                    content=f"""\n✅ Restarting with your last instruction: {last_user_message.content}
Agent is now continuing with the same task...
"""
                ),
                source=EventSource.ENVIRONMENT,
            )

            # Create a new action with the last user message
            new_action = MessageAction(
                content=last_user_message.content, wait_for_response=False
            )
            new_action._source = EventSource.USER  # type: ignore [attr-defined]

            # Process the action to restart the agent
            await self._handle_action(new_action)
        else:
            # If no user message found, fall back to regular recovery
            print('\n⚠️  No previous user message found. Using standard recovery.')
            await self._perform_loop_recovery(stuck_analysis)

    def save_state(self):
        self.state_tracker.save_state()
