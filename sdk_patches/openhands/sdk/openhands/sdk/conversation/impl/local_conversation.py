import atexit
import uuid
from collections.abc import Mapping
from pathlib import Path

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.context.prompts.prompt import render_template
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.secret_registry import SecretValue
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.conversation.stuck_detector import StuckDetector
from openhands.sdk.conversation.title_utils import generate_conversation_title
from openhands.sdk.conversation.types import (
    ConversationCallbackType,
    ConversationID,
    ConversationTokenCallbackType,
    StuckDetectionThresholds,
)
from openhands.sdk.conversation.visualizer import (
    ConversationVisualizerBase,
    DefaultConversationVisualizer,
)
from openhands.sdk.event import (
    CondensationRequest,
    MessageEvent,
    PauseEvent,
    UserRejectObservation,
)
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.hooks import HookConfig, HookEventProcessor, create_hook_callback
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.llm.llm_registry import LLMRegistry
from openhands.sdk.logger import get_logger
from openhands.sdk.observability.laminar import observe
from openhands.sdk.security.analyzer import SecurityAnalyzerBase
from openhands.sdk.security.confirmation_policy import (
    ConfirmationPolicyBase,
)
from openhands.sdk.workspace import LocalWorkspace


logger = get_logger(__name__)


class LocalConversation(BaseConversation):
    agent: AgentBase
    workspace: LocalWorkspace
    _state: ConversationState
    _visualizer: ConversationVisualizerBase | None
    _on_event: ConversationCallbackType
    _on_token: ConversationTokenCallbackType | None
    max_iteration_per_run: int
    _stuck_detector: StuckDetector | None
    llm_registry: LLMRegistry
    _cleanup_initiated: bool
    _hook_processor: HookEventProcessor | None

    def __init__(
        self,
        agent: AgentBase,
        workspace: str | Path | LocalWorkspace,
        persistence_dir: str | Path | None = None,
        conversation_id: ConversationID | None = None,
        callbacks: list[ConversationCallbackType] | None = None,
        token_callbacks: list[ConversationTokenCallbackType] | None = None,
        hook_config: HookConfig | None = None,
        max_iteration_per_run: int = 500,
        stuck_detection: bool = True,
        stuck_detection_thresholds: (
            StuckDetectionThresholds | Mapping[str, int] | None
        ) = None,
        visualizer: (
            type[ConversationVisualizerBase] | ConversationVisualizerBase | None
        ) = DefaultConversationVisualizer,
        secrets: Mapping[str, SecretValue] | None = None,
        **_: object,
    ):
        """Initialize the conversation.

        Args:
            agent: The agent to use for the conversation
            workspace: Working directory for agent operations and tool execution.
                Can be a string path, Path object, or LocalWorkspace instance.
            persistence_dir: Directory for persisting conversation state and events.
                Can be a string path or Path object.
            conversation_id: Optional ID for the conversation. If provided, will
                      be used to identify the conversation. The user might want to
                      suffix their persistent filestore with this ID.
            callbacks: Optional list of callback functions to handle events
            token_callbacks: Optional list of callbacks invoked for streaming deltas
            hook_config: Optional hook configuration to auto-wire session hooks
            max_iteration_per_run: Maximum number of iterations per run
            visualizer: Visualization configuration. Can be:
                       - ConversationVisualizerBase subclass: Class to instantiate
                         (default: ConversationVisualizer)
                       - ConversationVisualizerBase instance: Use custom visualizer
                       - None: No visualization
            stuck_detection: Whether to enable stuck detection
            stuck_detection_thresholds: Optional configuration for stuck detection
                      thresholds. Can be a StuckDetectionThresholds instance or
                      a dict with keys: 'action_observation', 'action_error',
                      'monologue', 'alternating_pattern'. Values are integers
                      representing the number of repetitions before triggering.
        """
        super().__init__()  # Initialize with span tracking
        # Mark cleanup as initiated as early as possible to avoid races or partially
        # initialized instances during interpreter shutdown.
        self._cleanup_initiated = False

        self.agent = agent
        if isinstance(workspace, (str, Path)):
            # LocalWorkspace accepts both str and Path via BeforeValidator
            workspace = LocalWorkspace(working_dir=workspace)
        assert isinstance(workspace, LocalWorkspace), (
            "workspace must be a LocalWorkspace instance"
        )
        self.workspace = workspace
        ws_path = Path(self.workspace.working_dir)
        if not ws_path.exists():
            ws_path.mkdir(parents=True, exist_ok=True)

        # Create-or-resume: factory inspects BASE_STATE to decide
        desired_id = conversation_id or uuid.uuid4()
        self._state = ConversationState.create(
            id=desired_id,
            agent=agent,
            workspace=self.workspace,
            persistence_dir=self.get_persistence_dir(persistence_dir, desired_id)
            if persistence_dir
            else None,
            max_iterations=max_iteration_per_run,
            stuck_detection=stuck_detection,
        )

        # Default callback: persist every event to state
        def _default_callback(e):
            self._state.events.append(e)

        self._hook_processor = None
        hook_callback = None
        if hook_config is not None:
            self._hook_processor, hook_callback = create_hook_callback(
                hook_config=hook_config,
                working_dir=str(self.workspace.working_dir),
                session_id=str(desired_id),
            )

        callback_list = list(callbacks) if callbacks else []
        if hook_callback is not None:
            callback_list.insert(0, hook_callback)

        composed_list = callback_list + [_default_callback]
        # Handle visualization configuration
        if isinstance(visualizer, ConversationVisualizerBase):
            # Use custom visualizer instance
            self._visualizer = visualizer
            # Initialize the visualizer with conversation state
            self._visualizer.initialize(self._state)
            composed_list = [self._visualizer.on_event] + composed_list
            # visualizer should happen first for visibility
        elif isinstance(visualizer, type) and issubclass(
            visualizer, ConversationVisualizerBase
        ):
            # Instantiate the visualizer class with appropriate parameters
            self._visualizer = visualizer()
            # Initialize with state
            self._visualizer.initialize(self._state)
            composed_list = [self._visualizer.on_event] + composed_list
            # visualizer should happen first for visibility
        else:
            # No visualization (visualizer is None)
            self._visualizer = None

        self._on_event = BaseConversation.compose_callbacks(composed_list)
        self._on_token = (
            BaseConversation.compose_callbacks(token_callbacks)
            if token_callbacks
            else None
        )

        self.max_iteration_per_run = max_iteration_per_run

        # Initialize stuck detector
        if stuck_detection:
            # Convert dict to StuckDetectionThresholds if needed
            if isinstance(stuck_detection_thresholds, Mapping):
                threshold_config = StuckDetectionThresholds(
                    **stuck_detection_thresholds
                )
            else:
                threshold_config = stuck_detection_thresholds
            self._stuck_detector = StuckDetector(
                self._state,
                thresholds=threshold_config,
            )
        else:
            self._stuck_detector = None

        if self._hook_processor is not None:
            self._hook_processor.set_conversation_state(self._state)
            self._hook_processor.run_session_start()

        with self._state:
            self.agent.init_state(self._state, on_event=self._on_event)

        # Register existing llms in agent
        self.llm_registry = LLMRegistry()
        self.llm_registry.subscribe(self._state.stats.register_llm)
        for llm in list(self.agent.get_all_llms()):
            self.llm_registry.add(llm)

        # Initialize secrets if provided
        if secrets:
            # Convert dict[str, str] to dict[str, SecretValue]
            secret_values: dict[str, SecretValue] = {k: v for k, v in secrets.items()}
            self.update_secrets(secret_values)

        atexit.register(self.close)
        self._start_observability_span(str(desired_id))

    @property
    def id(self) -> ConversationID:
        """Get the unique ID of the conversation."""
        return self._state.id

    @property
    def state(self) -> ConversationState:
        """Get the conversation state.

        It returns a protocol that has a subset of ConversationState methods
        and properties. We will have the ability to access the same properties
        of ConversationState on a remote conversation object.
        But we won't be able to access methods that mutate the state.
        """
        return self._state

    @property
    def conversation_stats(self):
        return self._state.stats

    @property
    def stuck_detector(self) -> StuckDetector | None:
        """Get the stuck detector instance if enabled."""
        return self._stuck_detector

    @observe(name="conversation.send_message")
    def send_message(self, message: str | Message, sender: str | None = None) -> None:
        """Send a message to the agent.

        Args:
            message: Either a string (which will be converted to a user message)
                    or a Message object
            sender: Optional identifier of the sender. Can be used to track
                   message origin in multi-agent scenarios. For example, when
                   one agent delegates to another, the sender can be set to
                   identify which agent is sending the message.
        """
        # Convert string to Message if needed
        if isinstance(message, str):
            message = Message(role="user", content=[TextContent(text=message)])

        assert message.role == "user", (
            "Only user messages are allowed to be sent to the agent."
        )
        with self._state:
            if self._state.execution_status == ConversationExecutionStatus.FINISHED:
                self._state.execution_status = (
                    ConversationExecutionStatus.IDLE
                )  # now we have a new message

            # TODO: We should add test cases for all these scenarios
            activated_skill_names: list[str] = []
            extended_content: list[TextContent] = []

            # Handle per-turn user message (i.e., knowledge agent trigger)
            if self.agent.agent_context:
                ctx = self.agent.agent_context.get_user_message_suffix(
                    user_message=message,
                    # We skip skills that were already activated
                    skip_skill_names=self._state.activated_knowledge_skills,
                )
                # TODO(calvin): we need to update
                # self._state.activated_knowledge_skills
                # so condenser can work
                if ctx:
                    content, activated_skill_names = ctx
                    logger.debug(
                        f"Got augmented user message content: {content}, "
                        f"activated skills: {activated_skill_names}"
                    )
                    extended_content.append(content)
                    self._state.activated_knowledge_skills.extend(activated_skill_names)

            user_msg_event = MessageEvent(
                source="user",
                llm_message=message,
                activated_skills=activated_skill_names,
                extended_content=extended_content,
                sender=sender,
            )
            self._on_event(user_msg_event)

    @observe(name="conversation.run")
    def run(self) -> None:
        """Runs the conversation until the agent finishes.

        In confirmation mode:
        - First call: creates actions but doesn't execute them, stops and waits
        - Second call: executes pending actions (implicit confirmation)

        In normal mode:
        - Creates and executes actions immediately

        Can be paused between steps
        """

        with self._state:
            if self._state.execution_status in [
                ConversationExecutionStatus.IDLE,
                ConversationExecutionStatus.PAUSED,
                ConversationExecutionStatus.ERROR,
            ]:
                self._state.execution_status = ConversationExecutionStatus.RUNNING

        iteration = 0
        try:
            while True:
                logger.debug(f"Conversation run iteration {iteration}")
                with self._state:
                    # Pause attempts to acquire the state lock
                    # Before value can be modified step can be taken
                    # Ensure step conditions are checked when lock is already acquired
                    if self._state.execution_status in [
                        ConversationExecutionStatus.FINISHED,
                        ConversationExecutionStatus.PAUSED,
                        ConversationExecutionStatus.STUCK,
                    ]:
                        break

                    # Check for stuck patterns if enabled
                    if self._stuck_detector:
                        is_stuck = self._stuck_detector.is_stuck()

                        if is_stuck:
                            logger.warning("Stuck pattern detected.")
                            self._state.execution_status = (
                                ConversationExecutionStatus.STUCK
                            )
                            continue

                    # clear the flag before calling agent.step() (user approved)
                    if (
                        self._state.execution_status
                        == ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
                    ):
                        self._state.execution_status = (
                            ConversationExecutionStatus.RUNNING
                        )

                    self.agent.step(
                        self, on_event=self._on_event, on_token=self._on_token
                    )
                    iteration += 1

                    # Check for non-finished terminal conditions
                    # Note: We intentionally do NOT check for FINISHED status here.
                    # This allows concurrent user messages to be processed:
                    # 1. Agent finishes and sets status to FINISHED
                    # 2. User sends message concurrently via send_message()
                    # 3. send_message() waits for FIFO lock, then sets status to IDLE
                    # 4. Run loop continues to next iteration and processes the message
                    # 5. Without this design, concurrent messages would be lost
                    if (
                        self.state.execution_status
                        == ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
                    ):
                        break

                    if iteration >= self.max_iteration_per_run:
                        error_msg = (
                            f"Agent reached maximum iterations limit "
                            f"({self.max_iteration_per_run})."
                        )
                        logger.error(error_msg)
                        self._state.execution_status = ConversationExecutionStatus.ERROR
                        self._on_event(
                            ConversationErrorEvent(
                                source="environment",
                                code="MaxIterationsReached",
                                detail=error_msg,
                            )
                        )
                        break
        except Exception as e:
            self._state.execution_status = ConversationExecutionStatus.ERROR

            # Add an error event
            self._on_event(
                ConversationErrorEvent(
                    source="environment",
                    code=e.__class__.__name__,
                    detail=str(e),
                )
            )

            # Re-raise with conversation id and persistence dir for better UX
            raise ConversationRunError(
                self._state.id, e, persistence_dir=self._state.persistence_dir
            ) from e

    def set_confirmation_policy(self, policy: ConfirmationPolicyBase) -> None:
        """Set the confirmation policy and store it in conversation state."""
        with self._state:
            self._state.confirmation_policy = policy
        logger.info(f"Confirmation policy set to: {policy}")

    def reject_pending_actions(self, reason: str = "User rejected the action") -> None:
        """Reject all pending actions from the agent.

        This is a non-invasive method to reject actions between run() calls.
        Also clears the agent_waiting_for_confirmation flag.
        """
        pending_actions = ConversationState.get_unmatched_actions(self._state.events)

        with self._state:
            # Always clear the agent_waiting_for_confirmation flag
            if (
                self._state.execution_status
                == ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
            ):
                self._state.execution_status = ConversationExecutionStatus.IDLE

            if not pending_actions:
                logger.warning("No pending actions to reject")
                return

            for action_event in pending_actions:
                # Create rejection observation
                rejection_event = UserRejectObservation(
                    action_id=action_event.id,
                    tool_name=action_event.tool_name,
                    tool_call_id=action_event.tool_call_id,
                    rejection_reason=reason,
                )
                self._on_event(rejection_event)
                logger.info(f"Rejected pending action: {action_event} - {reason}")

    def pause(self) -> None:
        """Pause agent execution.

        This method can be called from any thread to request that the agent
        pause execution. The pause will take effect at the next iteration
        of the run loop (between agent steps).

        Note: If called during an LLM completion, the pause will not take
        effect until the current LLM call completes.
        """

        if self._state.execution_status == ConversationExecutionStatus.PAUSED:
            return

        with self._state:
            # Only pause when running or idle
            if (
                self._state.execution_status == ConversationExecutionStatus.IDLE
                or self._state.execution_status == ConversationExecutionStatus.RUNNING
            ):
                self._state.execution_status = ConversationExecutionStatus.PAUSED
                pause_event = PauseEvent()
                self._on_event(pause_event)
                logger.info("Agent execution pause requested")

    def update_secrets(self, secrets: Mapping[str, SecretValue]) -> None:
        """Add secrets to the conversation.

        Args:
            secrets: Dictionary mapping secret keys to values or no-arg callables.
                     SecretValue = str | Callable[[], str]. Callables are invoked lazily
                     when a command references the secret key.
        """

        secret_registry = self._state.secret_registry
        secret_registry.update_secrets(secrets)
        logger.info(f"Added {len(secrets)} secrets to conversation")

    def set_security_analyzer(self, analyzer: SecurityAnalyzerBase | None) -> None:
        """Set the security analyzer for the conversation."""
        with self._state:
            self._state.security_analyzer = analyzer

    def close(self) -> None:
        """Close the conversation and clean up all tool executors."""
        # Use getattr for safety - object may be partially constructed
        if getattr(self, "_cleanup_initiated", False):
            return
        self._cleanup_initiated = True
        logger.debug("Closing conversation and cleaning up tool executors")
        hook_processor = getattr(self, "_hook_processor", None)
        if hook_processor is not None:
            hook_processor.run_session_end()
        try:
            self._end_observability_span()
        except AttributeError:
            # Object may be partially constructed; span fields may be missing.
            pass
        try:
            tools_map = self.agent.tools_map
        except (AttributeError, RuntimeError):
            # Agent not initialized or partially constructed
            return
        for tool in tools_map.values():
            try:
                executable_tool = tool.as_executable()
                executable_tool.executor.close()
            except NotImplementedError:
                # Tool has no executor, skip it without erroring
                continue
            except Exception as e:
                logger.warning(f"Error closing executor for tool '{tool.name}': {e}")

    def ask_agent(self, question: str) -> str:
        """Ask the agent a simple, stateless question and get a direct LLM response.

        This bypasses the normal conversation flow and does **not** modify, persist,
        or become part of the conversation state. The request is not remembered by
        the main agent, no events are recorded, and execution status is untouched.
        It is also thread-safe and may be called while `conversation.run()` is
        executing in another thread.

        Args:
            question: A simple string question to ask the agent

        Returns:
            A string response from the agent
        """
        # Import here to avoid circular imports
        from openhands.sdk.agent.utils import make_llm_completion, prepare_llm_messages

        template_dir = (
            Path(__file__).parent.parent.parent / "context" / "prompts" / "templates"
        )

        question_text = render_template(
            str(template_dir), "ask_agent_template.j2", question=question
        )

        # Create a user message with the context-aware question
        user_message = Message(
            role="user",
            content=[TextContent(text=question_text)],
        )

        messages = prepare_llm_messages(
            self.state.events, additional_messages=[user_message]
        )

        # Get or create the specialized ask-agent LLM
        try:
            question_llm = self.llm_registry.get("ask-agent-llm")
        except KeyError:
            question_llm = self.agent.llm.model_copy(
                update={
                    "usage_id": "ask-agent-llm",
                },
                deep=True,
            )
            self.llm_registry.add(question_llm)

        # Pass agent tools so LLM can understand tool_calls in conversation history
        response = make_llm_completion(
            question_llm, messages, tools=list(self.agent.tools_map.values())
        )

        message = response.message

        # Extract the text content from the LLMResponse message
        if message.content and len(message.content) > 0:
            # Look for the first TextContent in the response
            for content in response.message.content:
                if isinstance(content, TextContent):
                    return content.text

        raise Exception("Failed to generate summary")

    @observe(name="conversation.generate_title", ignore_inputs=["llm"])
    def generate_title(self, llm: LLM | None = None, max_length: int = 50) -> str:
        """Generate a title for the conversation based on the first user message.

        Args:
            llm: Optional LLM to use for title generation. If not provided,
                 uses self.agent.llm.
            max_length: Maximum length of the generated title.

        Returns:
            A generated title for the conversation.

        Raises:
            ValueError: If no user messages are found in the conversation.
        """
        # Use provided LLM or fall back to agent's LLM
        llm_to_use = llm or self.agent.llm

        return generate_conversation_title(
            events=self._state.events, llm=llm_to_use, max_length=max_length
        )

    def condense(self) -> None:
        """Synchronously force condense the conversation history.

        If the agent is currently running, `condense()` will wait for the
        ongoing step to finish before proceeding.

        Raises ValueError if no compatible condenser exists.
        """

        # Check if condenser is configured and handles condensation requests
        if (
            self.agent.condenser is None
            or not self.agent.condenser.handles_condensation_requests()
        ):
            condenser_info = (
                "No condenser configured"
                if self.agent.condenser is None
                else (
                    f"Condenser {type(self.agent.condenser).__name__} does not handle "
                    "condensation requests"
                )
            )
            raise ValueError(
                f"Cannot condense conversation: {condenser_info}. "
                "To enable manual condensation, configure an "
                "LLMSummarizingCondenser:\n\n"
                "from openhands.sdk.context.condenser import LLMSummarizingCondenser\n"
                "agent = Agent(\n"
                "    llm=your_llm,\n"
                "    condenser=LLMSummarizingCondenser(\n"
                "        llm=your_llm,\n"
                "        max_size=120,\n"
                "        keep_first=4\n"
                "    )\n"
                ")"
            )

        # Add a condensation request event
        condensation_request = CondensationRequest()
        self._on_event(condensation_request)

        # Force the agent to take a single step to process the condensation request
        # This will trigger the condenser if it handles condensation requests
        with self._state:
            # Take a single step to process the condensation request
            self.agent.step(self, on_event=self._on_event, on_token=self._on_token)

        logger.info("Condensation request processed")

    def __del__(self) -> None:
        """Ensure cleanup happens when conversation is destroyed."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error during conversation cleanup: {e}", exc_info=True)
