"""Hook integration for conversations."""

from typing import TYPE_CHECKING, Any

from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.hooks.config import HookConfig
from openhands.sdk.hooks.manager import HookManager
from openhands.sdk.hooks.types import HookEventType
from openhands.sdk.logger import get_logger


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

logger = get_logger(__name__)


class HookEventProcessor:
    """Processes events and runs hooks at appropriate points.

    Call set_conversation_state() after creating Conversation for blocking to work.

    Note on persistence: HookEvent/HookResult are ephemeral (for hook script I/O).
    If hook execution traces need to be persisted (e.g., for observability), create
    a HookExecutionObservation inheriting from Observation and emit it through the
    event stream, rather than modifying these hook classes.
    """

    def __init__(
        self,
        hook_manager: HookManager,
        original_callback: Any = None,
    ):
        self.hook_manager = hook_manager
        self.original_callback = original_callback
        self._conversation_state: ConversationState | None = None

    def set_conversation_state(self, state: "ConversationState") -> None:
        """Set conversation state for blocking support."""
        self._conversation_state = state

    def on_event(self, event: Event) -> None:
        """Process an event and run appropriate hooks."""
        # Run PreToolUse hooks for action events
        if isinstance(event, ActionEvent) and event.action is not None:
            self._handle_pre_tool_use(event)

        # Run PostToolUse hooks for observation events
        if isinstance(event, ObservationEvent):
            self._handle_post_tool_use(event)

        # Run UserPromptSubmit hooks for user messages
        if isinstance(event, MessageEvent) and event.source == "user":
            self._handle_user_prompt_submit(event)

        # Call original callback
        if self.original_callback:
            self.original_callback(event)

    def _handle_pre_tool_use(self, event: ActionEvent) -> None:
        """Handle PreToolUse hooks. Blocked actions are marked in conversation state."""
        if not self.hook_manager.has_hooks(HookEventType.PRE_TOOL_USE):
            return

        tool_name = event.tool_name
        tool_input = {}

        # Extract tool input from action
        if event.action is not None:
            try:
                tool_input = event.action.model_dump()
            except Exception as e:
                logger.debug(f"Could not extract tool input: {e}")

        should_continue, results = self.hook_manager.run_pre_tool_use(
            tool_name=tool_name,
            tool_input=tool_input,
        )

        if not should_continue:
            reason = self.hook_manager.get_blocking_reason(results)
            logger.warning(f"Hook blocked action {tool_name}: {reason}")

            # Mark this action as blocked in the conversation state
            # The Agent will check this and emit a rejection instead of executing
            if self._conversation_state is not None:
                block_reason = reason or "Blocked by hook"
                self._conversation_state.block_action(event.id, block_reason)
            else:
                logger.warning(
                    "Cannot block action: conversation state not set. "
                    "Call processor.set_conversation_state(conversation.state) "
                    "after creating the Conversation."
                )

    def _handle_post_tool_use(self, event: ObservationEvent) -> None:
        """Handle PostToolUse hooks after an action completes."""
        if not self.hook_manager.has_hooks(HookEventType.POST_TOOL_USE):
            return

        # O(1) lookup of corresponding action from state events
        action_event = None
        if self._conversation_state is not None:
            try:
                idx = self._conversation_state.events.get_index(event.action_id)
                event_at_idx = self._conversation_state.events[idx]
                if isinstance(event_at_idx, ActionEvent):
                    action_event = event_at_idx
            except KeyError:
                pass  # action not found

        if action_event is None:
            return

        tool_name = event.tool_name
        tool_input: dict[str, Any] = {}
        tool_response: dict[str, Any] = {}

        # Extract tool input from action
        if action_event.action is not None:
            try:
                tool_input = action_event.action.model_dump()
            except Exception as e:
                logger.debug(f"Could not extract tool input: {e}")

        # Extract structured tool response from observation
        if event.observation is not None:
            try:
                tool_response = event.observation.model_dump()
            except Exception as e:
                logger.debug(f"Could not extract tool response: {e}")

        results = self.hook_manager.run_post_tool_use(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
        )

        # Log any hook errors
        for result in results:
            if result.error:
                logger.warning(f"PostToolUse hook error: {result.error}")

    def _handle_user_prompt_submit(self, event: MessageEvent) -> None:
        """Handle UserPromptSubmit hooks before processing a user message."""
        if not self.hook_manager.has_hooks(HookEventType.USER_PROMPT_SUBMIT):
            return

        # Extract message text
        message = ""
        if event.llm_message and event.llm_message.content:
            from openhands.sdk.llm import TextContent

            for content in event.llm_message.content:
                if isinstance(content, TextContent):
                    message += content.text

        should_continue, additional_context, results = (
            self.hook_manager.run_user_prompt_submit(message=message)
        )

        if not should_continue:
            reason = self.hook_manager.get_blocking_reason(results)
            logger.warning(f"Hook blocked user message: {reason}")

            # Mark this message as blocked in the conversation state
            # The Agent will check this and skip processing the message
            if self._conversation_state is not None:
                block_reason = reason or "Blocked by hook"
                self._conversation_state.block_message(event.id, block_reason)
            else:
                logger.warning(
                    "Cannot block message: conversation state not set. "
                    "Call processor.set_conversation_state(conversation.state) "
                    "after creating the Conversation."
                )

        # TODO: Inject additional_context into the message
        if additional_context:
            logger.info(f"Hook injected context: {additional_context[:100]}...")

    def is_action_blocked(self, action_id: str) -> bool:
        """Check if an action was blocked by a hook."""
        if self._conversation_state is None:
            return False
        return action_id in self._conversation_state.blocked_actions

    def is_message_blocked(self, message_id: str) -> bool:
        """Check if a message was blocked by a hook."""
        if self._conversation_state is None:
            return False
        return message_id in self._conversation_state.blocked_messages

    def run_session_start(self) -> None:
        """Run SessionStart hooks. Call after conversation is created."""
        results = self.hook_manager.run_session_start()
        for r in results:
            if r.error:
                logger.warning(f"SessionStart hook error: {r.error}")

    def run_session_end(self) -> None:
        """Run SessionEnd hooks. Call before conversation is closed."""
        results = self.hook_manager.run_session_end()
        for r in results:
            if r.error:
                logger.warning(f"SessionEnd hook error: {r.error}")


def create_hook_callback(
    hook_config: HookConfig | None = None,
    working_dir: str | None = None,
    session_id: str | None = None,
    original_callback: Any = None,
) -> tuple[HookEventProcessor, Any]:
    """Create a hook-enabled event callback. Returns (processor, callback)."""
    hook_manager = HookManager(
        config=hook_config,
        working_dir=working_dir,
        session_id=session_id,
    )

    processor = HookEventProcessor(
        hook_manager=hook_manager,
        original_callback=original_callback,
    )

    return processor, processor.on_event
