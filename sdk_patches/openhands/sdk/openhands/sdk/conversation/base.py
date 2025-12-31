from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.conversation.events_list_base import EventsListBase
from openhands.sdk.conversation.secret_registry import SecretValue
from openhands.sdk.conversation.types import (
    ConversationCallbackType,
    ConversationID,
    ConversationTokenCallbackType,
)
from openhands.sdk.llm.llm import LLM
from openhands.sdk.llm.message import Message
from openhands.sdk.observability.laminar import (
    end_active_span,
    should_enable_observability,
    start_active_span,
)
from openhands.sdk.security.analyzer import SecurityAnalyzerBase
from openhands.sdk.security.confirmation_policy import (
    ConfirmationPolicyBase,
    NeverConfirm,
)
from openhands.sdk.workspace.base import BaseWorkspace


if TYPE_CHECKING:
    from openhands.sdk.agent.base import AgentBase
    from openhands.sdk.conversation.state import ConversationExecutionStatus


CallbackType = TypeVar(
    "CallbackType",
    ConversationCallbackType,
    ConversationTokenCallbackType,
)


class ConversationStateProtocol(Protocol):
    """Protocol defining the interface for conversation state objects."""

    @property
    def id(self) -> ConversationID:
        """The conversation ID."""
        ...

    @property
    def events(self) -> EventsListBase:
        """Access to the events list."""
        ...

    @property
    def execution_status(self) -> "ConversationExecutionStatus":
        """The current conversation execution status."""
        ...

    @property
    def confirmation_policy(self) -> ConfirmationPolicyBase:
        """The confirmation policy."""
        ...

    @property
    def security_analyzer(self) -> SecurityAnalyzerBase | None:
        """The security analyzer."""
        ...

    @property
    def activated_knowledge_skills(self) -> list[str]:
        """List of activated knowledge skills."""
        ...

    @property
    def workspace(self) -> BaseWorkspace:
        """The workspace for agent operations and tool execution."""
        ...

    @property
    def persistence_dir(self) -> str | None:
        """The persistence directory from the FileStore.

        If None, it means the conversation is not being persisted.
        """
        ...

    @property
    def agent(self) -> "AgentBase":
        """The agent running in the conversation."""
        ...

    @property
    def stats(self) -> ConversationStats:
        """The conversation statistics."""
        ...


class BaseConversation(ABC):
    """Abstract base class for conversation implementations.

    This class defines the interface that all conversation implementations must follow.
    Conversations manage the interaction between users and agents, handling message
    exchange, execution control, and state management.
    """

    def __init__(self) -> None:
        """Initialize the base conversation with span tracking."""
        self._span_ended = False

    def _start_observability_span(self, session_id: str) -> None:
        """Start an observability span if observability is enabled.

        Args:
            session_id: The session ID to associate with the span
        """
        if should_enable_observability():
            start_active_span("conversation", session_id=session_id)

    def _end_observability_span(self) -> None:
        """End the observability span if it hasn't been ended already."""
        if not self._span_ended and should_enable_observability():
            end_active_span()
            self._span_ended = True

    @property
    @abstractmethod
    def id(self) -> ConversationID: ...

    @property
    @abstractmethod
    def state(self) -> ConversationStateProtocol: ...

    @property
    @abstractmethod
    def conversation_stats(self) -> ConversationStats: ...

    @abstractmethod
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
        ...

    @abstractmethod
    def run(self) -> None:
        """Execute the agent to process messages and perform actions.

        This method runs the agent until it finishes processing the current
        message or reaches the maximum iteration limit.
        """
        ...

    @abstractmethod
    def set_confirmation_policy(self, policy: ConfirmationPolicyBase) -> None:
        """Set the confirmation policy for the conversation."""
        ...

    @property
    def confirmation_policy_active(self) -> bool:
        return not isinstance(self.state.confirmation_policy, NeverConfirm)

    @property
    def is_confirmation_mode_active(self) -> bool:
        """Check if confirmation mode is active.

        Returns True if BOTH conditions are met:
        1. The conversation state has a security analyzer set (not None)
        2. The confirmation policy is active

        """
        return (
            self.state.security_analyzer is not None and self.confirmation_policy_active
        )

    @abstractmethod
    def reject_pending_actions(
        self, reason: str = "User rejected the action"
    ) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def update_secrets(self, secrets: Mapping[str, SecretValue]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def generate_title(self, llm: LLM | None = None, max_length: int = 50) -> str:
        """Generate a title for the conversation based on the first user message.

        Args:
            llm: Optional LLM to use for title generation. If not provided,
                 uses the agent's LLM.
            max_length: Maximum length of the generated title.

        Returns:
            A generated title for the conversation.

        Raises:
            ValueError: If no user messages are found in the conversation.
        """
        ...

    @staticmethod
    def get_persistence_dir(
        persistence_base_dir: str | Path, conversation_id: ConversationID
    ) -> str:
        """Get the persistence directory for the conversation.

        Args:
            persistence_base_dir: Base directory for persistence. Can be a string
                path or Path object.
            conversation_id: Unique conversation ID.

        Returns:
            String path to the conversation-specific persistence directory.
            Always returns a normalized string path even if a Path was provided.
        """
        return str(Path(persistence_base_dir) / conversation_id.hex)

    @abstractmethod
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
        ...

    @abstractmethod
    def condense(self) -> None:
        """Force condensation of the conversation history.

        This method uses the existing condensation request pattern to trigger
        condensation. It adds a CondensationRequest event to the conversation
        and forces the agent to take a single step to process it.

        The condensation will be applied immediately and will modify the conversation
        state by adding a condensation event to the history.

        Raises:
            ValueError: If no condenser is configured or the condenser doesn't
                       handle condensation requests.
        """
        ...

    @staticmethod
    def compose_callbacks(callbacks: Iterable[CallbackType]) -> CallbackType:
        """Compose multiple callbacks into a single callback function.

        Args:
            callbacks: An iterable of callback functions

        Returns:
            A single callback function that calls all provided callbacks
        """

        def composed(event) -> None:
            for cb in callbacks:
                if cb:
                    cb(event)

        return cast(CallbackType, composed)
