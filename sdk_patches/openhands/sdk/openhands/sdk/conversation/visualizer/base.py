from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

from openhands.sdk.event.base import Event


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import ConversationStateProtocol
    from openhands.sdk.conversation.conversation_stats import ConversationStats


class ConversationVisualizerBase(ABC):
    """Base class for conversation visualizers.

    This abstract base class defines the interface that all conversation visualizers
    must implement. Visualizers can be created before the Conversation is initialized
    and will be configured with the conversation state automatically.

    The typical usage pattern:
    1. Create a visualizer instance:
       `viz = MyVisualizer()`
    2. Pass it to Conversation: `conv = Conversation(agent, visualizer=viz)`
    3. Conversation automatically calls `viz.initialize(state)` to attach the state

    You can also pass the uninstantiated class if you don't need extra args
        for initialization, and Conversation will create it:
         `conv = Conversation(agent, visualizer=MyVisualizer)`
    Conversation will then calls `MyVisualizer()` followed by `initialize(state)`
    """

    _state: "ConversationStateProtocol | None"

    def __init__(self):
        """Initialize the visualizer base."""
        self._state = None

    @final
    def initialize(self, state: "ConversationStateProtocol") -> None:
        """Initialize the visualizer with conversation state.

        This method is called by Conversation after the state is created,
        allowing the visualizer to access conversation stats and other
        state information.

        Subclasses should not override this method, to ensure the state is set.

        Args:
            state: The conversation state object
        """
        self._state = state

    @property
    def conversation_stats(self) -> "ConversationStats | None":
        """Get conversation stats from the state."""
        return self._state.stats if self._state else None

    @abstractmethod
    def on_event(self, event: Event) -> None:
        """Handle a conversation event.

        This method is called for each event in the conversation and should
        implement the visualization logic.

        Args:
            event: The event to visualize
        """
        pass
