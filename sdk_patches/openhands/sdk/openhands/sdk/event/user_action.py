from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


class PauseEvent(Event):
    """Event indicating that the agent execution was paused by user request."""

    source: SourceType = "user"

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this pause event."""
        content = Text()
        content.append("Conversation Paused", style="bold")
        return content

    def __str__(self) -> str:
        """Plain text string representation for PauseEvent."""
        return f"{self.__class__.__name__} ({self.source}): Agent execution paused"
