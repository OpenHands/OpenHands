"""Event for streaming LLM completion logs from remote agents to clients."""

from pydantic import Field

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


class LLMCompletionLogEvent(Event):
    """Event containing LLM completion log data.

    When an LLM is configured with log_completions=True in a remote conversation,
    this event streams the completion log data back to the client through WebSocket
    instead of writing it to a file inside the Docker container.
    """

    source: SourceType = "environment"
    filename: str = Field(
        ...,
        description="The intended filename for this log (relative to log directory)",
    )
    log_data: str = Field(
        ...,
        description="The JSON-encoded log data to be written to the file",
    )
    model_name: str = Field(
        default="unknown",
        description="The model name for context",
    )
    usage_id: str = Field(
        default="default",
        description="The LLM usage_id that produced this log",
    )

    def __str__(self) -> str:
        return (
            f"LLMCompletionLog(usage_id={self.usage_id}, model={self.model_name}, "
            f"file={self.filename})"
        )
