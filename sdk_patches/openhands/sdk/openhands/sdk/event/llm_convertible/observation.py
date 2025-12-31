from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import N_CHAR_PREVIEW, LLMConvertibleEvent
from openhands.sdk.event.types import EventID, SourceType, ToolCallID
from openhands.sdk.llm import Message, TextContent, content_to_str
from openhands.sdk.tool.schema import Observation


class ObservationBaseEvent(LLMConvertibleEvent):
    """Base class for anything as a response to a tool call.

    Examples include tool execution, error, user reject.
    """

    source: SourceType = "environment"
    tool_name: str = Field(
        ..., description="The tool name that this observation is responding to"
    )
    tool_call_id: ToolCallID = Field(
        ..., description="The tool call id that this observation is responding to"
    )


class ObservationEvent(ObservationBaseEvent):
    observation: Observation = Field(
        ..., description="The observation (tool call) sent to LLM"
    )
    action_id: EventID = Field(
        ..., description="The action id that this observation is responding to"
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this observation event."""
        to_viz = self.observation.visualize
        content = Text()
        if to_viz.plain.strip():
            content.append("Tool: ", style="bold")
            content.append(self.tool_name)
            content.append("\nResult:\n", style="bold")
            content.append(to_viz)
        return content

    def to_llm_message(self) -> Message:
        return Message(
            role="tool",
            content=self.observation.to_llm_content,
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for ObservationEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        content_str = "".join(content_to_str(self.observation.to_llm_content))
        obs_preview = (
            content_str[:N_CHAR_PREVIEW] + "..."
            if len(content_str) > N_CHAR_PREVIEW
            else content_str
        )
        return f"{base_str}\n  Tool: {self.tool_name}\n  Result: {obs_preview}"


class UserRejectObservation(ObservationBaseEvent):
    """Observation when user rejects an action in confirmation mode."""

    rejection_reason: str = Field(
        default="User rejected the action",
        description="Reason for rejecting the action",
    )
    action_id: EventID = Field(
        ..., description="The action id that this observation is responding to"
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this user rejection event."""
        content = Text()
        content.append("Tool: ", style="bold")
        content.append(self.tool_name)
        content.append("\n\nRejection Reason:\n", style="bold")
        content.append(self.rejection_reason)
        return content

    def to_llm_message(self) -> Message:
        return Message(
            role="tool",
            content=[TextContent(text=f"Action rejected: {self.rejection_reason}")],
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for UserRejectObservation."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        reason_preview = (
            self.rejection_reason[:N_CHAR_PREVIEW] + "..."
            if len(self.rejection_reason) > N_CHAR_PREVIEW
            else self.rejection_reason
        )
        return f"{base_str}\n  Tool: {self.tool_name}\n  Reason: {reason_preview}"


class AgentErrorEvent(ObservationBaseEvent):
    """Error triggered by the agent.

    Note: This event should not contain model "thought" or "reasoning_content". It
    represents an error produced by the agent/scaffold, not model output.
    """

    source: SourceType = "agent"
    error: str = Field(..., description="The error message from the scaffold")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this agent error event."""
        content = Text()
        content.append("Error Details:\n", style="bold")
        content.append(self.error)
        return content

    def to_llm_message(self) -> Message:
        # Provide plain string error content; serializers handle Chat vs Responses.
        # For Responses API, output is a string; JSON is not required.
        return Message(
            role="tool",
            content=[TextContent(text=self.error)],
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for AgentErrorEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        error_preview = (
            self.error[:N_CHAR_PREVIEW] + "..."
            if len(self.error) > N_CHAR_PREVIEW
            else self.error
        )
        return f"{base_str}\n  Error: {error_preview}"
