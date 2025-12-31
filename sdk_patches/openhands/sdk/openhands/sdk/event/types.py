from typing import Literal


EventType = Literal["action", "observation", "message", "system_prompt", "agent_error"]
SourceType = Literal["agent", "user", "environment"]

EventID = str
"""Type alias for event IDs."""

ToolCallID = str
"""Type alias for tool call IDs."""
