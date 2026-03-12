# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from typing import Any, Iterable

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


@dataclass
class LLM:
    vendor: str
    model: str


class Event(BaseModel):
    metadata: dict[str, Any] | None = Field(
        default_factory=lambda: dict(), description='Metadata associated with the event'
    )


class Function(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCall(Event):
    id: str
    type: str
    function: Function


class Message(Event):
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None

    def __rich_repr__(  # type: ignore[override]
        self,
    ) -> Iterable[Any | tuple[Any] | tuple[str, Any] | tuple[str, Any, Any]]:
        # Print on separate line
        yield 'role', self.role
        yield 'content', self.content
        yield 'tool_calls', self.tool_calls


class ToolOutput(Event):
    role: str
    content: str
    tool_call_id: str | None = None

    _tool_call: ToolCall | None = None
