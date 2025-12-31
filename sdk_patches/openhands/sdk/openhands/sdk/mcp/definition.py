"""MCPTool definition and implementation."""

import json
from typing import Any

import mcp.types
from pydantic import Field
from rich.text import Text

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import (
    Observation,
)
from openhands.sdk.tool.schema import Action
from openhands.sdk.utils.visualize import display_json


logger = get_logger(__name__)


# NOTE: We don't define MCPToolAction because it
# will be dynamically created from the MCP tool schema.


class MCPToolAction(Action):
    """Schema for MCP input action.

    It is just a thin wrapper around raw JSON and does
    not do any validation.

    Validation will be performed by MCPTool.__call__
    by constructing dynamically created Pydantic model
    from the MCP tool input schema.
    """

    data: dict[str, Any] = Field(
        default_factory=dict, description="Dynamic data fields from the tool call"
    )

    def to_mcp_arguments(self) -> dict:
        """Return the data field as MCP tool call arguments.

        This is used to convert this action to MCP tool call arguments.
        The data field contains the dynamic fields from the tool call.
        """
        return self.data


class MCPToolObservation(Observation):
    """Observation from MCP tool execution."""

    tool_name: str = Field(description="Name of the tool that was called")

    @classmethod
    def from_call_tool_result(
        cls, tool_name: str, result: mcp.types.CallToolResult
    ) -> "MCPToolObservation":
        """Create an MCPToolObservation from a CallToolResult."""

        native_content: list[mcp.types.ContentBlock] = result.content
        content: list[TextContent | ImageContent] = [
            TextContent(text=f"[Tool '{tool_name}' executed.]")
        ]
        for block in native_content:
            if isinstance(block, mcp.types.TextContent):
                content.append(TextContent(text=block.text))
            elif isinstance(block, mcp.types.ImageContent):
                content.append(
                    ImageContent(
                        image_urls=[f"data:{block.mimeType};base64,{block.data}"],
                    )
                )
            else:
                logger.warning(
                    f"Unsupported MCP content block type: {type(block)}. Ignoring."
                )

        return cls(
            content=content,
            is_error=result.isError,
            tool_name=tool_name,
        )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this observation."""
        text = Text()

        if self.is_error:
            text.append("❌ ", style="red bold")
            text.append(self.ERROR_MESSAGE_HEADER, style="bold red")

        text.append(f"[MCP Tool '{self.tool_name}' Observation]\n", style="bold")
        for block in self.content:
            if isinstance(block, TextContent):
                # try to see if block.text is a JSON
                try:
                    parsed = json.loads(block.text)
                    text.append(display_json(parsed))
                    continue
                except (json.JSONDecodeError, TypeError):
                    text.append(block.text + "\n")
            elif isinstance(block, ImageContent):
                text.append(f"[Image with {len(block.image_urls)} URLs]\n")
        return text
