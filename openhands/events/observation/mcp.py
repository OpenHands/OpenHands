from dataclasses import dataclass, field
from typing import Any

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class MCPImage:
    """An image returned from an MCP tool call."""

    data: str  # base64-encoded image data
    mime_type: str = 'image/png'

    def to_data_uri(self) -> str:
        """Return as a data URI for OpenAI-style image_url."""
        return f'data:{self.mime_type};base64,{self.data}'


@dataclass
class MCPObservation(Observation):
    """This data class represents the result of a MCP Server operation.

    Attributes:
        content: Text content from the MCP result
        name: The name of the MCP tool that was called
        arguments: The arguments passed to the MCP tool
        images: List of images returned by the tool (e.g., screenshots)
        is_error: Whether the MCP call resulted in an error
    """

    observation: str = ObservationType.MCP
    name: str = ''  # The name of the MCP tool that was called
    arguments: dict[str, Any] = field(
        default_factory=dict
    )  # The arguments passed to the MCP tool
    images: list[MCPImage] = field(default_factory=list)  # Images from the result
    is_error: bool = False

    @property
    def message(self) -> str:
        if self.images:
            return f'{self.content} [+{len(self.images)} image(s)]'
        return self.content

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0
