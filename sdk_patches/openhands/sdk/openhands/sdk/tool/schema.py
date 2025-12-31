from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pydantic import ConfigDict, Field, create_model
from rich.text import Text

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.llm.message import content_to_str
from openhands.sdk.utils.models import (
    DiscriminatedUnionMixin,
)
from openhands.sdk.utils.visualize import display_dict


if TYPE_CHECKING:
    from typing import Self

S = TypeVar("S", bound="Schema")


def py_type(spec: dict[str, Any]) -> Any:
    """Map JSON schema types to Python types."""
    t = spec.get("type")

    # Normalize union types like ["string", "null"] to a single representative type.
    # MCP schemas often mark optional fields this way; we keep the non-null type.
    if isinstance(t, (list, tuple, set)):
        types = list(t)
        non_null = [tp for tp in types if tp != "null"]
        if len(non_null) == 1:
            t = non_null[0]
        else:
            return Any
    if t == "array":
        items = spec.get("items", {})
        inner = py_type(items) if isinstance(items, dict) else Any
        return list[inner]  # type: ignore[index]
    if t == "object":
        return dict[str, Any]
    _map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    if t in _map:
        return _map[t]
    return Any


def _process_schema_node(node, defs):
    """Recursively process a schema node to simplify and resolve $ref.

    https://www.reddit.com/r/mcp/comments/1kjo9gt/toolinputschema_conversion_from_pydanticmodel/
    https://gist.github.com/leandromoreira/3de4819e4e4df9422d87f1d3e7465c16
    """
    # Handle $ref references
    if "$ref" in node:
        ref_path = node["$ref"]
        if ref_path.startswith("#/$defs/"):
            ref_name = ref_path.split("/")[-1]
            if ref_name in defs:
                # Process the referenced definition
                return _process_schema_node(defs[ref_name], defs)

    # Start with a new schema object
    result = {}

    # Copy the basic properties
    if "type" in node:
        result["type"] = node["type"]

    # Handle anyOf (often used for optional fields with None)
    if "anyOf" in node:
        non_null_types = [t for t in node["anyOf"] if t.get("type") != "null"]
        if non_null_types:
            # Process the first non-null type
            processed = _process_schema_node(non_null_types[0], defs)
            result.update(processed)

    # Handle description
    if "description" in node:
        result["description"] = node["description"]

    # Handle object properties recursively
    if node.get("type") == "object" and "properties" in node:
        result["type"] = "object"
        result["properties"] = {}

        # Process each property
        for prop_name, prop_schema in node["properties"].items():
            result["properties"][prop_name] = _process_schema_node(prop_schema, defs)

        # Add required fields if present
        if "required" in node:
            result["required"] = node["required"]

    # Handle arrays
    if node.get("type") == "array" and "items" in node:
        result["type"] = "array"
        result["items"] = _process_schema_node(node["items"], defs)

    # Handle enum
    if "enum" in node:
        result["enum"] = node["enum"]

    return result


class Schema(DiscriminatedUnionMixin):
    """Base schema for input action / output observation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def to_mcp_schema(cls) -> dict[str, Any]:
        """Convert to JSON schema format compatible with MCP."""
        full_schema = cls.model_json_schema()
        # This will get rid of all "anyOf" in the schema,
        # so it is fully compatible with MCP tool schema
        result = _process_schema_node(full_schema, full_schema.get("$defs", {}))

        # Remove 'kind' from properties if present (discriminator field, not for LLM)
        EXCLUDE_FIELDS = DiscriminatedUnionMixin.model_fields.keys()
        for f in EXCLUDE_FIELDS:
            if "properties" in result and f in result["properties"]:
                result["properties"].pop(f)
                # Also remove from required if present
                if "required" in result and f in result["required"]:
                    result["required"].remove(f)

        return result

    @classmethod
    def from_mcp_schema(
        cls: type[S], model_name: str, schema: dict[str, Any]
    ) -> type["S"]:
        """Create a Schema subclass from an MCP/JSON Schema object.

        For non-required fields, we annotate as `T | None`
        so explicit nulls are allowed.
        """
        assert isinstance(schema, dict), "Schema must be a dict"
        assert schema.get("type") == "object", "Only object schemas are supported"

        props: dict[str, Any] = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        fields: dict[str, tuple] = {}
        for fname, spec in props.items():
            spec = spec if isinstance(spec, dict) else {}
            tp = py_type(spec)

            # Add description if present
            desc: str | None = spec.get("description")

            # Required → bare type, ellipsis sentinel
            # Optional → make nullable via `| None`, default None
            if fname in required:
                anno = tp
                default = ...
            else:
                anno = tp | None  # allow explicit null in addition to omission
                default = None

            fields[fname] = (
                anno,
                Field(default=default, description=desc)
                if desc
                else Field(default=default),
            )

        return create_model(model_name, __base__=cls, **fields)  # type: ignore[return-value]


class Action(Schema, ABC):
    """Base schema for input action."""

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this action.

        This method can be overridden by subclasses to customize visualization.
        The base implementation displays all action fields systematically.
        """
        content = Text()

        # Display action name
        action_name = self.__class__.__name__
        content.append("Action: ", style="bold")
        content.append(action_name)
        content.append("\n\n")

        # Display all action fields systematically
        content.append("Arguments:", style="bold")
        action_fields = self.model_dump()
        content.append(display_dict(action_fields))

        return content


class Observation(Schema, ABC):
    """Base schema for output observation."""

    ERROR_MESSAGE_HEADER: ClassVar[str] = "[An error occurred during execution.]\n"

    content: list[TextContent | ImageContent] = Field(
        default_factory=list,
        description=(
            "Content returned from the tool as a list of "
            "TextContent/ImageContent objects. "
            "When there is an error, it should be written in this field."
        ),
    )
    is_error: bool = Field(
        default=False, description="Whether the observation indicates an error"
    )

    @classmethod
    def from_text(
        cls,
        text: str,
        is_error: bool = False,
        **kwargs: Any,
    ) -> "Self":
        """Utility to create an Observation from a simple text string.

        Args:
            text: The text content to include in the observation.
            is_error: Whether this observation represents an error.
            **kwargs: Additional fields for the observation subclass.

        Returns:
            An Observation instance with the text wrapped in a TextContent.
        """
        return cls(content=[TextContent(text=text)], is_error=is_error, **kwargs)

    @property
    def text(self) -> str:
        """Extract all text content from the observation.

        Returns:
            Concatenated text from all TextContent items in content.
        """
        return "".join(
            item.text for item in self.content if isinstance(item, TextContent)
        )

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """
        Default content formatting for converting observation to LLM readable content.
        Subclasses can override to provide richer content (e.g., images, diffs).
        """
        llm_content: list[TextContent | ImageContent] = []

        # If is_error is true, prepend error message
        if self.is_error:
            llm_content.append(TextContent(text=self.ERROR_MESSAGE_HEADER))

        # Add content (now always a list)
        llm_content.extend(self.content)

        return llm_content

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this observation.

        Subclasses can override for custom visualization; by default we show the
        same text that would be sent to the LLM.
        """
        text = Text()

        if self.is_error:
            text.append("❌ ", style="red bold")
            text.append(self.ERROR_MESSAGE_HEADER, style="bold red")

        text_parts = content_to_str(self.to_llm_content)
        if text_parts:
            full_content = "".join(text_parts)
            text.append(full_content)
        else:
            text.append("[no text content]")
        return text
