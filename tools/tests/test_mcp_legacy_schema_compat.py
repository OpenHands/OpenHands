"""Regression coverage for OpenHands/OpenHands#17615.

The agent-server resolves ``mcp`` 1.x (via the ``fastmcp<4`` pin) while
conversations created during the earlier 2.x window persisted snake_case MCP
tool fields. ``mcp_legacy_schema_compat`` is imported at agent-server startup so
the 1.x reader accepts either spelling.

Under mcp 2.x the reader already accepts both spellings and the module is a
no-op, so the reader cases are skipped there; the helper cases still run.
"""

import importlib.metadata
import json
import subprocess
import sys

import pytest

# Resolve the mcp major version before importing the compat module, which
# patches mcp.types as an import side effect.
_MCP_MAJOR = int(importlib.metadata.version("mcp").split(".")[0])

import mcp.types as mcp_types  # noqa: E402
from pydantic import AliasChoices  # noqa: E402

from mcp_legacy_schema_compat import (  # noqa: E402
    _field_variants,
    _to_camel_case,
    _to_snake_case,
    apply_mcp_field_compatibility,
)

requires_strict_mcp = pytest.mark.skipif(
    _MCP_MAJOR >= 2,
    reason="mcp 2.x already reads both field spellings",
)


def _sdk_installed() -> bool:
    try:
        import openhands.sdk.mcp.tool  # noqa: F401
    except Exception:
        return False
    return True


requires_sdk = pytest.mark.skipif(
    not _sdk_installed(), reason="openhands-sdk is not installed"
)


def test_to_snake_case_matches_python_convention():
    assert _to_snake_case("inputSchema") == "input_schema"
    assert _to_snake_case("isError") == "is_error"
    assert _to_snake_case("mimeType") == "mime_type"


def test_to_camel_case_matches_wire_convention():
    assert _to_camel_case("input_schema") == "inputSchema"
    assert _to_camel_case("mime_type") == "mimeType"


def test_field_variants_covers_both_spellings():
    assert _field_variants("inputSchema", None) == {"inputSchema", "input_schema"}
    assert _field_variants("input_schema", None) == {"input_schema", "inputSchema"}


def test_field_variants_preserves_existing_alias():
    assert _field_variants("meta", AliasChoices("_meta")) == {"meta", "_meta"}


@requires_strict_mcp
def test_unpatched_strict_reader_rejects_snake_case():
    # Documents the failure #17615 reports. Run in a subprocess so this module's
    # import-time patch has not already made the field readable.
    probe = (
        "import mcp.types as t;"
        "t.Tool.model_validate({'name':'p','input_schema':{'type':'object'}})"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "inputSchema" in result.stderr


@requires_strict_mcp
def test_import_applies_patch_accepting_snake_case_tool_payload():
    # The module patches mcp.types on import, which is the startup contract.
    tool = mcp_types.Tool.model_validate(
        {
            "name": "github_search",
            "description": "search",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }
    )
    assert tool.inputSchema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }


@requires_strict_mcp
def test_patch_still_accepts_camel_case_tool_payload():
    tool = mcp_types.Tool.model_validate(
        {"name": "probe", "inputSchema": {"type": "object"}}
    )
    assert tool.inputSchema == {"type": "object"}


@requires_strict_mcp
def test_snake_case_call_tool_result_is_readable():
    # isError/is_error is the other renamed pair persisted events carry.
    result = mcp_types.CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "boom"}], "is_error": True}
    )
    assert result.isError is True


@requires_strict_mcp
def test_outgoing_spelling_is_unchanged():
    # Outgoing events must keep the camelCase spelling mcp 1.x wrote so upgraded
    # and downgraded readers stay interchangeable.
    dumped = mcp_types.Tool(name="probe", inputSchema={"type": "object"}).model_dump()
    assert "inputSchema" in dumped
    assert "input_schema" not in dumped


@requires_strict_mcp
def test_apply_is_idempotent_after_import():
    assert apply_mcp_field_compatibility() is False


@requires_strict_mcp
def test_python_attribute_access_is_preserved():
    # The SDK reads ``self.mcp_tool.inputSchema`` on the Python side.
    tool = mcp_types.Tool.model_validate(
        {"name": "probe", "input_schema": {"type": "object"}}
    )
    assert tool.inputSchema == {"type": "object"}


@requires_strict_mcp
def test_serialized_mcp2_event_shape_is_readable():
    tool = mcp_types.Tool.model_validate_json(
        '{"name": "github_search", "description": "search",'
        ' "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}'
    )
    assert tool.inputSchema["properties"]["q"]["type"] == "string"


@requires_sdk
def test_sdk_rehydrates_event_with_mcp2_tool_after_patch():
    """The reported regression: an event holding a 2.x MCP tool must load.

    ``mcp_legacy_schema_compat`` patches ``mcp.types`` first, then the SDK's
    already-imported models are recompiled, so ``MCPToolDefinition`` (and the
    ``SystemPromptEvent`` wrapping it) read the serialized 2.x spelling.
    """
    # Arrange — build a real event through the SDK, then rewrite the MCP tool
    # payload into the snake_case shape mcp 2.x would have persisted.
    from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
    from openhands.sdk.llm import TextContent
    from openhands.sdk.mcp.definition import MCPToolAction, MCPToolObservation
    from openhands.sdk.mcp.tool import MCPToolDefinition

    mcp_tool = mcp_types.Tool.model_validate(
        {
            "name": "github_search",
            "description": "search",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }
    )
    event = SystemPromptEvent(
        source="agent",
        system_prompt=TextContent(text="sys"),
        tools=[
            MCPToolDefinition(
                description="search",
                action_type=MCPToolAction,
                observation_type=MCPToolObservation,
                mcp_tool=mcp_tool,
            )
        ],
    )
    payload = json.loads(event.model_dump_json(exclude_none=True))
    mcp2_payload = json.loads(
        json.dumps(payload).replace('"inputSchema"', '"input_schema"')
    )
    assert "input_schema" in mcp2_payload["tools"][0]["mcp_tool"]

    # Act
    restored = SystemPromptEvent.model_validate(mcp2_payload)

    # Assert
    assert (
        restored.tools[0].mcp_tool.inputSchema["properties"]["q"]["type"] == "string"
    )
