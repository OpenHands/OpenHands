"""Invoke the Canvas FinishTool factory — not source-text inspection.

Run with the pinned SDK, for example:

    uv run --with openhands-sdk==<versions.agentServer> python \\
        __tests__/tools/test_canvas_ui_tool.py
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


class _SampleOutcome(BaseModel):
    success: bool = Field(description="Whether the preset task succeeded.")


class RemoteConversationFinishToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Import after sys.path is set so the agent-server startup module
        # registers _RemoteConversationFinishTool as FinishTool.
        cls.canvas_ui_tool = importlib.import_module("canvas_ui_tool")

    def setUp(self) -> None:
        from openhands.sdk.tool import FinishTool, list_registered_tools, register_tool

        # Each test starts from the factory registration the module performs
        # at import, even if a prior test re-registered something else.
        register_tool(FinishTool.__name__, self.canvas_ui_tool._RemoteConversationFinishTool)
        self.assertIn(FinishTool.__name__, list_registered_tools())

    def test_create_with_response_schema_returns_finish_tools_with_schema(self) -> None:
        from openhands.sdk.tool import FinishTool

        tools = self.canvas_ui_tool._RemoteConversationFinishTool.create(
            response_schema=_SampleOutcome,
        )

        self.assertGreaterEqual(len(tools), 1)
        self.assertTrue(all(tool is not None for tool in tools))
        self.assertTrue(all(isinstance(tool, FinishTool) for tool in tools))
        self.assertIs(_SampleOutcome, tools[0].response_schema)

    def test_create_without_schema_returns_valid_finish_tools(self) -> None:
        from openhands.sdk.tool import FinishTool

        tools = self.canvas_ui_tool._RemoteConversationFinishTool.create()

        self.assertGreaterEqual(len(tools), 1)
        self.assertTrue(all(tool is not None for tool in tools))
        self.assertTrue(all(isinstance(tool, FinishTool) for tool in tools))
        self.assertIsNone(tools[0].response_schema)

    def test_create_applies_schema_when_set_response_schema_returns_none(self) -> None:
        from openhands.sdk.tool import FinishTool

        def mutate_and_return_none(self, response_schema):
            object.__setattr__(self, "response_schema", response_schema)
            return None

        with patch.object(FinishTool, "set_response_schema", mutate_and_return_none):
            tools = self.canvas_ui_tool._RemoteConversationFinishTool.create(
                response_schema=_SampleOutcome,
            )

        self.assertGreaterEqual(len(tools), 1)
        self.assertTrue(all(tool is not None for tool in tools))
        self.assertTrue(all(isinstance(tool, FinishTool) for tool in tools))
        self.assertIs(_SampleOutcome, tools[0].response_schema)

    def test_create_rejects_leftover_params_other_than_response_schema(self) -> None:
        with self.assertRaises(ValueError):
            self.canvas_ui_tool._RemoteConversationFinishTool.create(
                response_schema=_SampleOutcome,
                unexpected_param="not-a-preset-field",
            )

    def test_reregister_does_not_raise_and_plain_finish_tool_still_resolves(
        self,
    ) -> None:
        from openhands.sdk.tool import (
            FinishTool,
            Tool,
            list_registered_tools,
            register_tool,
            resolve_tool,
        )

        # Safe overwrite: registering the factory twice must not raise.
        register_tool(
            FinishTool.__name__,
            self.canvas_ui_tool._RemoteConversationFinishTool,
        )
        register_tool(
            FinishTool.__name__,
            self.canvas_ui_tool._RemoteConversationFinishTool,
        )
        self.assertIn(FinishTool.__name__, list_registered_tools())

        # Builtin FinishTool.create() (no leftover params) still works after
        # the factory occupies the registry name.
        plain = FinishTool.create()
        self.assertGreaterEqual(len(plain), 1)
        self.assertTrue(all(isinstance(tool, FinishTool) for tool in plain))
        self.assertIsNone(plain[0].response_schema)

        resolved = resolve_tool(Tool(name=FinishTool.__name__), conv_state=None)
        self.assertGreaterEqual(len(resolved), 1)
        self.assertTrue(all(tool is not None for tool in resolved))
        self.assertTrue(all(isinstance(tool, FinishTool) for tool in resolved))
        self.assertIsNone(resolved[0].response_schema)


if __name__ == "__main__":
    unittest.main()
