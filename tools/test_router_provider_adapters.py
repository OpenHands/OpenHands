"""Provider adapter tests — no real CLI binaries required."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from provider_adapters import (  # noqa: E402
    AdapterError,
    ClaudeCodeAdapter,
    CursorAdapter,
    OpenCodeAdapter,
    adapter_for,
    dispatch,
    usable_providers,
)


class AdapterShapeTests(unittest.TestCase):
    def test_cursor_command_matches_cursor_tool(self) -> None:
        adapter = CursorAdapter()
        cmd = adapter.build_cmd("/bin/agent", "fix the bug", "composer")
        self.assertEqual(
            cmd,
            [
                "/bin/agent",
                "-p",
                "fix the bug",
                "--output-format",
                "json",
                "--force",
                "--model",
                "composer",
            ],
        )

    def test_claude_and_opencode_commands(self) -> None:
        claude = ClaudeCodeAdapter().build_cmd(
            "/bin/claude", "review this", "claude-sonnet-4-5"
        )
        self.assertEqual(
            claude,
            [
                "/bin/claude",
                "-p",
                "review this",
                "--output-format",
                "json",
                "--model",
                "claude-sonnet-4-5",
            ],
        )
        opencode = OpenCodeAdapter().build_cmd(
            "/bin/opencode", "implement", "anthropic/claude-sonnet-4-6"
        )
        self.assertEqual(
            opencode,
            [
                "/bin/opencode",
                "run",
                "--format",
                "json",
                "implement",
                "--model",
                "anthropic/claude-sonnet-4-6",
            ],
        )

    def test_missing_binary_is_unusable(self) -> None:
        adapter = CursorAdapter()
        adapter.find_binary = lambda: None  # type: ignore[method-assign]
        with self.assertRaises(AdapterError) as ctx:
            adapter.run("hello", "composer")
        self.assertTrue(ctx.exception.unusable)

    def test_dispatch_uses_injected_runner(self) -> None:
        captured: list[list[str]] = []
        adapter = adapter_for("cursor-cli")
        assert adapter is not None
        adapter.find_binary = lambda: "/bin/agent"  # type: ignore[method-assign]

        def runner(cmd: list[str], cwd: str | None, timeout_s: float) -> dict[str, object]:
            captured.append(cmd)
            return {"ok": True, "stdout": "done", "stderr": "", "cmd": cmd}

        result = dispatch(
            "cursor-cli",
            "do the thing",
            "composer",
            runner=runner,
        )
        self.assertTrue(result["ok"])
        self.assertIn("--model", captured[0])
        self.assertIn("composer", captured[0])

    def test_usable_providers_respects_finder(self) -> None:
        found = usable_providers(
            connected=["cursor-cli", "opencode"],
            finder=lambda key: key == "cursor-cli",
        )
        self.assertEqual(found, ["cursor-cli"])
        self.assertIsNotNone(adapter_for("anthropic"))


if __name__ == "__main__":
    unittest.main()
