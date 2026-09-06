"""Resolve → adapter dispatch integration tests.

Struggle escalation, resume prompts, and loop-engine hooks land in PR #25.
This module exists so the Phase 7 verification command can collect it.
"""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from model_registry import ModelRegistry  # noqa: E402
from provider_adapters import adapter_for, dispatch  # noqa: E402
from router import RouterStore  # noqa: E402


class ResolveDispatchTests(unittest.TestCase):
    def test_resolve_then_mock_adapter_dispatch(self) -> None:
        store = RouterStore(":memory:", registry=ModelRegistry())
        self.addCleanup(store.close)
        result = store.resolve(
            {
                "task_text": "implement a parser",
                "work_type": "coding",
                "sensitivity": "default",
                "connected_providers": ["cursor-cli", "openhands", "anthropic"],
                "local_runtimes": {},
            }
        )
        decision = result["decision"]
        self.assertTrue(decision["usable"])
        adapter = adapter_for(decision["provider_key"])
        if adapter is None:
            self.skipTest(f"no CLI adapter for {decision['provider_key']}")
        adapter.find_binary = lambda: "/bin/fake"  # type: ignore[method-assign]
        captured: list[list[str]] = []

        def runner(cmd: list[str], cwd: str | None, timeout_s: float) -> dict[str, object]:
            captured.append(cmd)
            return {"ok": True, "stdout": "ok", "stderr": "", "cmd": cmd}

        output = dispatch(
            decision["provider_key"],
            "implement a parser",
            decision["model"],
            runner=runner,
        )
        self.assertTrue(output["ok"])
        self.assertTrue(captured)


if __name__ == "__main__":
    unittest.main()
