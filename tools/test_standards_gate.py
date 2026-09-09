"""End-to-end tests for standards prompt injection and the commit-loop gate."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from standards import STANDARDS_REQUIREMENTS_OPEN  # noqa: E402
from standards.audit_store import AuditStore  # noqa: E402
from standards.plugins.cyber_essentials_plugin import CE_PLUGIN_NAME  # noqa: E402
from standards.plugins.iso27001_plugin import ISO_PLUGIN_NAME, ISO_SECRET_AWS  # noqa: E402
from standards.registry import (  # noqa: E402
    StandardsRegistry,
    set_active_registry,
)
from standards_agent_hooks import (  # noqa: E402
    apply_dispatch_standards_context,
    compose_standards_block,
)
from standards_gate import handle_standards_stage  # noqa: E402


def _registry(test: unittest.TestCase, root: str | None = None) -> StandardsRegistry:
    store = AuditStore(":memory:")
    test.addCleanup(store.close)
    registry = StandardsRegistry(
        audit=store,
        user_dir=os.path.join(tempfile.mkdtemp(), "missing"),
        project_root=root,
    )
    registry.discover(root)
    set_active_registry(registry)
    test.addCleanup(lambda: set_active_registry(None))
    return registry


def _write(root: str, rel: str, body: str) -> None:
    path = Path(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


class StandardsPromptTests(unittest.TestCase):
    def test_composes_block_only_when_enabled_and_prompt_on(self) -> None:
        root = tempfile.mkdtemp()
        registry = _registry(self, root)
        spec = {"spec_text": "Implement the card.", "root": root}
        empty = apply_dispatch_standards_context(spec, registry=registry)
        self.assertIsNone(empty.get("standards_block"))
        self.assertNotIn(STANDARDS_REQUIREMENTS_OPEN, empty.get("spec_text") or "")

        registry.save_config(
            {
                "enforcement": {"prompt": False, "automated": True, "gates": True},
                "plugins": [{"name": ISO_PLUGIN_NAME, "enabled": True, "action": "warn"}],
            },
            root,
        )
        off = apply_dispatch_standards_context(spec, registry=registry)
        self.assertFalse(off.get("standards_block"))

        registry.save_config(
            {
                "enforcement": {"prompt": True, "automated": True, "gates": True},
                "plugins": [{"name": ISO_PLUGIN_NAME, "enabled": True, "action": "warn"}],
            },
            root,
        )
        on = apply_dispatch_standards_context(spec, registry=registry)
        self.assertIn(STANDARDS_REQUIREMENTS_OPEN, on["spec_text"])
        self.assertIn("Implement the card.", on["spec_text"])
        self.assertIn(ISO_PLUGIN_NAME, compose_standards_block(registry, project_root=root))


class StandardsGateTests(unittest.TestCase):
    def test_block_error_fails_and_warn_passes(self) -> None:
        root = tempfile.mkdtemp()
        _write(root, "secrets.py", 'key = "AKIAIOSFODNN7EXAMPLE"\n')
        registry = _registry(self, root)
        registry.save_config(
            {
                "enforcement": {"prompt": True, "automated": True, "gates": True},
                "plugins": [
                    {"name": ISO_PLUGIN_NAME, "enabled": True, "action": "block"}
                ],
            },
            root,
        )
        passed, output = handle_standards_stage("run", {"name": "standards"}, root, {})
        self.assertFalse(passed)
        self.assertIn(ISO_SECRET_AWS, output)
        audit = registry.audit.list_violations()
        self.assertGreaterEqual(len(audit["items"]), 1)

        registry.save_config(
            {"plugins": [{"name": ISO_PLUGIN_NAME, "enabled": True, "action": "warn"}]},
            root,
        )
        passed, _ = handle_standards_stage("run", {"name": "standards"}, root, {})
        self.assertTrue(passed)

    def test_auto_fix_debug_flag(self) -> None:
        root = tempfile.mkdtemp()
        _write(root, "settings.py", "DEBUG = True\n")
        registry = _registry(self, root)
        registry.save_config(
            {
                "enforcement": {"prompt": True, "automated": True, "gates": True},
                "plugins": [
                    {"name": CE_PLUGIN_NAME, "enabled": True, "action": "block"}
                ],
            },
            root,
        )
        passed, _output = handle_standards_stage("run", {"name": "standards"}, root, {})
        self.assertTrue(passed)
        self.assertEqual(
            Path(root, "settings.py").read_text(encoding="utf-8"),
            "DEBUG = False\n",
        )

    def test_disabled_plugin_and_enforcement_off_write_no_audit(self) -> None:
        root = tempfile.mkdtemp()
        _write(root, "secrets.py", 'key = "AKIAIOSFODNN7EXAMPLE"\n')
        registry = _registry(self, root)
        passed, output = handle_standards_stage("run", {"name": "standards"}, root, {})
        self.assertTrue(passed)
        self.assertIn("no enabled", output)
        self.assertEqual(registry.audit.list_violations()["items"], [])

        registry.save_config(
            {
                "enforcement": {"prompt": True, "automated": False, "gates": True},
                "plugins": [
                    {"name": ISO_PLUGIN_NAME, "enabled": True, "action": "block"}
                ],
            },
            root,
        )
        passed, output = handle_standards_stage("run", {"name": "standards"}, root, {})
        self.assertEqual(passed, "skipped")
        self.assertEqual(registry.audit.list_violations()["items"], [])


if __name__ == "__main__":
    unittest.main()
