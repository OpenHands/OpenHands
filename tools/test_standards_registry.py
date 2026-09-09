"""Unit tests for the standards plugin registry.

Run from the repo root:

    python3 -m unittest tools.test_standards_registry
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from standards.audit_store import AuditStore  # noqa: E402
from standards.plugins.demo_plugin import DEMO_PLUGIN_NAME, DEMO_RULE_TAB_INDENT  # noqa: E402
from standards.registry import StandardsRegistry  # noqa: E402


BROKEN_PLUGIN = """
raise RuntimeError("boom")
"""

OVERRIDE_PLUGIN = """
from standards import ACTION_WARN, SEVERITY_ERROR, StandardsPlugin, Violation

class OverridePlugin(StandardsPlugin):
    name = "demo"
    display_name = "Override demo"
    description = "Project override"
    version = "9.0.0"
    severity_default = SEVERITY_ERROR

    def check(self, file, content):
        return []

PLUGIN = OverridePlugin()
"""


def _registry(
    test: unittest.TestCase,
    *,
    user_dir: str | None = None,
    project_root: str | None = None,
) -> StandardsRegistry:
    store = AuditStore(":memory:")
    test.addCleanup(store.close)
    registry = StandardsRegistry(
        audit=store,
        user_dir=user_dir if user_dir is not None else os.path.join(tempfile.mkdtemp(), "missing"),
        project_root=project_root,
    )
    return registry


def _worktree_with_tab() -> str:
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "src"))
    os.makedirs(os.path.join(root, "node_modules"))
    with open(os.path.join(root, "src", "app.py"), "w", encoding="utf-8") as handle:
        handle.write("def main():\n\treturn 1\n")
    with open(os.path.join(root, "node_modules", "lib.py"), "w", encoding="utf-8") as handle:
        handle.write("\tignored\n")
    with open(os.path.join(root, "binary.bin"), "wb") as handle:
        handle.write(b"\x00\xff")
    return root


class StandardsRegistryTests(unittest.TestCase):
    def test_discovers_builtin_demo_plugin(self) -> None:
        registry = _registry(self)
        registry.discover()
        plugin = registry.get(DEMO_PLUGIN_NAME)
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.version, "1.0.0")
        self.assertFalse(registry.load_errors())

    def test_isolates_broken_third_party_module(self) -> None:
        user_dir = tempfile.mkdtemp()
        with open(os.path.join(user_dir, "broken.py"), "w", encoding="utf-8") as handle:
            handle.write(BROKEN_PLUGIN)
        registry = _registry(self, user_dir=user_dir)
        registry.discover()
        self.assertIsNotNone(registry.get(DEMO_PLUGIN_NAME))
        self.assertEqual(len(registry.load_errors()), 1)
        self.assertIn("boom", registry.load_errors()[0].message)

    def test_later_source_overrides_earlier(self) -> None:
        root = tempfile.mkdtemp()
        plugin_dir = os.path.join(root, ".openhands", "standards")
        os.makedirs(plugin_dir)
        with open(os.path.join(plugin_dir, "demo_plugin.py"), "w", encoding="utf-8") as handle:
            handle.write(OVERRIDE_PLUGIN)
        registry = _registry(self, project_root=root)
        registry.discover(root)
        plugin = registry.get(DEMO_PLUGIN_NAME)
        self.assertEqual(plugin.display_name, "Override demo")
        self.assertEqual(plugin.version, "9.0.0")

    def test_config_merge_persisted_over_project_yaml(self) -> None:
        root = tempfile.mkdtemp()
        oh = os.path.join(root, ".openhands")
        os.makedirs(oh)
        with open(os.path.join(oh, "project.yaml"), "w", encoding="utf-8") as handle:
            handle.write(
                "project:\n  name: demo\n  standards:\n    plugins:\n"
                "      - name: demo\n        enabled: true\n"
            )
        registry = _registry(self, project_root=root)
        registry.discover(root)
        loaded = registry.load_config(root)
        self.assertTrue(loaded["project_yaml_source"])
        demo = next(item for item in loaded["plugins"] if item["name"] == DEMO_PLUGIN_NAME)
        self.assertTrue(demo["enabled"])
        saved = registry.save_config(
            {
                "enforcement": {"prompt": False, "automated": True, "gates": True},
                "plugins": [{"name": DEMO_PLUGIN_NAME, "enabled": False, "action": "block"}],
            },
            root,
        )
        demo = next(item for item in saved["plugins"] if item["name"] == DEMO_PLUGIN_NAME)
        self.assertFalse(demo["enabled"])
        self.assertEqual(demo["action"], "block")
        self.assertFalse(saved["enforcement"]["prompt"])
        self.assertTrue(saved["project_yaml_source"])

    def test_run_checks_is_deterministic_and_skips_vendored(self) -> None:
        root = _worktree_with_tab()
        registry = _registry(self, project_root=root)
        registry.discover(root)
        registry.save_config(
            {"plugins": [{"name": DEMO_PLUGIN_NAME, "enabled": True}]},
            root,
        )
        first = registry.run_checks(root)
        second = registry.run_checks(root)
        self.assertEqual(first.summary["files_scanned"], second.summary["files_scanned"])
        self.assertEqual(len(first.violations), 1)
        self.assertEqual(first.violations[0].rule_id, DEMO_RULE_TAB_INDENT)
        self.assertEqual(first.violations[0].file, "src/app.py")
        self.assertEqual(first.violations[0].line, 2)
        files = {item.file for item in first.violations}
        self.assertNotIn("node_modules/lib.py", files)
        self.assertEqual(first.summary["warning"], 1)

    def test_audit_keyset_paging(self) -> None:
        root = _worktree_with_tab()
        registry = _registry(self, project_root=root)
        registry.discover(root)
        registry.save_config(
            {"plugins": [{"name": DEMO_PLUGIN_NAME, "enabled": True}]},
            root,
        )
        registry.run_checks(root)
        registry.run_checks(root)
        page = registry.audit.list_violations(limit=1)
        self.assertEqual(len(page["items"]), 1)
        self.assertIsNotNone(page["next_before_id"])
        older = registry.audit.list_violations(
            limit=1, before_id=page["next_before_id"]
        )
        self.assertEqual(len(older["items"]), 1)
        self.assertNotEqual(page["items"][0]["id"], older["items"][0]["id"])
        self.assertEqual(older["items"][0]["rule_id"], DEMO_RULE_TAB_INDENT)


if __name__ == "__main__":
    unittest.main()
