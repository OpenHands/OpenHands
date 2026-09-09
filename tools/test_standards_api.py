"""HTTP API tests for standards plugins.

Run from the repo root:

    python3 -m unittest tools.test_standards_api
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from standards.api import StandardsService, handle_request  # noqa: E402
from standards.audit_store import AuditStore  # noqa: E402
from standards.plugins.demo_plugin import DEMO_PLUGIN_NAME, DEMO_RULE_TAB_INDENT  # noqa: E402
from standards.registry import StandardsRegistry  # noqa: E402
from test_standards_registry import _worktree_with_tab  # noqa: E402


def _service(test: unittest.TestCase) -> StandardsService:
    store = AuditStore(":memory:")
    test.addCleanup(store.close)
    registry = StandardsRegistry(
        audit=store,
        user_dir=os.path.join(tempfile.mkdtemp(), "missing"),
    )
    registry.discover()
    return StandardsService(registry)


class StandardsApiTests(unittest.TestCase):
    def test_list_plugins_includes_demo(self) -> None:
        service = _service(self)
        status, payload = handle_request(service, "GET", "/api/standards/plugins")
        self.assertEqual(status, 200)
        names = {item["name"] for item in payload["plugins"]}
        self.assertIn(DEMO_PLUGIN_NAME, names)
        demo = next(item for item in payload["plugins"] if item["name"] == DEMO_PLUGIN_NAME)
        self.assertEqual(demo["source"], "builtin")
        self.assertIn("load_errors", payload)

    def test_config_get_and_put(self) -> None:
        service = _service(self)
        status, config = handle_request(service, "GET", "/api/standards/config")
        self.assertEqual(status, 200)
        self.assertTrue(config["enforcement"]["prompt"])
        status, updated = handle_request(
            service,
            "PUT",
            "/api/standards/config",
            {
                "enforcement": {"prompt": True, "automated": False, "gates": True},
                "plugins": [{"name": DEMO_PLUGIN_NAME, "enabled": True, "action": "block"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertFalse(updated["enforcement"]["automated"])
        demo = next(item for item in updated["plugins"] if item["name"] == DEMO_PLUGIN_NAME)
        self.assertTrue(demo["enabled"])
        self.assertEqual(demo["action"], "block")

    def test_run_and_audit(self) -> None:
        service = _service(self)
        root = _worktree_with_tab()
        handle_request(
            service,
            "PUT",
            "/api/standards/config",
            {"root": root, "plugins": [{"name": DEMO_PLUGIN_NAME, "enabled": True}]},
        )
        status, payload = handle_request(
            service, "POST", "/api/standards/run", {"root": root}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["summary"]["warning"], 1)
        self.assertEqual(payload["violations"][0]["rule_id"], DEMO_RULE_TAB_INDENT)
        self.assertIn("run_id", payload)
        status, audit = handle_request(
            service,
            "GET",
            f"/api/standards/audit?plugin={DEMO_PLUGIN_NAME}&limit=10",
        )
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(audit["items"]), 1)
        self.assertEqual(audit["items"][0]["run_id"], payload["run_id"])

    def test_run_requires_root(self) -> None:
        service = _service(self)
        status, payload = handle_request(service, "POST", "/api/standards/run", {})
        self.assertEqual(status, 400)
        self.assertIn("root", payload["error"])

    def test_unknown_route_is_404(self) -> None:
        service = _service(self)
        status, payload = handle_request(service, "GET", "/api/standards/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
