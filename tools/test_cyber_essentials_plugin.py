"""Fixture tests for the Cyber Essentials heuristic plugin."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from standards.plugins.cyber_essentials_plugin import (  # noqa: E402
    CE_DEBUG_FLAG,
    CE_DEBUG_ROUTE,
    CE_DEFAULT_CREDS,
    CE_PERMISSIVE_CORS,
    CE_PLUGIN_NAME,
    CE_UNBOUNDED_AUTH,
    CE_WEAK_CRYPTO,
    PLUGIN,
)


class CyberEssentialsPluginTests(unittest.TestCase):
    def test_flags_violating_fixture(self) -> None:
        content = "\n".join(
            [
                'password = "admin"',
                "digest = hashlib.md5(password.encode()).hexdigest()",
                'response.headers["Access-Control-Allow-Origin"] = "*"',
                '@app.route("/debug")',
                "DEBUG = True",
            ]
        )
        hits = PLUGIN.check("app.py", content)
        rules = {item.rule_id for item in hits}
        self.assertEqual(PLUGIN.name, CE_PLUGIN_NAME)
        self.assertIn(CE_DEFAULT_CREDS, rules)
        self.assertIn(CE_WEAK_CRYPTO, rules)
        self.assertIn(CE_PERMISSIVE_CORS, rules)
        self.assertIn(CE_DEBUG_ROUTE, rules)
        self.assertIn(CE_DEBUG_FLAG, rules)
        self.assertIn(CE_UNBOUNDED_AUTH, rules)

    def test_compliant_fixture_is_clean(self) -> None:
        content = "\n".join(
            [
                "password = os.environ['ADMIN_PASSWORD']",
                "digest = hashlib.sha256(password.encode()).hexdigest()",
                'response.headers["Access-Control-Allow-Origin"] = "https://app.example.com"',
                "@login_required",
                '@app.route("/account")',
                "DEBUG = False",
            ]
        )
        self.assertEqual(PLUGIN.check("app.py", content), [])

    def test_auto_fix_debug_flag(self) -> None:
        original = "DEBUG = True\nHOST = '0.0.0.0'\n"
        fixed = PLUGIN.auto_fix("settings.py", original)
        self.assertIsNotNone(fixed)
        self.assertIn("DEBUG = False", fixed or "")
        self.assertEqual(PLUGIN.check("settings.py", fixed or ""), [])


if __name__ == "__main__":
    unittest.main()
