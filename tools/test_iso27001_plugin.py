"""Fixture tests for the ISO 27001 heuristic plugin."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from standards.plugins.iso27001_plugin import (  # noqa: E402
    ISO_ENV_COMMITTED,
    ISO_HTTP_INSECURE,
    ISO_PLUGIN_NAME,
    ISO_SECRET_AWS,
    ISO_SECRET_KEY,
    ISO_SECRET_PRIVATE_KEY,
    ISO_TLS_DISABLED,
    ISO_UNPINNED_DEP,
    PLUGIN,
)


class Iso27001PluginTests(unittest.TestCase):
    def test_flags_secrets_tls_and_http(self) -> None:
        content = "\n".join(
            [
                'key = "AKIAIOSFODNN7EXAMPLE"',
                'api_key = "sk_live_abcdefghijklmnopqrstuv"',
                "-----BEGIN RSA PRIVATE KEY-----",
                'client.get("http://api.vendor.test/v1")',
                "requests.get(url, verify=False)",
            ]
        )
        hits = PLUGIN.check("settings.py", content)
        rules = {item.rule_id for item in hits}
        self.assertEqual(PLUGIN.name, ISO_PLUGIN_NAME)
        self.assertIn(ISO_SECRET_AWS, rules)
        self.assertIn(ISO_SECRET_KEY, rules)
        self.assertIn(ISO_SECRET_PRIVATE_KEY, rules)
        self.assertIn(ISO_HTTP_INSECURE, rules)
        self.assertIn(ISO_TLS_DISABLED, rules)

    def test_flags_env_and_unpinned_deps(self) -> None:
        env_hits = PLUGIN.check(".env", "SECRET=1\n")
        self.assertIn(ISO_ENV_COMMITTED, {item.rule_id for item in env_hits})
        req_hits = PLUGIN.check("requirements.txt", "flask>=2.0\n")
        self.assertIn(ISO_UNPINNED_DEP, {item.rule_id for item in req_hits})
        pkg_hits = PLUGIN.check(
            "package.json",
            '{"dependencies": {"leftpad": "^1.0.0"}}',
        )
        self.assertIn(ISO_UNPINNED_DEP, {item.rule_id for item in pkg_hits})

    def test_compliant_fixture_is_clean(self) -> None:
        py = "\n".join(
            [
                "requests.get('https://api.vendor.test/v1', verify=True)",
                "TOKEN = os.environ['API_TOKEN']",
            ]
        )
        self.assertEqual(PLUGIN.check("settings.py", py), [])
        self.assertEqual(PLUGIN.check("requirements.txt", "flask==2.3.2\n"), [])
        self.assertEqual(
            PLUGIN.check("package.json", '{"dependencies": {"leftpad": "1.0.0"}}'),
            [],
        )
        self.assertEqual(PLUGIN.check(".env.example", "SECRET=\n"), [])

    def test_localhost_http_is_allowed(self) -> None:
        hits = PLUGIN.check("dev.py", 'url = "http://localhost:8000/health"\n')
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
