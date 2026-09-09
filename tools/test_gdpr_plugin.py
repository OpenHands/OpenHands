"""Fixture tests for the GDPR heuristic plugin."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from standards.plugins.gdpr_plugin import (  # noqa: E402
    GDPR_CONSENT,
    GDPR_EMAIL,
    GDPR_ERASURE,
    GDPR_PERSONAL_ID,
    GDPR_PHONE,
    GDPR_PII_LOG,
    GDPR_PLUGIN_NAME,
    GDPR_UNENCRYPTED,
    PLUGIN,
)


class GdprPluginTests(unittest.TestCase):
    def test_flags_violating_fixture(self) -> None:
        content = "\n".join(
            [
                'contact = "ada@company.co"',
                'phone = "+33123456789"',
                "PESEL: 44051401359",
                'logger.info("email=" + contact)',
                '<input type="email" name="email" />',
                "pickle.dump(user_record, fh)",
            ]
        )
        hits = PLUGIN.check("app.py", content)
        rules = {item.rule_id for item in hits}
        self.assertEqual(PLUGIN.name, GDPR_PLUGIN_NAME)
        self.assertIn(GDPR_EMAIL, rules)
        self.assertIn(GDPR_PHONE, rules)
        self.assertIn(GDPR_PERSONAL_ID, rules)
        self.assertIn(GDPR_PII_LOG, rules)
        self.assertIn(GDPR_CONSENT, rules)
        self.assertIn(GDPR_ERASURE, rules)
        self.assertIn(GDPR_UNENCRYPTED, rules)

    def test_compliant_fixture_is_clean(self) -> None:
        content = "\n".join(
            [
                'contact = "user@example.com"',
                '<input type="email" name="email" />',
                '<input type="checkbox" name="consent" /> opt-in GDPR',
                '@app.delete("/users/<id>/erasure")',
                "def forget_me(): pass",
            ]
        )
        hits = PLUGIN.check("routes.py", content)
        self.assertEqual(hits, [])

    def test_deterministic(self) -> None:
        text = 'x = "ada@company.co"\n'
        self.assertEqual(PLUGIN.check("a.py", text), PLUGIN.check("a.py", text))

    def test_prompt_mentions_rules(self) -> None:
        text = PLUGIN.prompt_instructions()
        self.assertIn(GDPR_CONSENT, text)
        self.assertTrue(text)


if __name__ == "__main__":
    unittest.main()
