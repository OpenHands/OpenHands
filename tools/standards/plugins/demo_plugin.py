"""Trivial deterministic plugin: reject tab-indented lines."""

from __future__ import annotations

from standards import (
    ACTION_WARN,
    SEVERITY_WARNING,
    StandardsPlugin,
    Violation,
)

DEMO_PLUGIN_NAME = "demo"
DEMO_RULE_TAB_INDENT = "DEMO-TAB-INDENT"
DEMO_VERSION = "1.0.0"


class DemoPlugin(StandardsPlugin):
    name = DEMO_PLUGIN_NAME
    display_name = "Demo (tab indentation)"
    description = "Flags lines that start with a tab character. Reference plugin."
    version = DEMO_VERSION
    severity_default = SEVERITY_WARNING

    def check(self, file: str, content: str) -> list[Violation]:
        hits: list[Violation] = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            if line.startswith("\t"):
                hits.append(
                    Violation(
                        plugin_name=self.name,
                        rule_id=DEMO_RULE_TAB_INDENT,
                        severity=self.severity_default,
                        file=file,
                        line=line_no,
                        message="Line uses tab indentation",
                        remediation="Replace leading tabs with spaces",
                        action=ACTION_WARN,
                        fixable=True,
                    )
                )
        return hits

    def prompt_instructions(self) -> str:
        return (
            "Do not indent with tab characters. Use spaces for indentation "
            f"(rule {DEMO_RULE_TAB_INDENT})."
        )

    def auto_fix(self, file: str, content: str) -> str | None:
        if "\t" not in content:
            return None
        return content.replace("\t", "    ")


PLUGIN = DemoPlugin()
