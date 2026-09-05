"""Tests for `.openhands/project.yaml` parsing and validation."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from project_config import (  # noqa: E402
    ProjectConfigError,
    load_schema,
    parse_project_yaml,
    validate_project_config,
)

VALID_YAML = """
project:
  name: Demo
  description: Local kanban dogfood
  routing:
    rules:
      - task_type: code
        provider: claude
        model: opus
    default:
      provider: cursor
      model: sonnet
  kanban:
    columns:
      - name: Backlog
        color: "#6b7280"
    card_types:
      - task
      - bug
  lint:
    rules:
      - sonarjs
  channels:
    - type: slack
      config:
        channel: eng
  standards:
    plugins:
      - name: gdpr
        enabled: true
        config:
          strict: false
  cost_cap: 25.5
"""


class ProjectConfigTests(unittest.TestCase):
    def test_schema_file_is_json(self) -> None:
        schema = load_schema()
        self.assertEqual(schema["required"], ["project"])

    def test_valid_yaml_parses_and_validates(self) -> None:
        data = validate_project_config(parse_project_yaml(VALID_YAML))
        self.assertEqual(data["project"]["name"], "Demo")
        self.assertEqual(data["project"]["routing"]["default"]["provider"], "cursor")
        self.assertEqual(data["project"]["kanban"]["card_types"], ["task", "bug"])
        self.assertEqual(data["project"]["lint"]["rules"], ["sonarjs"])
        self.assertEqual(data["project"]["channels"][0]["type"], "slack")
        self.assertTrue(data["project"]["standards"]["plugins"][0]["enabled"])
        self.assertAlmostEqual(data["project"]["cost_cap"], 25.5)

    def test_rejects_unknown_provider(self) -> None:
        data = parse_project_yaml(VALID_YAML)
        data["project"]["routing"]["default"]["provider"] = "gpt"
        with self.assertRaises(ProjectConfigError):
            validate_project_config(data)

    def test_rejects_missing_name(self) -> None:
        with self.assertRaises(ProjectConfigError):
            validate_project_config({"project": {"description": "x"}})

    def test_rejects_unknown_card_type(self) -> None:
        data = {"project": {"name": "x", "kanban": {"card_types": ["epic"]}}}
        with self.assertRaises(ProjectConfigError):
            validate_project_config(data)

    def test_rejects_negative_cost_cap(self) -> None:
        with self.assertRaises(ProjectConfigError):
            validate_project_config({"project": {"name": "x", "cost_cap": -1}})

    def test_load_from_file(self) -> None:
        from project_config import load_project_config

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "project.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(VALID_YAML)
            loaded = load_project_config(path)
            self.assertEqual(loaded["project"]["name"], "Demo")


if __name__ == "__main__":
    unittest.main()
