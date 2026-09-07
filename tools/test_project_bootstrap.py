"""Tests for project bootstrap: decomposition, detection, and kanban seeding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kanban import KanbanStore  # noqa: E402
from kanban_api import handle_request  # noqa: E402
from project_bootstrap import (  # noqa: E402
    DecompositionError,
    decompose_spec,
    detect_project,
    init_project,
    preview_project,
    stub_llm_complete,
    validate_decomposition,
)

VALID_DECOMPOSITION = {
    "features": [
        {
            "name": "Auth",
            "epics": [
                {
                    "name": "Login",
                    "tickets": [
                        {
                            "title": "Add login form",
                            "description": "Users can sign in",
                            "acceptance": ["Form submits", "Errors shown"],
                        }
                    ],
                }
            ],
        }
    ]
}


def _llm(_prompt: str) -> str:
    return json.dumps(VALID_DECOMPOSITION)


class DecompositionTests(unittest.TestCase):
    def test_valid_spec_flattens_tickets(self) -> None:
        cards = decompose_spec("Build login", _llm)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].title, "Add login form")
        self.assertEqual(cards[0].source, "decomposition")
        self.assertEqual(cards[0].acceptance, ["Form submits", "Errors shown"])

    def test_rejects_feature_without_epics(self) -> None:
        with self.assertRaises(DecompositionError):
            validate_decomposition({"features": [{"name": "Auth", "epics": []}]})

    def test_rejects_epic_without_tickets(self) -> None:
        with self.assertRaises(DecompositionError):
            validate_decomposition(
                {
                    "features": [
                        {
                            "name": "Auth",
                            "epics": [{"name": "Login", "tickets": []}],
                        }
                    ]
                }
            )

    def test_rejects_ticket_without_acceptance(self) -> None:
        with self.assertRaises(DecompositionError):
            validate_decomposition(
                {
                    "features": [
                        {
                            "name": "Auth",
                            "epics": [
                                {
                                    "name": "Login",
                                    "tickets": [
                                        {
                                            "title": "Login",
                                            "description": "",
                                            "acceptance": [],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            )

    def test_stub_llm_produces_valid_structure(self) -> None:
        cards = decompose_spec("Ship billing", stub_llm_complete)
        self.assertEqual(cards[0].title, "Ship billing")


class DetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_detects_readme_and_code_todos(self) -> None:
        (self.root / "README.md").write_text("# App\nTODO: document API\n", encoding="utf-8")
        (self.root / "app.py").write_text("# FIXME: handle empty input\n", encoding="utf-8")
        sources = {card.source for card in detect_project(self.root)}
        titles = [card.title for card in detect_project(self.root)]
        self.assertIn("readme", sources)
        self.assertIn("code_todo", sources)
        self.assertTrue(any("document API" in title for title in titles))
        self.assertTrue(any("handle empty input" in title for title in titles))

    def test_detects_missing_package_scripts_and_ci(self) -> None:
        (self.root / "package.json").write_text(
            json.dumps({"scripts": {"dev": "vite"}}), encoding="utf-8"
        )
        cards = detect_project(self.root)
        sources = {card.source for card in cards}
        self.assertIn("package_json", sources)
        self.assertIn("ci", sources)

    def test_detects_git_hotspots(self) -> None:
        subprocess.run(["git", "init"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        target = self.root / "hot.py"
        target.write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "add", "hot.py"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "one"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        target.write_text("two\n", encoding="utf-8")
        subprocess.run(["git", "add", "hot.py"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "two"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        cards = detect_project(self.root)
        self.assertTrue(any(card.source == "git_history" for card in cards))


class SeedTests(unittest.TestCase):
    def test_init_project_writes_cards_to_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("TODO: ship docs\n", encoding="utf-8")
            store = KanbanStore(":memory:")
            result = init_project(root, "Build login", store, llm_complete=_llm)
            titles = {card["title"] for card in result["cards"]}
            self.assertIn("Add login form", titles)
            self.assertTrue(any("ship docs" in title for title in titles))
            self.assertEqual(result["board"]["name"], "Project board")
            store.close()

    def test_preview_does_not_require_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cards = preview_project(tmp, "Build login", _llm)
            self.assertEqual(cards[0].title, "Add login form")


class ApiTests(unittest.TestCase):
    def test_preview_and_init_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("TODO: ship docs\n", encoding="utf-8")
            store = KanbanStore(":memory:")
            status, preview = handle_request(
                store,
                "POST",
                "/api/project/preview",
                {"spec": "Build login", "root": str(root)},
            )
            self.assertEqual(status, 200)
            self.assertTrue(preview["suggested"])
            status, created = handle_request(
                store,
                "POST",
                "/api/project/init",
                {
                    "spec": "Build login",
                    "root": str(root),
                    "board_name": "Demo",
                },
            )
            self.assertEqual(status, 201)
            self.assertEqual(created["board"]["name"], "Demo")
            self.assertGreaterEqual(len(created["cards"]), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
