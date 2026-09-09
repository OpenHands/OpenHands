"""Unit and API tests for manager/worker coordination.

Run from the repo root:

    python3 tools/test_coordinator.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from coordinator import (  # noqa: E402
    CoordinatorError,
    CoordinatorStore,
    cheapest_capable_model,
    read_shared_file,
)
from coordinator_api import handle_request  # noqa: E402
from fleet import FleetStore  # noqa: E402
from kanban import KanbanStore  # noqa: E402
from projects import ProjectStore  # noqa: E402


def _request(
    store: CoordinatorStore,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return handle_request(store, method, path, body)


def _git(args: list[str], cwd: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    _git(["init", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("hello\n")
    _git(["add", "."], path)
    _git(["commit", "-m", "init"], path)


class CoordinatorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kanban = KanbanStore(":memory:")
        self.fleet = FleetStore(":memory:", kanban_store=self.kanban)
        self.store = CoordinatorStore(
            ":memory:",
            kanban_store=self.kanban,
            fleet_store=self.fleet,
        )
        self.board = self.kanban.create_board("Work", project_id="proj-1")
        self.backlog_id = self.board["columns"][0]["id"]
        self.review_id = self.board["columns"][2]["id"]
        self.card_a = self.kanban.create_card(self.backlog_id, title="Agent A")
        self.card_b = self.kanban.create_card(self.backlog_id, title="Agent B")

    def tearDown(self) -> None:
        self.store.close()
        self.fleet.close()
        self.kanban.close()

    def test_sequence_builds_dependency_graph(self) -> None:
        sequence = self.store.create_sequence(
            "proj-1", [self.card_a["id"], self.card_b["id"]], name="chain"
        )
        self.assertEqual(sequence["name"], "chain")
        self.assertEqual(len(sequence["steps"]), 2)
        self.assertIsNone(sequence["steps"][0]["depends_on_card_id"])
        self.assertEqual(
            sequence["steps"][1]["depends_on_card_id"], self.card_a["id"]
        )
        graph = sequence["dependency_graph"]
        self.assertEqual({node["card_id"] for node in graph["nodes"]}, {
            self.card_a["id"],
            self.card_b["id"],
        })
        self.assertEqual(
            graph["edges"],
            [{"from": self.card_a["id"], "to": self.card_b["id"]}],
        )

    def test_advance_waits_for_review_then_starts_dependent(self) -> None:
        sequence = self.store.create_sequence(
            "proj-1", [self.card_a["id"], self.card_b["id"]]
        )
        first = self.store.advance_sequence(sequence["id"])
        self.assertEqual(first["card_id"], self.card_a["id"])
        self.assertEqual(first["status"], "running")
        self.assertTrue(first["session_id"])

        blocked = self.store.advance_sequence(sequence["id"])
        self.assertIsNone(blocked)

        self.kanban.move_card(self.card_a["id"], self.review_id, 0)
        second = self.store.advance_sequence(sequence["id"])
        self.assertIsNotNone(second)
        self.assertEqual(second["card_id"], self.card_b["id"])
        self.assertEqual(second["status"], "running")
        loaded = self.store.get_sequence(sequence["id"])
        self.assertEqual(loaded["steps"][0]["status"], "complete")
        self.assertEqual(loaded["steps"][1]["status"], "running")

    def test_cheapest_capable_model_prefers_lower_price(self) -> None:
        self.assertEqual(cheapest_capable_model("docs"), "glm-5.2")
        self.assertEqual(cheapest_capable_model("code"), "gpt-4o")
        self.assertEqual(cheapest_capable_model("refactor"), "claude-sonnet")
        with self.assertRaises(CoordinatorError):
            cheapest_capable_model("unknown")

    def test_assign_spawns_fleet_session_with_cheapest_model(self) -> None:
        assigned = self.store.assign_card(
            "proj-1", self.card_a["id"], task_type="docs"
        )
        self.assertEqual(assigned["model"], "glm-5.2")
        self.assertEqual(assigned["card_id"], self.card_a["id"])
        session = self.fleet.get_session(assigned["id"])
        self.assertEqual(session["project_id"], "proj-1")

    def test_shared_context_reads_peer_worktree(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = os.path.join(tmp.name, "repo")
        _init_repo(source)
        projects = ProjectStore(
            db_path=":memory:",
            projects_root=os.path.join(tmp.name, "projects"),
        )
        self.addCleanup(projects.close)
        project = projects.create_project("Alpha", local_path=source)
        first = projects.create_worktree(project["id"], "agent-a")
        second = projects.create_worktree(project["id"], "agent-b")
        peer_file = os.path.join(first["path"], "notes.txt")
        with open(peer_file, "w", encoding="utf-8") as handle:
            handle.write("shared context\n")
        text = read_shared_file(
            projects, project["id"], "notes.txt", from_branch="agent-a"
        )
        self.assertEqual(text, "shared context\n")
        self.assertTrue(os.path.isdir(second["path"]))


class CoordinatorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kanban = KanbanStore(":memory:")
        self.fleet = FleetStore(":memory:", kanban_store=self.kanban)
        self.store = CoordinatorStore(
            ":memory:",
            kanban_store=self.kanban,
            fleet_store=self.fleet,
        )
        self.board = self.kanban.create_board("Work", project_id="proj-1")
        self.backlog_id = self.board["columns"][0]["id"]
        self.review_id = self.board["columns"][2]["id"]
        self.card_a = self.kanban.create_card(self.backlog_id, title="A")
        self.card_b = self.kanban.create_card(self.backlog_id, title="B")

    def tearDown(self) -> None:
        self.store.close()
        self.fleet.close()
        self.kanban.close()

    def test_sequence_and_assign_endpoints(self) -> None:
        status, sequence = _request(
            self.store,
            "POST",
            "/api/coordinator/sequences",
            {
                "project_id": "proj-1",
                "card_ids": [self.card_a["id"], self.card_b["id"]],
                "name": "pipeline",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(sequence["name"], "pipeline")
        self.assertEqual(len(sequence["dependency_graph"]["edges"]), 1)

        status, listed = _request(self.store, "GET", "/api/coordinator/sequences")
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

        status, loaded = _request(
            self.store, "GET", f"/api/coordinator/sequences/{sequence['id']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(loaded["id"], sequence["id"])

        status, started = _request(
            self.store,
            "POST",
            f"/api/coordinator/sequences/{sequence['id']}/advance",
        )
        self.assertEqual(status, 200)
        self.assertEqual(started["card_id"], self.card_a["id"])

        status, agents = _request(self.store, "GET", "/api/coordinator/agents")
        self.assertEqual(status, 200)
        self.assertTrue(agents)
        self.assertIn("model", agents[0])

        status, assigned = _request(
            self.store,
            "POST",
            "/api/coordinator/assign",
            {
                "project_id": "proj-1",
                "card_id": self.card_b["id"],
                "task_type": "docs",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(assigned["model"], "glm-5.2")

    def test_missing_and_validation(self) -> None:
        status, payload = _request(
            self.store, "GET", "/api/coordinator/sequences/nope"
        )
        self.assertEqual(status, 404)
        self.assertIn("error", payload)
        status, payload = _request(
            self.store, "POST", "/api/coordinator/sequences", {}
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
