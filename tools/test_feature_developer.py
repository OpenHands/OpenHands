"""Integration tests for the text-to-feature orchestrator.

Run from the repo root:

    python3 tools/test_feature_developer.py
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from feature_developer import (  # noqa: E402
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_PASSED,
    STATUS_PAUSED,
    TICKET_FAILED,
    TICKET_PASSED,
    TICKET_PENDING,
    FeatureDeveloper,
)
from feature_developer_api import handle_request  # noqa: E402
from fleet import FleetStore  # noqa: E402
from kanban import KanbanStore  # noqa: E402


def _llm(_prompt: str) -> str:
    return """
    {"features":[{"name":"Auth","epics":[{"name":"Login","tickets":[
      {"title":"Add login form","description":"Form","acceptance":["Renders"]},
      {"title":"Add session cookie","description":"Cookie","acceptance":["Sets cookie"]}
    ]}]}]}
    """


class _FakeCommitLoop:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_indexes: set[int] = set()

    def setup(self, project_id: str, **_kwargs: Any) -> dict[str, Any]:
        return {"id": "loop-1", "project_id": project_id, "name": "commit-loop"}

    def run_session(self, session_id: str, **_kwargs: Any) -> dict[str, Any]:
        index = len(self.calls)
        self.calls.append(session_id)
        if index in self.fail_indexes:
            return {
                "status": STATUS_FAILED,
                "run": {"total_cost_usd": 0.1},
                "commit_sha": None,
                "commit_message": None,
            }
        return {
            "status": STATUS_PASSED,
            "run": {"total_cost_usd": 0.25},
            "commit_sha": "abc123",
            "commit_message": "feat: ticket",
        }


class FeatureDeveloperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kanban = KanbanStore(":memory:")
        self.fleet = FleetStore(":memory:", kanban_store=self.kanban)
        self.commit = _FakeCommitLoop()
        self.dev = FeatureDeveloper(
            kanban_store=self.kanban,
            fleet_store=self.fleet,
            commit_loop=self.commit,
            llm_complete=_llm,
        )

    def tearDown(self) -> None:
        self.dev.close()
        self.fleet.close()
        self.kanban.close()

    def test_decompose_seeds_kanban_and_estimates(self) -> None:
        run = self.dev.start_run("proj-1", "Build login")
        self.assertEqual(len(run["tickets"]), 2)
        board = self.kanban.get_board(run["board_id"])
        titles = [
            card["title"]
            for column in board["columns"]
            for card in column["cards"]
        ]
        self.assertIn("Add login form", titles)
        self.assertIn("Add session cookie", titles)
        self.assertGreater(run["total_estimate_usd"], 0)
        self.assertEqual(len(self.commit.calls), 2)
        self.assertEqual(run["status"], STATUS_PASSED)
        self.assertAlmostEqual(run["total_actual_usd"], 0.5)

    def test_partial_when_one_ticket_fails(self) -> None:
        self.commit.fail_indexes.add(1)
        run = self.dev.start_run(
            "proj-1", "Build login", continue_on_failure=True
        )
        self.assertEqual(run["status"], STATUS_PARTIAL)
        statuses = [ticket["status"] for ticket in run["tickets"]]
        self.assertEqual(statuses, [TICKET_PASSED, TICKET_FAILED])
        self.assertTrue(run["tickets"][1]["error"])

    def test_abort_stops_remaining_tickets(self) -> None:
        seen: list[str] = []

        def implement(run: dict[str, Any], ticket: dict[str, Any]) -> dict[str, Any]:
            seen.append(ticket["title"])
            self.dev.abort_run(run["id"])
            return {"status": TICKET_PASSED, "actual_usd": 0.1, "branch_name": "b"}

        self.dev.implement_fn = implement
        run = self.dev.start_run("proj-1", "Build login")
        self.assertEqual(run["status"], STATUS_ABORTED)
        self.assertEqual(seen, ["Add login form"])
        self.assertEqual(run["tickets"][1]["status"], "skipped")

    def test_pause_and_resume(self) -> None:
        def implement(run: dict[str, Any], ticket: dict[str, Any]) -> dict[str, Any]:
            if ticket["title"] == "Add login form":
                self.dev.pause_run(run["id"])
            return {
                "status": TICKET_PASSED,
                "actual_usd": 0.1,
                "branch_name": f"feat/{ticket['id'][:8]}",
            }

        self.dev.implement_fn = implement
        run = self.dev.start_run("proj-1", "Build login")
        self.assertEqual(run["status"], STATUS_PAUSED)
        self.assertEqual(run["tickets"][0]["status"], TICKET_PASSED)
        self.assertEqual(run["tickets"][1]["status"], TICKET_PENDING)
        resumed = self.dev.resume_run(run["id"])
        self.assertEqual(resumed["status"], STATUS_PASSED)
        self.assertEqual(resumed["tickets"][1]["status"], TICKET_PASSED)

    def test_report_markdown_table(self) -> None:
        run = self.dev.start_run("proj-1", "Build login")
        report = self.dev.report(run["id"])
        self.assertIn("| Title | Status | Cost | Branch |", report)
        self.assertIn("Add login form", report)
        self.assertIn("passed", report)

    def test_api_start_list_detail_and_report(self) -> None:
        status, run = handle_request(
            self.dev,
            "POST",
            "/api/feature-developer/runs",
            {"project_id": "proj-1", "spec_text": "Build login"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(run["status"], STATUS_PASSED)
        status, listed = handle_request(
            self.dev, "GET", "/api/feature-developer/runs?project_id=proj-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)
        status, detail = handle_request(
            self.dev, "GET", f"/api/feature-developer/runs/{run['id']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(detail["tickets"]), 2)
        status, report = handle_request(
            self.dev, "GET", f"/api/feature-developer/runs/{run['id']}/report"
        )
        self.assertEqual(status, 200)
        self.assertIn("Add login form", report["markdown"])

    def test_api_abort_paused_run(self) -> None:
        def implement(run: dict[str, Any], ticket: dict[str, Any]) -> dict[str, Any]:
            self.dev.pause_run(run["id"])
            return {
                "status": TICKET_PASSED,
                "actual_usd": 0.1,
                "branch_name": "feat/one",
            }

        self.dev.implement_fn = implement
        _, run = handle_request(
            self.dev,
            "POST",
            "/api/feature-developer/runs",
            {"project_id": "proj-1", "spec_text": "Build login"},
        )
        self.assertEqual(run["status"], STATUS_PAUSED)
        status, aborted = handle_request(
            self.dev, "POST", f"/api/feature-developer/runs/{run['id']}/abort"
        )
        self.assertEqual(status, 200)
        self.assertEqual(aborted["status"], STATUS_ABORTED)


if __name__ == "__main__":
    unittest.main()
