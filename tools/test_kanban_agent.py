"""Integration tests for agent session linking and auto-progress."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kanban import KanbanStore  # noqa: E402
from kanban_agent import complete_session, link_session, record_progress  # noqa: E402
from kanban_api import handle_request  # noqa: E402


class KanbanAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KanbanStore(":memory:")
        self.board = self.store.create_board("Work")
        self.backlog_id = self.board["columns"][0]["id"]
        self.progress_id = self.board["columns"][1]["id"]
        self.review_id = self.board["columns"][2]["id"]
        self.card = self.store.create_card(self.backlog_id, title="Implement API")

    def tearDown(self) -> None:
        self.store.close()

    def test_link_session_moves_card_to_in_progress(self) -> None:
        linked = link_session(self.store, self.card["id"], "sess-1")
        self.assertEqual(linked["agent_session_id"], "sess-1")
        self.assertEqual(linked["column_id"], self.progress_id)
        self.assertEqual(linked["status"], "in_progress")
        self.assertTrue(linked["activity_log"])

    def test_progress_appends_activity(self) -> None:
        link_session(self.store, self.card["id"], "sess-1")
        updated = record_progress(
            self.store, self.card["id"], "Wrote tests", status="in_progress"
        )
        messages = [item["message"] for item in updated["activity_log"]]
        self.assertIn("Wrote tests", messages)

    def test_complete_moves_to_review_when_tests_pass(self) -> None:
        link_session(self.store, self.card["id"], "sess-1")
        done = complete_session(
            self.store,
            self.card["id"],
            tests_passed=True,
            actual_tokens=800,
            actual_cost=0.4,
            tool_calls=3,
            agent_time=12.5,
            model_used="openhands/glm-5.2",
        )
        self.assertEqual(done["column_id"], self.review_id)
        self.assertAlmostEqual(done["actual_cost"], 0.4)
        self.assertEqual(done["status"], "review")

    def test_complete_stays_in_progress_when_tests_fail(self) -> None:
        link_session(self.store, self.card["id"], "sess-1")
        done = complete_session(
            self.store,
            self.card["id"],
            tests_passed=False,
            actual_cost=0.2,
        )
        self.assertEqual(done["column_id"], self.progress_id)
        self.assertEqual(done["status"], "in_progress")

    def test_link_session_http(self) -> None:
        status, payload = handle_request(
            self.store,
            "POST",
            f"/api/cards/{self.card['id']}/link-session",
            {"session_id": "sess-http"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["agent_session_id"], "sess-http")


if __name__ == "__main__":
    unittest.main()
