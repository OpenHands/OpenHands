"""
tools/test_kanban.py - Test suite covering KanbanStore and Kanban REST API.
"""

import gc
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request

from kanban import KanbanStore, KanbanRequestHandler, create_server


def _safe_remove(path: str, retries: int = 5, delay: float = 0.1):
    gc.collect()
    for _ in range(retries):
        if not os.path.exists(path):
            return
        try:
            os.remove(path)
            return
        except PermissionError:
            time.sleep(delay)
            gc.collect()


class TestKanbanStore(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        self.store = KanbanStore(self.temp_file.name)

    def tearDown(self):
        self.store = None
        _safe_remove(self.temp_file.name)

    def test_create_board_seeds_default_columns(self):
        board = self.store.create_board("Sprint 1")
        self.assertEqual(board["name"], "Sprint 1")
        cols = [col["name"] for col in board["columns"]]
        self.assertEqual(cols, ["Backlog", "In Progress", "Review", "Done"])

    def test_card_crud_and_priorities(self):
        board = self.store.create_board("Dev Board")
        backlog_col_id = board["columns"][0]["id"]

        card = self.store.create_card(
            column_id=backlog_col_id,
            title="Fix DB Race Condition",
            description="Details here",
            priority="P0",
            assignee="prakhar",
            branch="fix/race-condition",
            pr="#17141",
            estimated_cost=25.0,
            actual_cost=10.0,
        )
        self.assertEqual(card["priority"], "P0")
        self.assertEqual(card["estimated_cost"], 25.0)

        updated = self.store.update_card(card["id"], actual_cost=22.5, priority="P1")
        self.assertEqual(updated["actual_cost"], 22.5)
        self.assertEqual(updated["priority"], "P1")

        self.assertTrue(self.store.delete_card(card["id"]))
        self.assertIsNone(self.store.get_card(card["id"]))

    def test_move_card_between_columns(self):
        board = self.store.create_board("Project Alpha")
        backlog_id = board["columns"][0]["id"]
        in_progress_id = board["columns"][1]["id"]

        card = self.store.create_card(column_id=backlog_id, title="Deploy to Staging")
        moved = self.store.move_card(card["id"], target_column_id=in_progress_id)

        self.assertEqual(moved["column_id"], in_progress_id)

    def test_cost_aggregates(self):
        board = self.store.create_board("Cost Tracker")
        col1_id = board["columns"][0]["id"]
        col2_id = board["columns"][1]["id"]

        self.store.create_card(col1_id, "Task 1", estimated_cost=15.0, actual_cost=10.0)
        self.store.create_card(col2_id, "Task 2", estimated_cost=35.0, actual_cost=40.0)

        costs = self.store.get_board_cost_aggregates(board["id"])
        self.assertEqual(costs["total_cards"], 2)
        self.assertEqual(costs["total_estimated_cost"], 50.0)
        self.assertEqual(costs["total_actual_cost"], 50.0)


class TestKanbanAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        cls.temp_file.close()
        cls.server = create_server(host="127.0.0.1", port=0, db_path=cls.temp_file.name)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)
        KanbanRequestHandler.store = None
        _safe_remove(cls.temp_file.name)

    def _request(self, method: str, path: str, data: dict = None):
        url = f"{self.base_url}{path}"
        req_data = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=req_data, method=method)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_full_rest_api_lifecycle(self):
        status, board = self._request("POST", "/api/boards", {"name": "Release 1.0"})
        self.assertEqual(status, 201)
        board_id = board["id"]
        backlog_id = board["columns"][0]["id"]
        in_prog_id = board["columns"][1]["id"]

        status, card = self._request(
            "POST",
            f"/api/columns/{backlog_id}/cards",
            {
                "title": "Build REST API",
                "priority": "P0",
                "estimated_cost": 50.0,
                "actual_cost": 20.0,
            },
        )
        self.assertEqual(status, 201)
        card_id = card["id"]

        status, moved = self._request(
            "POST",
            f"/api/cards/{card_id}/move",
            {"column_id": in_prog_id, "position": 0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(moved["column_id"], in_prog_id)

        status, costs = self._request("GET", f"/api/boards/{board_id}/costs")
        self.assertEqual(status, 200)
        self.assertEqual(costs["total_cards"], 1)
        self.assertEqual(costs["total_estimated_cost"], 50.0)
        self.assertEqual(costs["total_actual_cost"], 20.0)

        status, _ = self._request("DELETE", f"/api/boards/{board_id}")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()