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
import urllib.error
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

    def test_create_board_rejects_blank_name(self):
        with self.assertRaises(ValueError):
            self.store.create_board(None)
        with self.assertRaises(ValueError):
            self.store.create_board("   ")

    def test_create_card_rejects_invalid_inputs(self):
        board = self.store.create_board("Validation")
        col_id = board["columns"][0]["id"]

        with self.assertRaises(ValueError):
            self.store.create_card(col_id, None)
        with self.assertRaises(ValueError):
            self.store.create_card(col_id, "  ")
        with self.assertRaises(ValueError):
            self.store.create_card(col_id, "Task", priority="P9")
        with self.assertRaises(ValueError):
            self.store.create_card(col_id, "Task", estimated_cost="free")
        with self.assertRaises(ValueError):
            self.store.create_card(col_id, "Task", position="x")

    def test_update_card_coerces_and_validates_types(self):
        board = self.store.create_board("Coercion")
        col_id = board["columns"][0]["id"]
        card = self.store.create_card(col_id, "Task")

        updated = self.store.update_card(card["id"], estimated_cost="42.5")
        self.assertEqual(updated["estimated_cost"], 42.5)

        updated = self.store.update_card(card["id"], actual_cost=7)
        self.assertEqual(updated["actual_cost"], 7.0)

        updated = self.store.update_card(card["id"], priority="P0")
        self.assertEqual(updated["priority"], "P0")

        with self.assertRaises(ValueError):
            self.store.update_card(card["id"], priority=["P1"])
        with self.assertRaises(ValueError):
            self.store.update_card(card["id"], estimated_cost="free")
        with self.assertRaises(ValueError):
            self.store.update_card(card["id"], position=1.7)
        with self.assertRaises(ValueError):
            self.store.update_card(card["id"], title=None)

    def test_move_card_rejects_cross_board_move(self):
        board_a = self.store.create_board("Board A")
        board_b = self.store.create_board("Board B")
        card = self.store.create_card(board_a["columns"][0]["id"], "Task", estimated_cost=50.0)

        with self.assertRaises(ValueError):
            self.store.move_card(card["id"], board_b["columns"][0]["id"])

        # The card should remain untouched after the rejected move.
        self.assertEqual(self.store.get_card(card["id"])["column_id"], board_a["columns"][0]["id"])

    def test_move_card_rejects_nonexistent_target(self):
        board = self.store.create_board("Move Target")
        card = self.store.create_card(board["columns"][0]["id"], "Task")
        with self.assertRaises(ValueError):
            self.store.move_card(card["id"], 99999)

    def test_move_card_validates_position(self):
        board = self.store.create_board("Move Position")
        card = self.store.create_card(board["columns"][0]["id"], "Task")
        with self.assertRaises(ValueError):
            self.store.move_card(card["id"], board["columns"][1]["id"], position="x")


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

    def _request(self, method: str, path: str, data: dict = None, raw_body: bytes = None, content_type: str = "application/json"):
        url = f"{self.base_url}{path}"
        if raw_body is not None:
            req = urllib.request.Request(url, data=raw_body, method=method)
            req.add_header("Content-Type", content_type)
        else:
            req_data = json.dumps(data).encode("utf-8") if data is not None else None
            req = urllib.request.Request(url, data=req_data, method=method)
            if data is not None:
                req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

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

    def test_invalid_json_returns_400(self):
        status, body = self._request("POST", "/api/boards", raw_body=b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_create_board_rejects_blank_name(self):
        status, body = self._request("POST", "/api/boards", {"name": None})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

        status, body = self._request("POST", "/api/boards", {"name": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_create_column_rejects_blank_name(self):
        status, board = self._request("POST", "/api/boards", {"name": "Col Board"})
        self.assertEqual(status, 201)
        status, body = self._request("POST", f'/api/boards/{board["id"]}/columns', {"name": None})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_create_card_rejects_invalid_payloads(self):
        status, board = self._request("POST", "/api/boards", {"name": "Card Board"})
        self.assertEqual(status, 201)
        col_id = board["columns"][0]["id"]

        status, body = self._request("POST", f"/api/columns/{col_id}/cards", {"title": None})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

        status, body = self._request(
            "POST", f"/api/columns/{col_id}/cards", {"title": "Task", "priority": ["P1"]}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_update_card_coerces_and_validates_types(self):
        status, board = self._request("POST", "/api/boards", {"name": "PUT Board"})
        self.assertEqual(status, 201)
        col_id = board["columns"][0]["id"]
        status, card = self._request("POST", f"/api/columns/{col_id}/cards", {"title": "Task"})
        self.assertEqual(status, 201)
        card_id = card["id"]

        status, updated = self._request("PUT", f"/api/cards/{card_id}", {"estimated_cost": "50"})
        self.assertEqual(status, 200)
        self.assertEqual(updated["estimated_cost"], 50.0)

        status, body = self._request("PUT", f"/api/cards/{card_id}", {"estimated_cost": "free"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

        status, body = self._request("PUT", f"/api/cards/{card_id}", {"priority": ["P1"]})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_move_card_rejects_cross_board_move(self):
        _, board_a = self._request("POST", "/api/boards", {"name": "Board A"})
        _, board_b = self._request("POST", "/api/boards", {"name": "Board B"})
        col_a = board_a["columns"][0]["id"]
        col_b = board_b["columns"][0]["id"]

        _, card = self._request("POST", f"/api/columns/{col_a}/cards", {"title": "Cost Card", "estimated_cost": 50.0})
        card_id = card["id"]

        status, body = self._request("POST", f"/api/cards/{card_id}/move", {"column_id": col_b})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

        # The card stays in board A and its cost stays attributed to board A.
        status, costs = self._request("GET", f"/api/boards/{board_a['id']}/costs")
        self.assertEqual(status, 200)
        self.assertEqual(costs["total_cards"], 1)
        self.assertEqual(costs["total_estimated_cost"], 50.0)

    def test_move_card_rejects_invalid_position(self):
        status, board = self._request("POST", "/api/boards", {"name": "Move Board"})
        self.assertEqual(status, 201)
        col_id = board["columns"][0]["id"]
        _, card = self._request("POST", f"/api/columns/{col_id}/cards", {"title": "Task"})
        status, body = self._request(
            "POST",
            f"/api/cards/{card['id']}/move",
            {"column_id": board["columns"][1]["id"], "position": "x"},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_column_read_and_update_routes(self):
        status, board = self._request("POST", "/api/boards", {"name": "Col Routes"})
        self.assertEqual(status, 201)
        board_id = board["id"]

        status, columns = self._request("GET", f"/api/boards/{board_id}/columns")
        self.assertEqual(status, 200)
        self.assertEqual(len(columns), 4)

        col_id = columns[3]["id"]
        status, column = self._request("GET", f"/api/columns/{col_id}")
        self.assertEqual(status, 200)
        self.assertEqual(column["name"], "Done")

        status, updated = self._request("PUT", f"/api/columns/{col_id}", {"name": "Shipped"})
        self.assertEqual(status, 200)
        self.assertEqual(updated["name"], "Shipped")

        status, _ = self._request("DELETE", f"/api/columns/{col_id}")
        self.assertEqual(status, 200)
        status, _ = self._request("GET", f"/api/columns/{col_id}")
        self.assertEqual(status, 404)

    def test_board_rename_and_card_listing_routes(self):
        status, board = self._request("POST", "/api/boards", {"name": "Rename Board"})
        self.assertEqual(status, 201)
        board_id = board["id"]
        col_id = board["columns"][0]["id"]

        _, card = self._request("POST", f"/api/columns/{col_id}/cards", {"title": "List Me"})
        card_id = card["id"]

        status, boards = self._request("GET", "/api/boards")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(boards), 1)

        status, cards = self._request("GET", "/api/cards")
        self.assertEqual(status, 200)
        self.assertIn(card_id, [c["id"] for c in cards])

        status, updated = self._request("PUT", f"/api/boards/{board_id}", {"name": "Renamed Board"})
        self.assertEqual(status, 200)
        self.assertEqual(updated["name"], "Renamed Board")

        status, _ = self._request("PUT", f"/api/boards/{board_id}", {})
        self.assertEqual(status, 400)

    def test_unknown_route_returns_404(self):
        status, body = self._request("GET", "/api/does-not-exist")
        self.assertEqual(status, 404)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()