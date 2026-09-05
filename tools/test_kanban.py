"""Unit and API tests for the local kanban store.

Run from the repo root:

    python3 tools/test_kanban.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from threading import Thread
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kanban import (  # noqa: E402
    DEFAULT_COLUMNS,
    PRIORITIES,
    KanbanError,
    KanbanStore,
    NotFoundError,
)
from kanban_api import (  # noqa: E402
    handle_request,
    serve_kanban,
)


def _request(
    store: KanbanStore,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return handle_request(store, method, path, body)


class KanbanStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KanbanStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_create_board_seeds_default_columns(self) -> None:
        board = self.store.create_board("Alpha", project_id="proj-1")
        self.assertEqual(board["name"], "Alpha")
        self.assertEqual(board["project_id"], "proj-1")
        self.assertEqual(len(board["columns"]), len(DEFAULT_COLUMNS))
        names = [column["name"] for column in board["columns"]]
        self.assertEqual(names, [column["name"] for column in DEFAULT_COLUMNS])
        for column in board["columns"]:
            self.assertEqual(column["cards"], [])

    def test_list_boards_returns_created_boards(self) -> None:
        self.store.create_board("One")
        self.store.create_board("Two")
        boards = self.store.list_boards()
        self.assertEqual({board["name"] for board in boards}, {"One", "Two"})

    def test_get_board_includes_cards(self) -> None:
        board = self.store.create_board("Work")
        column_id = board["columns"][0]["id"]
        card = self.store.create_card(column_id, title="Ship kanban")
        loaded = self.store.get_board(board["id"])
        self.assertEqual(loaded["columns"][0]["cards"][0]["id"], card["id"])
        self.assertEqual(loaded["columns"][0]["cards"][0]["title"], "Ship kanban")

    def test_create_card_rejects_invalid_priority(self) -> None:
        board = self.store.create_board("Work")
        column_id = board["columns"][0]["id"]
        with self.assertRaises(KanbanError):
            self.store.create_card(column_id, title="Bad", priority="P9")

    def test_create_card_defaults_priority(self) -> None:
        board = self.store.create_board("Work")
        column_id = board["columns"][0]["id"]
        card = self.store.create_card(column_id, title="Default prio")
        self.assertEqual(card["priority"], "P2")
        self.assertIn(card["priority"], PRIORITIES)

    def test_move_card_reorders_within_and_across_columns(self) -> None:
        board = self.store.create_board("Work")
        backlog_id = board["columns"][0]["id"]
        progress_id = board["columns"][1]["id"]
        first = self.store.create_card(backlog_id, title="A")
        second = self.store.create_card(backlog_id, title="B")
        third = self.store.create_card(backlog_id, title="C")

        self.store.move_card(third["id"], backlog_id, 0)
        titles = [
            card["title"]
            for card in self.store.get_board(board["id"])["columns"][0]["cards"]
        ]
        self.assertEqual(titles, ["C", "A", "B"])

        moved = self.store.move_card(second["id"], progress_id, 0)
        self.assertEqual(moved["column_id"], progress_id)
        loaded = self.store.get_board(board["id"])
        backlog_titles = [card["title"] for card in loaded["columns"][0]["cards"]]
        progress_titles = [card["title"] for card in loaded["columns"][1]["cards"]]
        self.assertEqual(backlog_titles, ["C", "A"])
        self.assertEqual(progress_titles, ["B"])
        self.assertEqual(first["id"], loaded["columns"][0]["cards"][1]["id"])

    def test_delete_column_cascades_cards(self) -> None:
        board = self.store.create_board("Work")
        column_id = board["columns"][0]["id"]
        card = self.store.create_card(column_id, title="Gone")
        self.store.delete_column(column_id)
        with self.assertRaises(NotFoundError):
            self.store.get_card(card["id"])

    def test_unknown_board_raises_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.store.get_board("missing")

    def test_board_costs_aggregate_estimate_and_actual(self) -> None:
        board = self.store.create_board("Work")
        backlog_id = board["columns"][0]["id"]
        done_id = board["columns"][3]["id"]
        self.store.create_card(
            backlog_id,
            title="Estimate only",
            estimate_tokens=1000,
            estimate_cost=0.5,
        )
        done = self.store.create_card(
            done_id,
            title="Finished",
            estimate_tokens=2000,
            estimate_cost=1.0,
            actual_tokens=1800,
            actual_cost=0.9,
        )
        self.store.update_card(done["id"], status="done")
        costs = self.store.board_costs(board["id"])
        self.assertAlmostEqual(costs["total_estimate_cost"], 1.5)
        self.assertAlmostEqual(costs["total_actual_cost"], 0.9)
        self.assertEqual(costs["total_estimate_tokens"], 3000)
        self.assertEqual(costs["total_actual_tokens"], 1800)
        by_name = {column["name"]: column for column in costs["columns"]}
        self.assertAlmostEqual(by_name["Backlog"]["estimate_cost"], 0.5)
        self.assertAlmostEqual(by_name["Done"]["actual_cost"], 0.9)

    def test_file_db_survives_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "kanban.sqlite")
            store = KanbanStore(path)
            board = store.create_board("Persisted")
            store.close()
            reopened = KanbanStore(path)
            loaded = reopened.get_board(board["id"])
            self.assertEqual(loaded["name"], "Persisted")
            reopened.close()


class KanbanApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KanbanStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_list_and_create_boards(self) -> None:
        status, listed = _request(self.store, "GET", "/api/boards")
        self.assertEqual(status, 200)
        self.assertEqual(listed, [])

        status, created = _request(
            self.store,
            "POST",
            "/api/boards",
            {"name": "Roadmap", "project_id": "p1"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["name"], "Roadmap")
        self.assertEqual(len(created["columns"]), 4)

        status, listed = _request(self.store, "GET", "/api/boards")
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

    def test_get_board_with_columns_and_cards(self) -> None:
        _, board = _request(self.store, "POST", "/api/boards", {"name": "B"})
        column_id = board["columns"][0]["id"]
        status, card = _request(
            self.store,
            "POST",
            f"/api/columns/{column_id}/cards",
            {"title": "Write tests", "priority": "P0"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(card["priority"], "P0")

        status, loaded = _request(self.store, "GET", f"/api/boards/{board['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(loaded["columns"][0]["cards"][0]["title"], "Write tests")

    def test_column_crud(self) -> None:
        _, board = _request(self.store, "POST", "/api/boards", {"name": "B"})
        status, column = _request(
            self.store,
            "POST",
            f"/api/boards/{board['id']}/columns",
            {"name": "Blocked", "color": "#ef4444"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(column["name"], "Blocked")
        self.assertEqual(column["position"], 4)

        status, updated = _request(
            self.store,
            "PATCH",
            f"/api/columns/{column['id']}",
            {"name": "Waiting", "position": 1, "color": "#a855f7"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["name"], "Waiting")
        self.assertEqual(updated["position"], 1)
        self.assertEqual(updated["color"], "#a855f7")

        status, _ = _request(self.store, "DELETE", f"/api/columns/{column['id']}")
        self.assertEqual(status, 204)

    def test_card_update_delete_and_move(self) -> None:
        _, board = _request(self.store, "POST", "/api/boards", {"name": "B"})
        backlog_id = board["columns"][0]["id"]
        review_id = board["columns"][2]["id"]
        _, card = _request(
            self.store,
            "POST",
            f"/api/columns/{backlog_id}/cards",
            {
                "title": "Implement API",
                "description": "CRUD",
                "assignee": "agent",
                "estimate_cost": 2.5,
            },
        )
        status, updated = _request(
            self.store,
            "PATCH",
            f"/api/cards/{card['id']}",
            {
                "title": "Implement API v1",
                "status": "in_progress",
                "actual_cost": 1.1,
                "actual_tokens": 900,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["title"], "Implement API v1")
        self.assertEqual(updated["status"], "in_progress")
        self.assertAlmostEqual(updated["actual_cost"], 1.1)

        status, moved = _request(
            self.store,
            "POST",
            f"/api/cards/{card['id']}/move",
            {"column_id": review_id, "position": 0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(moved["column_id"], review_id)

        status, _ = _request(self.store, "DELETE", f"/api/cards/{card['id']}")
        self.assertEqual(status, 204)
        status, payload = _request(
            self.store, "PATCH", f"/api/cards/{card['id']}", {"title": "nope"}
        )
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_board_costs_endpoint(self) -> None:
        _, board = _request(self.store, "POST", "/api/boards", {"name": "B"})
        column_id = board["columns"][0]["id"]
        _request(
            self.store,
            "POST",
            f"/api/columns/{column_id}/cards",
            {"title": "A", "estimate_cost": 3, "actual_cost": 1},
        )
        status, costs = _request(
            self.store, "GET", f"/api/boards/{board['id']}/costs"
        )
        self.assertEqual(status, 200)
        self.assertAlmostEqual(costs["total_estimate_cost"], 3)
        self.assertAlmostEqual(costs["total_actual_cost"], 1)
        self.assertEqual(costs["board_id"], board["id"])

    def test_missing_routes_and_validation(self) -> None:
        status, payload = _request(self.store, "GET", "/api/boards/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

        status, payload = _request(self.store, "POST", "/api/boards", {})
        self.assertEqual(status, 400)

        _, board = _request(self.store, "POST", "/api/boards", {"name": "B"})
        column_id = board["columns"][0]["id"]
        status, payload = _request(
            self.store,
            "POST",
            f"/api/columns/{column_id}/cards",
            {"title": "x", "priority": "high"},
        )
        self.assertEqual(status, 400)

        status, payload = _request(self.store, "GET", "/api/unknown")
        self.assertEqual(status, 404)


class KanbanHttpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KanbanStore(":memory:")
        self.server = serve_kanban("127.0.0.1", 0, store=self.store)
        self.port = self.server.server_address[1]
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.store.close()

    def _http(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int, Any]:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        parsed = json.loads(raw) if raw else None
        return response.status, parsed

    def test_http_roundtrip(self) -> None:
        status, board = self._http("POST", "/api/boards", {"name": "HTTP"})
        self.assertEqual(status, 201)
        status, listed = self._http("GET", "/api/boards")
        self.assertEqual(status, 200)
        self.assertEqual(listed[0]["id"], board["id"])


if __name__ == "__main__":
    unittest.main()
