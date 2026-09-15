"""
tools/kanban.py - Local Kanban SQLite Store and REST API for Agent Canvas.
"""

from contextlib import contextmanager
import json
import sqlite3
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

DEFAULT_COLUMNS = ["Backlog", "In Progress", "Review", "Done"]
ALLOWED_PRIORITIES = {"P0", "P1", "P2", "P3"}


class KanbanStore:
    def __init__(self, db_path: str = "kanban.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS boards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS columns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    board_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(board_id) REFERENCES boards(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    column_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    priority TEXT DEFAULT 'P2',
                    assignee TEXT DEFAULT '',
                    branch TEXT DEFAULT '',
                    pr TEXT DEFAULT '',
                    estimated_cost REAL DEFAULT 0.0,
                    actual_cost REAL DEFAULT 0.0,
                    position INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(column_id) REFERENCES columns(id) ON DELETE CASCADE
                );
                """
            )

    # --- Board Methods ---

    def create_board(self, name: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO boards (name) VALUES (?)", (name.strip(),))
            board_id = cur.lastrowid

            for idx, col_name in enumerate(DEFAULT_COLUMNS):
                cur.execute(
                    "INSERT INTO columns (board_id, name, position) VALUES (?, ?, ?)",
                    (board_id, col_name, idx),
                )
            conn.commit()

        return self.get_board(board_id)

    def get_board(self, board_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM boards WHERE id = ?", (board_id,))
            board_row = cur.fetchone()
            if not board_row:
                return None
            board = dict(board_row)

            cur.execute("SELECT * FROM columns WHERE board_id = ? ORDER BY position ASC", (board_id,))
            columns = [dict(c) for c in cur.fetchall()]

            for col in columns:
                cur.execute("SELECT * FROM cards WHERE column_id = ? ORDER BY position ASC", (col["id"],))
                col["cards"] = [dict(card) for card in cur.fetchall()]

            board["columns"] = columns
            return board

    def list_boards(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM boards ORDER BY id DESC")
            return [dict(r) for r in cur.fetchall()]

    def delete_board(self, board_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM boards WHERE id = ?", (board_id,))
            conn.commit()
            return cur.rowcount > 0

    # --- Column Methods ---

    def create_column(self, board_id: int, name: str, position: Optional[int] = None) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if position is None:
                cur.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM columns WHERE board_id = ?", (board_id,))
                position = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO columns (board_id, name, position) VALUES (?, ?, ?)",
                (board_id, name.strip(), position),
            )
            col_id = cur.lastrowid
            conn.commit()
            cur.execute("SELECT * FROM columns WHERE id = ?", (col_id,))
            return dict(cur.fetchone())

    def delete_column(self, column_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM columns WHERE id = ?", (column_id,))
            conn.commit()
            return cur.rowcount > 0

    # --- Card Methods ---

    def create_card(
        self,
        column_id: int,
        title: str,
        description: str = "",
        priority: str = "P2",
        assignee: str = "",
        branch: str = "",
        pr: str = "",
        estimated_cost: float = 0.0,
        actual_cost: float = 0.0,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        if priority not in ALLOWED_PRIORITIES:
            priority = "P2"

        with self._get_connection() as conn:
            cur = conn.cursor()
            if position is None:
                cur.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM cards WHERE column_id = ?", (column_id,))
                position = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO cards (
                    column_id, title, description, priority, assignee,
                    branch, pr, estimated_cost, actual_cost, position
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    column_id,
                    title.strip(),
                    description,
                    priority,
                    assignee,
                    branch,
                    pr,
                    float(estimated_cost),
                    float(actual_cost),
                    position,
                ),
            )
            card_id = cur.lastrowid
            conn.commit()
            cur.execute("SELECT * FROM cards WHERE id = ?", (card_id,))
            return dict(cur.fetchone())

    def get_card(self, card_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM cards WHERE id = ?", (card_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_card(self, card_id: int, **fields) -> Optional[Dict[str, Any]]:
        allowed = {
            "title", "description", "priority", "assignee",
            "branch", "pr", "estimated_cost", "actual_cost", "position"
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_card(card_id)

        if "priority" in updates and updates["priority"] not in ALLOWED_PRIORITIES:
            updates["priority"] = "P2"

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        params = list(updates.values())
        params.append(card_id)

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE cards SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                params,
            )
            conn.commit()

        return self.get_card(card_id)

    def move_card(self, card_id: int, target_column_id: int, position: Optional[int] = None) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if position is None:
                cur.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM cards WHERE column_id = ?", (target_column_id,))
                position = cur.fetchone()[0]

            cur.execute(
                "UPDATE cards SET column_id = ?, position = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (target_column_id, position, card_id),
            )
            conn.commit()

        return self.get_card(card_id)

    def delete_card(self, card_id: int) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            conn.commit()
            return cur.rowcount > 0

    # --- Cost Aggregation ---

    def get_board_cost_aggregates(self, board_id: int) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(cards.id) as total_cards,
                    COALESCE(SUM(cards.estimated_cost), 0.0) as total_estimated_cost,
                    COALESCE(SUM(cards.actual_cost), 0.0) as total_actual_cost
                FROM columns
                LEFT JOIN cards ON columns.id = cards.column_id
                WHERE columns.board_id = ?
                """,
                (board_id,),
            )
            summary = dict(cur.fetchone())

            cur.execute(
                """
                SELECT
                    columns.id as column_id,
                    columns.name as column_name,
                    COUNT(cards.id) as card_count,
                    COALESCE(SUM(cards.estimated_cost), 0.0) as estimated_cost,
                    COALESCE(SUM(cards.actual_cost), 0.0) as actual_cost
                FROM columns
                LEFT JOIN cards ON columns.id = cards.column_id
                WHERE columns.board_id = ?
                GROUP BY columns.id
                ORDER BY columns.position ASC
                """,
                (board_id,),
            )
            columns_breakdown = [dict(r) for r in cur.fetchall()]

            summary["board_id"] = board_id
            summary["by_column"] = columns_breakdown
            return summary


# --- HTTP REST Request Handler ---

class KanbanRequestHandler(BaseHTTPRequestHandler):
    store: KanbanStore = None

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        parts = [p for p in urllib.parse.urlparse(self.path).path.strip("/").split("/") if p]
        if parts == ["api", "boards"]:
            self._send_json(self.store.list_boards())
            return
        if len(parts) == 3 and parts[:2] == ["api", "boards"] and parts[2].isdigit():
            board = self.store.get_board(int(parts[2]))
            if board:
                self._send_json(board)
            else:
                self._send_json({"error": "Board not found"}, 404)
            return
        if len(parts) == 4 and parts[:2] == ["api", "boards"] and parts[2].isdigit() and parts[3] == "costs":
            self._send_json(self.store.get_board_cost_aggregates(int(parts[2])))
            return
        if len(parts) == 3 and parts[:2] == ["api", "cards"] and parts[2].isdigit():
            card = self.store.get_card(int(parts[2]))
            if card:
                self._send_json(card)
            else:
                self._send_json({"error": "Card not found"}, 404)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parts = [p for p in urllib.parse.urlparse(self.path).path.strip("/").split("/") if p]
        payload = self._read_json()

        if parts == ["api", "boards"]:
            name = payload.get("name", "New Board")
            board = self.store.create_board(name)
            self._send_json(board, 201)
            return

        if len(parts) == 4 and parts[:2] == ["api", "boards"] and parts[2].isdigit() and parts[3] == "columns":
            board_id = int(parts[2])
            name = payload.get("name", "New Column")
            position = payload.get("position")
            col = self.store.create_column(board_id, name, position)
            self._send_json(col, 201)
            return

        if len(parts) == 4 and parts[:2] == ["api", "columns"] and parts[2].isdigit() and parts[3] == "cards":
            col_id = int(parts[2])
            card = self.store.create_card(
                column_id=col_id,
                title=payload.get("title", "Untitled"),
                description=payload.get("description", ""),
                priority=payload.get("priority", "P2"),
                assignee=payload.get("assignee", ""),
                branch=payload.get("branch", ""),
                pr=payload.get("pr", ""),
                estimated_cost=payload.get("estimated_cost", 0.0),
                actual_cost=payload.get("actual_cost", 0.0),
                position=payload.get("position"),
            )
            self._send_json(card, 201)
            return

        if len(parts) == 4 and parts[:2] == ["api", "cards"] and parts[2].isdigit() and parts[3] == "move":
            card_id = int(parts[2])
            target_col = payload.get("column_id")
            position = payload.get("position")
            if target_col is None:
                self._send_json({"error": "column_id is required"}, 400)
                return
            card = self.store.move_card(card_id, int(target_col), position)
            if card:
                self._send_json(card)
            else:
                self._send_json({"error": "Card not found"}, 404)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_PUT(self):
        parts = [p for p in urllib.parse.urlparse(self.path).path.strip("/").split("/") if p]
        payload = self._read_json()

        if len(parts) == 3 and parts[:2] == ["api", "cards"] and parts[2].isdigit():
            card = self.store.update_card(int(parts[2]), **payload)
            if card:
                self._send_json(card)
            else:
                self._send_json({"error": "Card not found"}, 404)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_DELETE(self):
        parts = [p for p in urllib.parse.urlparse(self.path).path.strip("/").split("/") if p]

        if len(parts) == 3 and parts[:2] == ["api", "boards"] and parts[2].isdigit():
            if self.store.delete_board(int(parts[2])):
                self._send_json({"status": "deleted"})
            else:
                self._send_json({"error": "Board not found"}, 404)
            return

        if len(parts) == 3 and parts[:2] == ["api", "cards"] and parts[2].isdigit():
            if self.store.delete_card(int(parts[2])):
                self._send_json({"status": "deleted"})
            else:
                self._send_json({"error": "Card not found"}, 404)
            return

        self._send_json({"error": "Not found"}, 404)


def create_server(host: str = "127.0.0.1", port: int = 8080, db_path: str = "kanban.db") -> HTTPServer:
    KanbanRequestHandler.store = KanbanStore(db_path)
    return HTTPServer((host, port), KanbanRequestHandler)


if __name__ == "__main__":
    server = create_server()
    print("Starting Kanban REST API on http://127.0.0.1:8080 ...")
    server.serve_forever()