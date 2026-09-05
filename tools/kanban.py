"""Local kanban persistence backed by SQLite.

The agent-server already keeps process state on disk under
``~/.openhands/agent-canvas/``. Boards, columns, and cards live in a sibling
SQLite file so they survive restarts without a new database service.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

PRIORITIES = ("P0", "P1", "P2", "P3")
DEFAULT_PRIORITY = "P2"
DEFAULT_CARD_STATUS = "todo"

DEFAULT_COLUMNS: tuple[dict[str, str], ...] = (
    {"name": "Backlog", "color": "#6b7280"},
    {"name": "In Progress", "color": "#3b82f6"},
    {"name": "Review", "color": "#f59e0b"},
    {"name": "Done", "color": "#22c55e"},
)

KANBAN_DB_FILENAME = "kanban.sqlite"
CARD_PATCH_FIELDS = (
    "title",
    "description",
    "priority",
    "status",
    "assignee",
    "linked_branch",
    "linked_pr",
    "estimate_tokens",
    "estimate_cost",
    "actual_tokens",
    "actual_cost",
    "model_used",
    "tool_calls",
    "agent_time",
    "agent_session_id",
)
NUMERIC_CARD_FIELDS = {
    "estimate_tokens",
    "estimate_cost",
    "actual_tokens",
    "actual_cost",
    "tool_calls",
    "agent_time",
}


class KanbanError(Exception):
    """Raised for invalid kanban operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(KanbanError):
    """Raised when a board, column, or card does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, KANBAN_DB_FILENAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _hydrate_card(card: dict[str, Any]) -> dict[str, Any]:
    raw = card.get("activity_log") or "[]"
    if isinstance(raw, str):
        try:
            card["activity_log"] = json.loads(raw)
        except json.JSONDecodeError:
            card["activity_log"] = []
    elif not isinstance(card.get("activity_log"), list):
        card["activity_log"] = []
    return card


class KanbanStore:
    """CRUD store for boards, columns, and cards."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS boards (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                project_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS columns (
                id TEXT PRIMARY KEY,
                board_id TEXT NOT NULL,
                name TEXT NOT NULL,
                position INTEGER NOT NULL,
                color TEXT,
                FOREIGN KEY (board_id) REFERENCES boards(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY,
                column_id TEXT NOT NULL,
                board_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                assignee TEXT,
                linked_branch TEXT,
                linked_pr TEXT,
                estimate_tokens INTEGER,
                estimate_cost REAL,
                actual_tokens INTEGER,
                actual_cost REAL,
                model_used TEXT,
                tool_calls INTEGER,
                agent_time REAL,
                agent_session_id TEXT,
                position INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (column_id) REFERENCES columns(id) ON DELETE CASCADE,
                FOREIGN KEY (board_id) REFERENCES boards(id) ON DELETE CASCADE
            );
            """
        )
        try:
            self.conn.execute(
                "ALTER TABLE cards ADD COLUMN activity_log TEXT NOT NULL DEFAULT '[]'"
            )
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def create_board(
        self, name: str, project_id: str | None = None
    ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise KanbanError("Board name is required")
        board_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO boards (id, name, project_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (board_id, name, project_id, now, now),
            )
            for index, column in enumerate(DEFAULT_COLUMNS):
                self.conn.execute(
                    """
                    INSERT INTO columns (id, board_id, name, position, color)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (new_id(), board_id, column["name"], index, column["color"]),
                )
            self.conn.commit()
        return self.get_board(board_id)

    def list_boards(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM boards ORDER BY created_at ASC"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]  # type: ignore[misc]

    def get_board(self, board_id: str) -> dict[str, Any]:
        with self._lock:
            board = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM boards WHERE id = ?", (board_id,)
                ).fetchone()
            )
            if board is None:
                raise NotFoundError(f"Board {board_id} not found")
            columns = self.conn.execute(
                "SELECT * FROM columns WHERE board_id = ? ORDER BY position ASC",
                (board_id,),
            ).fetchall()
            cards = self.conn.execute(
                """
                SELECT * FROM cards
                WHERE board_id = ?
                ORDER BY position ASC, created_at ASC
                """,
                (board_id,),
            ).fetchall()
        cards_by_column: dict[str, list[dict[str, Any]]] = {}
        for card in cards:
            payload = _row_to_dict(card)
            assert payload is not None
            cards_by_column.setdefault(payload["column_id"], []).append(
                _hydrate_card(payload)
            )
        board["columns"] = []
        for column in columns:
            payload = _row_to_dict(column)
            assert payload is not None
            payload["cards"] = cards_by_column.get(payload["id"], [])
            board["columns"].append(payload)
        return board

    def add_column(
        self,
        board_id: str,
        name: str,
        color: str | None = None,
        position: int | None = None,
    ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise KanbanError("Column name is required")
        self.get_board(board_id)
        with self._lock:
            if position is None:
                row = self.conn.execute(
                    "SELECT COALESCE(MAX(position), -1) AS max_pos FROM columns WHERE board_id = ?",
                    (board_id,),
                ).fetchone()
                position = int(row["max_pos"]) + 1
            else:
                self._shift_column_positions(board_id, int(position), 1)
            column_id = new_id()
            self.conn.execute(
                """
                INSERT INTO columns (id, board_id, name, position, color)
                VALUES (?, ?, ?, ?, ?)
                """,
                (column_id, board_id, name, int(position), color),
            )
            self._touch_board(board_id)
            self.conn.commit()
        return self.get_column(column_id)

    def get_column(self, column_id: str) -> dict[str, Any]:
        with self._lock:
            column = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM columns WHERE id = ?", (column_id,)
                ).fetchone()
            )
        if column is None:
            raise NotFoundError(f"Column {column_id} not found")
        return column

    def update_column(
        self,
        column_id: str,
        name: str | None = None,
        position: int | None = None,
        color: str | None = None,
    ) -> dict[str, Any]:
        column = self.get_column(column_id)
        updates: dict[str, Any] = {}
        if name is not None:
            name = name.strip()
            if not name:
                raise KanbanError("Column name is required")
            updates["name"] = name
        if color is not None:
            updates["color"] = color
        with self._lock:
            if position is not None and int(position) != column["position"]:
                self._reposition_column(column, int(position))
            if updates:
                assignments = ", ".join(f"{key} = ?" for key in updates)
                self.conn.execute(
                    f"UPDATE columns SET {assignments} WHERE id = ?",
                    (*updates.values(), column_id),
                )
            self._touch_board(column["board_id"])
            self.conn.commit()
        return self.get_column(column_id)

    def delete_column(self, column_id: str) -> None:
        column = self.get_column(column_id)
        with self._lock:
            self.conn.execute("DELETE FROM columns WHERE id = ?", (column_id,))
            self._reindex_columns(column["board_id"])
            self._touch_board(column["board_id"])
            self.conn.commit()

    def create_card(
        self,
        column_id: str,
        title: str,
        **fields: Any,
    ) -> dict[str, Any]:
        title = (title or "").strip()
        if not title:
            raise KanbanError("Card title is required")
        column = self.get_column(column_id)
        priority = fields.get("priority", DEFAULT_PRIORITY)
        self._validate_priority(priority)
        card_id = new_id()
        now = utc_now()
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(position), -1) AS max_pos FROM cards WHERE column_id = ?",
                (column_id,),
            ).fetchone()
            position = int(row["max_pos"]) + 1
            self.conn.execute(
                """
                INSERT INTO cards (
                    id, column_id, board_id, title, description, priority, status,
                    assignee, linked_branch, linked_pr, estimate_tokens, estimate_cost,
                    actual_tokens, actual_cost, model_used, tool_calls, agent_time,
                    agent_session_id, position, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card_id,
                    column_id,
                    column["board_id"],
                    title,
                    fields.get("description"),
                    priority,
                    fields.get("status") or DEFAULT_CARD_STATUS,
                    fields.get("assignee"),
                    fields.get("linked_branch"),
                    fields.get("linked_pr"),
                    fields.get("estimate_tokens"),
                    fields.get("estimate_cost"),
                    fields.get("actual_tokens"),
                    fields.get("actual_cost"),
                    fields.get("model_used"),
                    fields.get("tool_calls"),
                    fields.get("agent_time"),
                    fields.get("agent_session_id"),
                    position,
                    now,
                    now,
                ),
            )
            self._touch_board(column["board_id"])
            self.conn.commit()
        return self.get_card(card_id)

    def get_card(self, card_id: str) -> dict[str, Any]:
        with self._lock:
            card = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM cards WHERE id = ?", (card_id,)
                ).fetchone()
            )
        if card is None:
            raise NotFoundError(f"Card {card_id} not found")
        return _hydrate_card(card)

    def append_activity(self, card_id: str, message: str) -> dict[str, Any]:
        card = self.get_card(card_id)
        log = list(card.get("activity_log") or [])
        log.append({"timestamp": utc_now(), "message": message})
        with self._lock:
            self.conn.execute(
                "UPDATE cards SET activity_log = ?, updated_at = ? WHERE id = ?",
                (json.dumps(log), utc_now(), card_id),
            )
            self._touch_board(card["board_id"])
            self.conn.commit()
        return self.get_card(card_id)

    def column_named(self, board_id: str, name: str) -> dict[str, Any]:
        board = self.get_board(board_id)
        for column in board["columns"]:
            if column["name"] == name:
                return column
        raise NotFoundError(f"Column {name!r} not found on board {board_id}")

    def update_card(self, card_id: str, **fields: Any) -> dict[str, Any]:
        card = self.get_card(card_id)
        updates: dict[str, Any] = {}
        for key in CARD_PATCH_FIELDS:
            if key not in fields:
                continue
            value = fields[key]
            if key == "title":
                value = (value or "").strip()
                if not value:
                    raise KanbanError("Card title is required")
            if key == "priority":
                self._validate_priority(value)
            if key in NUMERIC_CARD_FIELDS and value is not None:
                value = float(value) if key.endswith("cost") or key == "agent_time" else int(value)
            updates[key] = value
        if not updates:
            return card
        updates["updated_at"] = utc_now()
        with self._lock:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            self.conn.execute(
                f"UPDATE cards SET {assignments} WHERE id = ?",
                (*updates.values(), card_id),
            )
            self._touch_board(card["board_id"])
            self.conn.commit()
        return self.get_card(card_id)

    def delete_card(self, card_id: str) -> None:
        card = self.get_card(card_id)
        with self._lock:
            self.conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            self._reindex_cards(card["column_id"])
            self._touch_board(card["board_id"])
            self.conn.commit()

    def move_card(
        self, card_id: str, column_id: str, position: int
    ) -> dict[str, Any]:
        card = self.get_card(card_id)
        dest = self.get_column(column_id)
        if dest["board_id"] != card["board_id"]:
            raise KanbanError("Cannot move a card to a column on another board")
        position = max(0, int(position))
        with self._lock:
            if card["column_id"] != column_id:
                self.conn.execute(
                    "UPDATE cards SET column_id = ?, updated_at = ? WHERE id = ?",
                    (column_id, utc_now(), card_id),
                )
                self._reindex_cards(card["column_id"])
            self._place_card(column_id, card_id, position)
            self._touch_board(card["board_id"])
            self.conn.commit()
        return self.get_card(card_id)

    def board_costs(self, board_id: str) -> dict[str, Any]:
        board = self.get_board(board_id)
        columns = []
        total_estimate_cost = 0.0
        total_actual_cost = 0.0
        total_estimate_tokens = 0
        total_actual_tokens = 0
        for column in board["columns"]:
            estimate_cost = sum(
                float(card["estimate_cost"] or 0) for card in column["cards"]
            )
            actual_cost = sum(
                float(card["actual_cost"] or 0) for card in column["cards"]
            )
            estimate_tokens = sum(
                int(card["estimate_tokens"] or 0) for card in column["cards"]
            )
            actual_tokens = sum(
                int(card["actual_tokens"] or 0) for card in column["cards"]
            )
            columns.append(
                {
                    "id": column["id"],
                    "name": column["name"],
                    "estimate_cost": estimate_cost,
                    "actual_cost": actual_cost,
                    "estimate_tokens": estimate_tokens,
                    "actual_tokens": actual_tokens,
                }
            )
            total_estimate_cost += estimate_cost
            total_actual_cost += actual_cost
            total_estimate_tokens += estimate_tokens
            total_actual_tokens += actual_tokens
        return {
            "board_id": board_id,
            "total_estimate_cost": total_estimate_cost,
            "total_actual_cost": total_actual_cost,
            "total_estimate_tokens": total_estimate_tokens,
            "total_actual_tokens": total_actual_tokens,
            "columns": columns,
        }

    def _validate_priority(self, priority: Any) -> None:
        if priority not in PRIORITIES:
            raise KanbanError(
                f"Priority must be one of {', '.join(PRIORITIES)}"
            )

    def _touch_board(self, board_id: str) -> None:
        self.conn.execute(
            "UPDATE boards SET updated_at = ? WHERE id = ?",
            (utc_now(), board_id),
        )

    def _shift_column_positions(
        self, board_id: str, start: int, delta: int
    ) -> None:
        self.conn.execute(
            """
            UPDATE columns
            SET position = position + ?
            WHERE board_id = ? AND position >= ?
            """,
            (delta, board_id, start),
        )

    def _reposition_column(self, column: dict[str, Any], position: int) -> None:
        board_id = column["board_id"]
        current = int(column["position"])
        position = max(0, position)
        if position == current:
            return
        if position > current:
            self.conn.execute(
                """
                UPDATE columns
                SET position = position - 1
                WHERE board_id = ? AND position > ? AND position <= ?
                """,
                (board_id, current, position),
            )
        else:
            self.conn.execute(
                """
                UPDATE columns
                SET position = position + 1
                WHERE board_id = ? AND position >= ? AND position < ?
                """,
                (board_id, position, current),
            )
        self.conn.execute(
            "UPDATE columns SET position = ? WHERE id = ?",
            (position, column["id"]),
        )

    def _reindex_columns(self, board_id: str) -> None:
        rows = self.conn.execute(
            "SELECT id FROM columns WHERE board_id = ? ORDER BY position ASC",
            (board_id,),
        ).fetchall()
        for index, row in enumerate(rows):
            self.conn.execute(
                "UPDATE columns SET position = ? WHERE id = ?",
                (index, row["id"]),
            )

    def _reindex_cards(self, column_id: str) -> None:
        rows = self.conn.execute(
            "SELECT id FROM cards WHERE column_id = ? ORDER BY position ASC, created_at ASC",
            (column_id,),
        ).fetchall()
        for index, row in enumerate(rows):
            self.conn.execute(
                "UPDATE cards SET position = ? WHERE id = ?",
                (index, row["id"]),
            )

    def _place_card(self, column_id: str, card_id: str, position: int) -> None:
        rows = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM cards WHERE column_id = ? ORDER BY position ASC, created_at ASC",
                (column_id,),
            ).fetchall()
            if row["id"] != card_id
        ]
        position = min(position, len(rows))
        rows.insert(position, card_id)
        now = utc_now()
        for index, item_id in enumerate(rows):
            self.conn.execute(
                "UPDATE cards SET position = ?, updated_at = ? WHERE id = ?",
                (index, now, item_id),
            )
