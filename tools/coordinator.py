"""Manager/worker coordination, dependency chains, and model load balancing."""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any

from cost_estimator import MODEL_PRICES
from fleet import FleetStore
from kanban import KanbanStore, new_id, utc_now
from pr_creator import read_worktree_file

COORDINATOR_DB_FILENAME = "coordinator.sqlite"
STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_COMPLETE = "complete"
STEP_STATUSES = (STEP_PENDING, STEP_RUNNING, STEP_COMPLETE)
COMPLETE_COLUMNS = ("Review", "Done")
COMPLETE_CARD_STATUSES = ("review", "done")
DEFAULT_TASK_TYPE = "code"
TASK_TYPES = ("docs", "code", "test", "refactor")
WORKER_AGENTS: tuple[dict[str, Any], ...] = (
    {"id": "glm-5.2", "model": "glm-5.2", "task_types": ("docs", "test")},
    {"id": "gpt-4o", "model": "gpt-4o", "task_types": ("docs", "code", "test")},
    {
        "id": "claude-sonnet",
        "model": "claude-sonnet",
        "task_types": ("code", "test", "refactor"),
    },
    {
        "id": "claude-opus",
        "model": "claude-opus",
        "task_types": ("code", "refactor"),
    },
)


class CoordinatorError(Exception):
    """Raised for invalid coordinator operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(CoordinatorError):
    """Raised when a sequence does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, COORDINATOR_DB_FILENAME)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def model_cost_rank(model: str) -> float:
    prices = MODEL_PRICES[model]
    return float(prices["input"]) + float(prices["output"])


def list_worker_agents() -> list[dict[str, Any]]:
    agents = []
    for agent in WORKER_AGENTS:
        agents.append(
            {
                **agent,
                "task_types": list(agent["task_types"]),
                "cost_rank": model_cost_rank(agent["model"]),
            }
        )
    agents.sort(key=lambda item: item["cost_rank"])
    return agents


def cheapest_capable_model(task_type: str) -> str:
    task_type = (task_type or DEFAULT_TASK_TYPE).strip() or DEFAULT_TASK_TYPE
    capable = [
        agent
        for agent in list_worker_agents()
        if task_type in agent["task_types"]
    ]
    if not capable:
        raise CoordinatorError(f"No worker capable of task type {task_type!r}")
    return capable[0]["model"]


def read_shared_file(
    project_store: Any,
    project_id: str,
    relative_path: str,
    from_branch: str | None = None,
) -> str:
    worktrees = project_store.list_worktrees(project_id)
    if from_branch:
        worktrees = [
            tree for tree in worktrees if tree["branch_name"] == from_branch
        ]
    if not worktrees:
        raise CoordinatorError("No readable worktrees in this project")
    return read_worktree_file(worktrees[0]["path"], relative_path)


class CoordinatorStore:
    """Sequential agent chains and worker assignment."""

    def __init__(
        self,
        db_path: str = ":memory:",
        kanban_store: KanbanStore | None = None,
        fleet_store: FleetStore | None = None,
        project_store: Any | None = None,
    ) -> None:
        self.db_path = db_path
        self.kanban_store = kanban_store
        self.fleet_store = fleet_store
        self.project_store = project_store
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
            CREATE TABLE IF NOT EXISTS sequences (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS steps (
                id TEXT PRIMARY KEY,
                sequence_id TEXT NOT NULL,
                card_id TEXT NOT NULL,
                depends_on_card_id TEXT,
                session_id TEXT,
                status TEXT NOT NULL,
                position INTEGER NOT NULL,
                FOREIGN KEY (sequence_id) REFERENCES sequences(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def create_sequence(
        self,
        project_id: str,
        card_ids: list[str],
        name: str | None = None,
    ) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        if not project_id:
            raise CoordinatorError("project_id is required")
        ids = [str(card_id).strip() for card_id in (card_ids or []) if str(card_id).strip()]
        if len(ids) < 1:
            raise CoordinatorError("card_ids must contain at least one card")
        sequence_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO sequences (id, project_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sequence_id, project_id, name, now, now),
            )
            previous = None
            for position, card_id in enumerate(ids):
                self.conn.execute(
                    """
                    INSERT INTO steps (
                        id, sequence_id, card_id, depends_on_card_id,
                        session_id, status, position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id(),
                        sequence_id,
                        card_id,
                        previous,
                        None,
                        STEP_PENDING,
                        position,
                    ),
                )
                previous = card_id
            self.conn.commit()
        return self.get_sequence(sequence_id)

    def list_sequences(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM sequences ORDER BY created_at ASC"
            ).fetchall()
        return [self.get_sequence(row["id"]) for row in rows]

    def get_sequence(self, sequence_id: str) -> dict[str, Any]:
        with self._lock:
            sequence = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM sequences WHERE id = ?", (sequence_id,)
                ).fetchone()
            )
            if sequence is None:
                raise NotFoundError(f"Sequence {sequence_id} not found")
            steps = self.conn.execute(
                """
                SELECT * FROM steps
                WHERE sequence_id = ?
                ORDER BY position ASC
                """,
                (sequence_id,),
            ).fetchall()
        sequence["steps"] = [_row_to_dict(row) for row in steps]
        sequence["dependency_graph"] = self._graph(sequence["steps"])
        return sequence

    def advance_sequence(self, sequence_id: str) -> dict[str, Any] | None:
        sequence = self.get_sequence(sequence_id)
        self._sync_completed_steps(sequence)
        sequence = self.get_sequence(sequence_id)
        for step in sequence["steps"]:
            if step["status"] != STEP_PENDING:
                continue
            if step["depends_on_card_id"] and not self._card_complete(
                sequence["steps"], step["depends_on_card_id"]
            ):
                continue
            session = self._spawn(sequence["project_id"], step["card_id"])
            now = utc_now()
            with self._lock:
                self.conn.execute(
                    """
                    UPDATE steps
                    SET status = ?, session_id = ?
                    WHERE id = ?
                    """,
                    (STEP_RUNNING, session["id"], step["id"]),
                )
                self.conn.execute(
                    "UPDATE sequences SET updated_at = ? WHERE id = ?",
                    (now, sequence_id),
                )
                self.conn.commit()
            sequence = self.get_sequence(sequence_id)
            return next(
                item for item in sequence["steps"] if item["id"] == step["id"]
            )
        return None

    def assign_card(
        self,
        project_id: str,
        card_id: str,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        if self.fleet_store is None:
            raise CoordinatorError("fleet store is required to assign workers")
        model = cheapest_capable_model(task_type or DEFAULT_TASK_TYPE)
        return self.fleet_store.spawn_session(
            project_id, card_id, model=model
        )

    def _spawn(self, project_id: str, card_id: str) -> dict[str, Any]:
        if self.fleet_store is None:
            raise CoordinatorError("fleet store is required to advance sequences")
        return self.fleet_store.spawn_session(project_id, card_id)

    def _sync_completed_steps(self, sequence: dict[str, Any]) -> None:
        for step in sequence["steps"]:
            if step["status"] != STEP_RUNNING:
                continue
            if self._card_is_review_or_done(step["card_id"]):
                with self._lock:
                    self.conn.execute(
                        "UPDATE steps SET status = ? WHERE id = ?",
                        (STEP_COMPLETE, step["id"]),
                    )
                    self.conn.commit()

    def _card_complete(self, steps: list[dict[str, Any]], card_id: str) -> bool:
        for step in steps:
            if step["card_id"] == card_id and step["status"] == STEP_COMPLETE:
                return True
        return self._card_is_review_or_done(card_id)

    def _card_is_review_or_done(self, card_id: str) -> bool:
        if self.kanban_store is None:
            return False
        card = self.kanban_store.get_card(card_id)
        if card.get("status") in COMPLETE_CARD_STATUSES:
            return True
        column = self.kanban_store.get_column(card["column_id"])
        return column["name"] in COMPLETE_COLUMNS

    def _graph(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = [
            {
                "card_id": step["card_id"],
                "status": step["status"],
                "session_id": step["session_id"],
            }
            for step in steps
        ]
        edges = [
            {"from": step["depends_on_card_id"], "to": step["card_id"]}
            for step in steps
            if step["depends_on_card_id"]
        ]
        return {"nodes": nodes, "edges": edges}
