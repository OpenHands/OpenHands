"""Fleet session management and per-project cost caps.

Sessions live in a SQLite file beside the kanban/projects stores under
``~/.openhands/agent-canvas/``. Spawning an agent for a kanban card
optionally creates a git worktree and links the card to the session.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from kanban import KanbanStore, utc_now, new_id
from kanban_agent import link_session
from pr_creator import agent_branch_name, slugify

FLEET_DB_FILENAME = "fleet.sqlite"
ACTIVE_STATUS = "active"
DONE_STATUS = "done"
STOPPED_STATUS = "stopped"
ARCHIVED_STATUS = "archived"
SESSION_STATUSES = (ACTIVE_STATUS, DONE_STATUS, STOPPED_STATUS, ARCHIVED_STATUS)
RESET_PERIODS = ("never", "daily", "weekly", "monthly")
DEFAULT_RESET_PERIOD = "never"
DEFAULT_MODEL = "claude-sonnet"
SESSION_PATCH_FIELDS = ("status", "model", "duration_seconds", "cost", "branch_name")
PERIOD_DELTAS = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
    "monthly": timedelta(days=30),
}


class FleetError(Exception):
    """Raised for invalid fleet operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(FleetError):
    """Raised when a session or cost cap does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


class CapExceededError(FleetError):
    """Raised when a project cost cap blocks a new session."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=409)


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, FLEET_DB_FILENAME)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


class FleetStore:
    """CRUD store for fleet sessions and cost caps."""

    def __init__(
        self,
        db_path: str = ":memory:",
        kanban_store: KanbanStore | None = None,
        project_store: Any | None = None,
    ) -> None:
        self.db_path = db_path
        self.kanban_store = kanban_store
        self.project_store = project_store
        self._lock = threading.RLock()  # ponytail: nested helpers share this lock
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                card_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                status TEXT NOT NULL,
                model TEXT,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0,
                agent_session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cost_caps (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL UNIQUE,
                cap_usd REAL NOT NULL,
                used_usd REAL NOT NULL DEFAULT 0,
                reset_period TEXT NOT NULL,
                last_reset_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def spawn_session(
        self,
        project_id: str,
        card_id: str,
        model: str | None = None,
        agent_session_id: str | None = None,
        branch_name: str | None = None,
    ) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        card_id = (card_id or "").strip()
        if not project_id:
            raise FleetError("project_id is required")
        if not card_id:
            raise FleetError("card_id is required")
        card = self._load_card(card_id)
        pending = float(card.get("actual_cost") or card.get("estimate_cost") or 0)
        self.assert_within_cap(project_id, pending_cost=pending)
        session_id = new_id()
        agent_session_id = (agent_session_id or "").strip() or new_id()
        model = (model or "").strip() or (card.get("model_used") if card else None) or DEFAULT_MODEL
        title = str(card.get("title") or "task") if card else "task"
        branch_name = (branch_name or "").strip() or agent_branch_name(
            agent_session_id, slugify(title)
        )
        cost = float(card.get("actual_cost") or 0) if card else 0.0
        duration = int(card.get("agent_time") or 0) if card else 0
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO sessions (
                    id, project_id, card_id, branch_name, status, model,
                    duration_seconds, cost, agent_session_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    project_id,
                    card_id,
                    branch_name,
                    ACTIVE_STATUS,
                    model,
                    duration,
                    cost,
                    agent_session_id,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        if self.project_store is not None:
            worktree = self.project_store.create_worktree(
                project_id, branch_name, status="working"
            )
            self.project_store.assign_worktree(
                project_id, worktree["id"], agent_session_id
            )
        if self.kanban_store is not None:
            link_session(self.kanban_store, card_id, agent_session_id)
            self.kanban_store.update_card(card_id, linked_branch=branch_name, model_used=model)
        self._refresh_cap_usage(project_id)
        return self.get_session(session_id)

    def list_sessions(
        self,
        project_id: str | None = None,
        status: str | None = None,
        model: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        values: list[Any] = []
        resolved_status = status if status is not None else ACTIVE_STATUS
        if resolved_status:
            clauses.append("status = ?")
            values.append(resolved_status)
        if project_id:
            clauses.append("project_id = ?")
            values.append(project_id)
        if model:
            clauses.append("model = ?")
            values.append(model)
        if created_after:
            clauses.append("created_at >= ?")
            values.append(created_after)
        if created_before:
            clauses.append("created_at <= ?")
            values.append(created_before)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM sessions {where} ORDER BY created_at ASC",
                values,
            ).fetchall()
        return [_row_to_dict(row) for row in rows]  # type: ignore[misc]

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
            )
        if session is None:
            raise NotFoundError(f"Session {session_id} not found")
        return session

    def update_session(self, session_id: str, **fields: Any) -> dict[str, Any]:
        session = self.get_session(session_id)
        updates: dict[str, Any] = {}
        for key in SESSION_PATCH_FIELDS:
            if key not in fields:
                continue
            value = fields[key]
            if key == "status":
                self._validate_status(value)
            if key == "cost" and value is not None:
                value = float(value)
            if key == "duration_seconds" and value is not None:
                value = int(value)
            updates[key] = value
        if "status" in updates and "duration_seconds" not in updates:
            updates["duration_seconds"] = self._elapsed(session["created_at"])
        if not updates:
            return session
        updates["updated_at"] = utc_now()
        with self._lock:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            self.conn.execute(
                f"UPDATE sessions SET {assignments} WHERE id = ?",
                (*updates.values(), session_id),
            )
            self.conn.commit()
        self._refresh_cap_usage(session["project_id"])
        return self.get_session(session_id)

    def archive_session(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        now = utc_now()
        duration = self._elapsed(session["created_at"])
        with self._lock:
            self.conn.execute(
                """
                UPDATE sessions
                SET status = ?, duration_seconds = ?, updated_at = ?
                WHERE id = ?
                """,
                (ARCHIVED_STATUS, duration, now, session_id),
            )
            self.conn.commit()
        self._refresh_cap_usage(session["project_id"])
        return self.get_session(session_id)

    def set_cost_cap(
        self,
        project_id: str,
        cap_usd: float,
        reset_period: str | None = None,
    ) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        if not project_id:
            raise FleetError("project_id is required")
        cap_usd = self._validate_cap(cap_usd)
        period = reset_period or DEFAULT_RESET_PERIOD
        if period not in RESET_PERIODS:
            raise FleetError(f"reset_period must be one of {', '.join(RESET_PERIODS)}")
        now = utc_now()
        used = self._used_usd(project_id)
        with self._lock:
            existing = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM cost_caps WHERE project_id = ?", (project_id,)
                ).fetchone()
            )
            if existing is None:
                cap_id = new_id()
                self.conn.execute(
                    """
                    INSERT INTO cost_caps (
                        id, project_id, cap_usd, used_usd, reset_period,
                        last_reset_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cap_id, project_id, cap_usd, used, period, now, now, now),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE cost_caps
                    SET cap_usd = ?, used_usd = ?, reset_period = ?, updated_at = ?
                    WHERE project_id = ?
                    """,
                    (cap_usd, used, period, now, project_id),
                )
            self.conn.commit()
        return self.get_cost_cap(project_id)

    def list_cost_caps(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT project_id FROM cost_caps ORDER BY created_at ASC"
            ).fetchall()
        return [self.get_cost_cap(row["project_id"]) for row in rows]

    def get_cost_cap(self, project_id: str) -> dict[str, Any]:
        self._maybe_reset_cap(project_id)
        with self._lock:
            cap = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM cost_caps WHERE project_id = ?", (project_id,)
                ).fetchone()
            )
        if cap is None:
            raise NotFoundError(f"Cost cap for project {project_id} not found")
        used = self._used_usd(project_id)
        cap["used_usd"] = used
        return cap

    def assert_within_cap(
        self, project_id: str, pending_cost: float = 0.0
    ) -> None:
        with self._lock:
            cap = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM cost_caps WHERE project_id = ?", (project_id,)
                ).fetchone()
            )
        if cap is None:
            return
        self._maybe_reset_cap(project_id)
        used = self._used_usd(project_id)
        cap_usd = float(cap["cap_usd"])
        if used >= cap_usd or used + float(pending_cost or 0) > cap_usd:
            raise CapExceededError(
                f"Project {project_id} cost cap ${cap_usd} exceeded"
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT
                    COUNT(*) AS total_sessions,
                    COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0) AS active_sessions,
                    COALESCE(SUM(cost), 0) AS total_cost,
                    COALESCE(SUM(duration_seconds), 0) AS total_duration_seconds
                FROM sessions
                """,
                (ACTIVE_STATUS,),
            ).fetchone()
        return {
            "total_sessions": int(row["total_sessions"]),
            "active_sessions": int(row["active_sessions"]),
            "total_cost": float(row["total_cost"]),
            "total_duration_seconds": int(row["total_duration_seconds"]),
        }

    def _load_card(self, card_id: str) -> dict[str, Any]:
        if self.kanban_store is None:
            return {}
        return self.kanban_store.get_card(card_id)

    def _used_usd(self, project_id: str) -> float:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT COALESCE(SUM(cost), 0) AS used
                FROM sessions
                WHERE project_id = ? AND status != ?
                """,
                (project_id, ARCHIVED_STATUS),
            ).fetchone()
        return float(row["used"])

    def _refresh_cap_usage(self, project_id: str) -> None:
        used = self._used_usd(project_id)
        with self._lock:
            self.conn.execute(
                "UPDATE cost_caps SET used_usd = ?, updated_at = ? WHERE project_id = ?",
                (used, utc_now(), project_id),
            )
            self.conn.commit()

    def _maybe_reset_cap(self, project_id: str) -> None:
        with self._lock:
            cap = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM cost_caps WHERE project_id = ?", (project_id,)
                ).fetchone()
            )
            if cap is None:
                return
            delta = PERIOD_DELTAS.get(cap["reset_period"])
            if delta is None:
                return
            last = _parse_ts(cap["last_reset_at"])
            if datetime.now(timezone.utc) - last < delta:
                return
            now = utc_now()
            self.conn.execute(
                """
                UPDATE cost_caps
                SET used_usd = 0, last_reset_at = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (now, now, project_id),
            )
            self.conn.commit()

    def _elapsed(self, created_at: str) -> int:
        start = _parse_ts(created_at)
        return max(0, int((datetime.now(timezone.utc) - start).total_seconds()))

    def _validate_status(self, status: Any) -> None:
        if status not in SESSION_STATUSES:
            raise FleetError(f"status must be one of {', '.join(SESSION_STATUSES)}")

    def _validate_cap(self, cap_usd: Any) -> float:
        try:
            value = float(cap_usd)
        except (TypeError, ValueError) as exc:
            raise FleetError("cap_usd must be a number") from exc
        if value < 0:
            raise FleetError("cap_usd must be >= 0")
        return value
