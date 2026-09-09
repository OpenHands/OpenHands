"""Append-only SQLite audit log for standards runs and violations."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from kanban import new_id, utc_now
from standards import ACTION_WARN, SEVERITY_WARNING, Violation

STANDARDS_DB_FILENAME = "standards.sqlite"
DEFAULT_AUDIT_LIMIT = 50
MAX_AUDIT_LIMIT = 200
CONFIG_KEY = "config"


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, STANDARDS_DB_FILENAME)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class AuditStore:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_all()

    def create_all(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_runs (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                worktree TEXT,
                enforcement TEXT,
                duration_ms INTEGER,
                status TEXT,
                summary TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_violations (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                plugin_name TEXT,
                rule_id TEXT,
                severity TEXT,
                file TEXT,
                line INTEGER,
                message TEXT,
                remediation TEXT,
                action TEXT,
                FOREIGN KEY (run_id) REFERENCES audit_runs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_audit_violations_run
                ON audit_violations(run_id, created_at DESC, id DESC);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_persisted_config(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT value FROM kv WHERE key = ?", (CONFIG_KEY,)
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["value"])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def save_persisted_config(self, config: dict[str, Any]) -> None:
        payload = json.dumps(config)
        with self._lock:
            self.conn.execute(
                "INSERT INTO kv(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (CONFIG_KEY, payload),
            )
            self.conn.commit()

    def write_run(
        self,
        *,
        run_id: str,
        started_at: str,
        worktree: str,
        enforcement: dict[str, Any],
        duration_ms: int,
        status: str,
        summary: dict[str, Any],
    ) -> None:
        try:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO audit_runs(id, started_at, worktree, enforcement, "
                    "duration_ms, status, summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        started_at,
                        worktree,
                        json.dumps(enforcement),
                        duration_ms,
                        status,
                        json.dumps(summary),
                    ),
                )
                self.conn.commit()
        except Exception:
            return

    def write_violations(self, run_id: str, violations: list[Violation]) -> None:
        if not violations:
            return
        created = utc_now()
        try:
            with self._lock:
                self.conn.executemany(
                    "INSERT INTO audit_violations(id, run_id, created_at, plugin_name, "
                    "rule_id, severity, file, line, message, remediation, action) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            new_id(),
                            run_id,
                            created,
                            item.plugin_name,
                            item.rule_id,
                            item.severity,
                            item.file,
                            item.line,
                            item.message,
                            item.remediation,
                            item.action or ACTION_WARN,
                        )
                        for item in violations
                    ],
                )
                self.conn.commit()
        except Exception:
            return

    def list_violations(
        self,
        *,
        run_id: str | None = None,
        plugin: str | None = None,
        severity: str | None = None,
        file: str | None = None,
        limit: int = DEFAULT_AUDIT_LIMIT,
        before_id: str | None = None,
    ) -> dict[str, Any]:
        clamped = max(1, min(int(limit or DEFAULT_AUDIT_LIMIT), MAX_AUDIT_LIMIT))
        clauses = ["1=1"]
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if plugin:
            clauses.append("plugin_name = ?")
            params.append(plugin)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if file:
            clauses.append("file LIKE ?")
            params.append(f"%{file}%")
        if before_id:
            cursor = self.conn.execute(
                "SELECT created_at, id FROM audit_violations WHERE id = ?",
                (before_id,),
            ).fetchone()
            if cursor is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND id < ?))")
                params.extend([cursor["created_at"], cursor["created_at"], before_id])
        where = " AND ".join(clauses)
        rows = self.conn.execute(
            f"SELECT * FROM audit_violations WHERE {where} "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            [*params, clamped],
        ).fetchall()
        items = []
        for row in rows:
            item = _row_to_dict(row) or {}
            item["severity"] = item.get("severity") or SEVERITY_WARNING
            items.append(item)
        next_before_id = items[-1]["id"] if len(items) == clamped else None
        return {
            "items": items,
            "limit": clamped,
            "next_before_id": next_before_id,
        }
