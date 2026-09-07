"""Event-driven loop triggers with a lightweight scheduler.

Trigger definitions and fire history live beside the loop-runner store under
``~/.openhands/agent-canvas/``. ``fire_trigger`` starts a ``LoopRun`` via
``loop_runner``; scheduled ticks, commit/PR hooks, and manual REST fire share
that path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from kanban import new_id, utc_now
from loop_runner import LoopError, LoopStore, NotFoundError, STATUS_RUNNING, default_db_path as loop_db_path

TRIGGER_DB_FILENAME = "loop_triggers.sqlite"
TRIGGER_SCHEDULED = "scheduled"
TRIGGER_ON_COMMIT = "on_commit"
TRIGGER_ON_PR = "on_pr"
TRIGGER_MANUAL = "manual"
TRIGGER_TYPES = (
    TRIGGER_SCHEDULED,
    TRIGGER_ON_COMMIT,
    TRIGGER_ON_PR,
    TRIGGER_MANUAL,
)
SCHEDULE_CRON = "cron"
SCHEDULE_INTERVAL = "interval"
SCHEDULE_TYPES = (SCHEDULE_CRON, SCHEDULE_INTERVAL)
EVENT_FIRED = "fired"
EVENT_SKIPPED = "skipped"
EVENT_ERROR = "error"
EVENT_STATUSES = (EVENT_FIRED, EVENT_SKIPPED, EVENT_ERROR)
DEFAULT_EVENT_LIMIT = 50
DEFAULT_POLL_SECONDS = 1.0
ACTIVE_REASON = "run already active"

_active: "LoopTriggerService | None" = None


class TriggerError(Exception):
    """Raised for invalid trigger operations."""

    def __init__(
        self,
        message: str,
        status: int = 400,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, TRIGGER_DB_FILENAME)


def set_active_service(service: "LoopTriggerService | None") -> None:
    global _active
    _active = service


def get_active_service() -> "LoopTriggerService | None":
    return _active


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes")


def _validate_cron(expr: str) -> str:
    expr = (expr or "").strip()
    fields = expr.split()
    if len(fields) not in (5, 6):
        raise TriggerError("cron_expr must have 5 or 6 fields")
    try:
        from croniter import croniter  # type: ignore[import-untyped]

        if not croniter.is_valid(expr):
            raise TriggerError(f"invalid cron_expr: {expr}")
    except ImportError:
        pass
    return expr


def _cron_is_due(expr: str, base: datetime, now: datetime) -> bool:
    try:
        from croniter import croniter  # type: ignore[import-untyped]
    except ImportError:
        if expr.split() == ["*", "*", "*", "*", "*"]:
            return (now - base).total_seconds() >= 60
        raise TriggerError("croniter is required for this cron expression")
    nxt = croniter(expr, base).get_next(datetime)
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=base.tzinfo or timezone.utc)
    return nxt <= now


class LoopTriggerService:
    """CRUD + fire + scheduler for loop triggers."""

    def __init__(
        self,
        db_path: str = ":memory:",
        loop_store: LoopStore | None = None,
        project_store: Any | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        router_store: Any | None = None,
        kanban_store: Any | None = None,
        graph_store: Any | None = None,
    ) -> None:
        self.db_path = db_path
        self.loop_store = loop_store or LoopStore()
        self.project_store = project_store
        self.poll_seconds = poll_seconds
        self.router_store = router_store
        self.kanban_store = kanban_store
        self.graph_store = graph_store
        self._owns_loop_store = loop_store is None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.stop_scheduler()
        self.conn.close()
        if self._owns_loop_store:
            self.loop_store.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS loop_triggers (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                loop_definition_id TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                schedule_type TEXT,
                cron_expr TEXT,
                interval_seconds INTEGER,
                payload TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                last_fired_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trigger_events (
                id TEXT PRIMARY KEY,
                trigger_id TEXT NOT NULL,
                loop_run_id TEXT,
                fired_at TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                FOREIGN KEY (trigger_id) REFERENCES loop_triggers(id)
                    ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def create_trigger(
        self,
        project_id: str,
        loop_definition_id: str,
        trigger_type: str,
        schedule_type: str | None = None,
        cron_expr: str | None = None,
        interval_seconds: int | None = None,
        payload: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        loop_definition_id = (loop_definition_id or "").strip()
        trigger_type = (trigger_type or "").strip()
        if not project_id:
            raise TriggerError("project_id is required")
        if not loop_definition_id:
            raise TriggerError("loop_definition_id is required")
        if trigger_type not in TRIGGER_TYPES:
            raise TriggerError(f"trigger_type must be one of {', '.join(TRIGGER_TYPES)}")
        try:
            self.loop_store.get_definition(loop_definition_id)
        except NotFoundError as exc:
            raise TriggerError(str(exc), status=404) from exc
        sched, cron, interval = self._validate_schedule(
            trigger_type, schedule_type, cron_expr, interval_seconds
        )
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TriggerError("payload must be an object")
        trigger_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO loop_triggers (
                    id, project_id, loop_definition_id, trigger_type,
                    schedule_type, cron_expr, interval_seconds, payload,
                    enabled, last_fired_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger_id,
                    project_id,
                    loop_definition_id,
                    trigger_type,
                    sched,
                    cron,
                    interval,
                    json.dumps(payload),
                    1 if enabled else 0,
                    None,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self.get_trigger(trigger_id)

    def list_triggers(
        self,
        project_id: str | None = None,
        trigger_type: str | None = None,
        enabled: bool | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            values.append(project_id)
        if trigger_type:
            clauses.append("trigger_type = ?")
            values.append(trigger_type)
        if enabled is not None:
            clauses.append("enabled = ?")
            values.append(1 if enabled else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id FROM loop_triggers {where} ORDER BY created_at ASC",
                values,
            ).fetchall()
        return [self.get_trigger(row["id"]) for row in rows]

    def get_trigger(self, trigger_id: str) -> dict[str, Any]:
        with self._lock:
            row = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM loop_triggers WHERE id = ?", (trigger_id,)
                ).fetchone()
            )
        if row is None:
            raise TriggerError(f"Trigger {trigger_id} not found", status=404)
        row["payload"] = json.loads(row["payload"] or "{}")
        row["enabled"] = bool(row["enabled"])
        return row

    def update_trigger(self, trigger_id: str, **fields: Any) -> dict[str, Any]:
        trigger = self.get_trigger(trigger_id)
        allowed = {
            "enabled",
            "schedule_type",
            "cron_expr",
            "interval_seconds",
            "payload",
            "loop_definition_id",
        }
        updates = {key: fields[key] for key in allowed if key in fields}
        if "loop_definition_id" in updates:
            definition_id = str(updates["loop_definition_id"] or "").strip()
            try:
                self.loop_store.get_definition(definition_id)
            except NotFoundError as exc:
                raise TriggerError(str(exc), status=404) from exc
            updates["loop_definition_id"] = definition_id
        if "payload" in updates:
            if updates["payload"] is None:
                updates["payload"] = {}
            if not isinstance(updates["payload"], dict):
                raise TriggerError("payload must be an object")
            updates["payload"] = json.dumps(updates["payload"])
        if "enabled" in updates:
            updates["enabled"] = 1 if _as_bool(updates["enabled"]) else 0
        merged_type = trigger["trigger_type"]
        merged_sched = updates.get("schedule_type", trigger["schedule_type"])
        merged_cron = updates.get("cron_expr", trigger["cron_expr"])
        merged_interval = updates.get(
            "interval_seconds", trigger["interval_seconds"]
        )
        if any(
            key in fields
            for key in ("schedule_type", "cron_expr", "interval_seconds")
        ):
            sched, cron, interval = self._validate_schedule(
                merged_type, merged_sched, merged_cron, merged_interval
            )
            updates["schedule_type"] = sched
            updates["cron_expr"] = cron
            updates["interval_seconds"] = interval
        if not updates:
            return trigger
        updates["updated_at"] = utc_now()
        with self._lock:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            self.conn.execute(
                f"UPDATE loop_triggers SET {assignments} WHERE id = ?",
                (*updates.values(), trigger_id),
            )
            self.conn.commit()
        return self.get_trigger(trigger_id)

    def delete_trigger(self, trigger_id: str) -> None:
        self.get_trigger(trigger_id)
        with self._lock:
            self.conn.execute(
                "DELETE FROM loop_triggers WHERE id = ?", (trigger_id,)
            )
            self.conn.commit()

    def list_events(
        self,
        trigger_id: str | None = None,
        limit: int = DEFAULT_EVENT_LIMIT,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if trigger_id:
            self.get_trigger(trigger_id)
        limit = max(1, int(limit or DEFAULT_EVENT_LIMIT))
        offset = max(0, int(offset or 0))
        clauses: list[str] = []
        values: list[Any] = []
        if trigger_id:
            clauses.append("trigger_id = ?")
            values.append(trigger_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT * FROM trigger_events
                {where}
                ORDER BY fired_at DESC, rowid DESC
                LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def fire_trigger(
        self,
        trigger_id: str,
        context: dict[str, Any] | None = None,
        *,
        ignore_enabled: bool = False,
    ) -> dict[str, Any]:
        trigger = self.get_trigger(trigger_id)
        context = dict(context or {})
        if not trigger["enabled"] and not ignore_enabled:
            raise TriggerError("trigger is disabled", status=409)
        if self._has_active_run(trigger["loop_definition_id"], trigger["project_id"]):
            event = self._record_event(
                trigger_id, None, EVENT_SKIPPED, ACTIVE_REASON
            )
            raise TriggerError(
                ACTIVE_REASON,
                status=409,
                payload={"reason": ACTIVE_REASON, "event": event},
            )
        try:
            worktree_dir = self._resolve_worktree(trigger, context)
            run = self.loop_store.start_run(
                trigger["loop_definition_id"],
                worktree_dir=worktree_dir,
                session_id=context.get("session_id"),
            )
        except (LoopError, TriggerError, OSError, ValueError) as exc:
            event = self._record_event(
                trigger_id, None, EVENT_ERROR, str(exc)
            )
            status = getattr(exc, "status", 400)
            raise TriggerError(
                str(exc),
                status=status if isinstance(status, int) else 400,
                payload={"reason": str(exc), "event": event},
            ) from exc
        self._touch_last_fired(trigger_id)
        event = self._record_event(
            trigger_id, run["id"], EVENT_FIRED, context.get("reason")
        )
        result: dict[str, Any] = {"event": event, "run": run}
        payload = trigger.get("payload") or {}
        task_text = str(
            context.get("task_text")
            or payload.get("task_text")
            or f"{trigger['trigger_type']} {trigger['loop_definition_id']}"
        )
        try:
            from graph_agent_hooks import apply_dispatch_graph_context

            graph = apply_dispatch_graph_context(
                {
                    "task_text": task_text,
                    "worktree_dir": worktree_dir,
                    "root": worktree_dir or context.get("root"),
                    "card_id": context.get("card_id") or payload.get("card_id"),
                    "graph_enabled": context.get("graph_enabled"),
                    "seeds": context.get("seeds"),
                },
                store=self.graph_store,
                kanban_store=self.kanban_store,
            )
            result["graph_context"] = graph.get("graph_context")
            if graph.get("spec_text"):
                result["prompt"] = graph["spec_text"]
        except Exception:
            pass
        try:
            from standards_agent_hooks import apply_dispatch_standards_context

            standards = apply_dispatch_standards_context(
                {
                    "spec_text": result.get("prompt") or "",
                    "task_text": task_text,
                    "worktree_dir": worktree_dir,
                    "root": worktree_dir or context.get("root"),
                }
            )
            if standards.get("spec_text"):
                result["prompt"] = standards["spec_text"]
            result["standards_block"] = standards.get("standards_block")
        except Exception:
            pass
        if self.router_store is not None:
            from router_runtime import persist_dispatch_trace, resolve_for_dispatch

            routing = resolve_for_dispatch(
                self.router_store,
                task_text=task_text,
                run_id=run["id"],
                card_id=context.get("card_id") or payload.get("card_id"),
                connected_providers=context.get("connected_providers"),
                local_runtimes=context.get("local_runtimes") or {},
            )
            persist_dispatch_trace(
                routing,
                worktree_dir=worktree_dir,
                kanban_store=self.kanban_store,
                card_id=context.get("card_id") or payload.get("card_id"),
            )
            result["routing"] = routing
        return result

    def tick(
        self,
        now: datetime | None = None,
        worktree_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        clock = now or datetime.now(timezone.utc)
        fired: list[dict[str, Any]] = []
        for trigger in self.list_triggers(
            trigger_type=TRIGGER_SCHEDULED, enabled=True
        ):
            if not self._is_due(trigger, clock):
                continue
            context: dict[str, Any] = {"reason": "scheduled"}
            if worktree_dir:
                context["worktree_dir"] = worktree_dir
            try:
                result = self.fire_trigger(trigger["id"], context)
                fired.append(result["event"])
            except TriggerError:
                continue
        return fired

    def notify_commit(
        self,
        project_id: str,
        branch: str,
        commit_hash: str,
        worktree_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._notify_hooks(
            TRIGGER_ON_COMMIT,
            project_id,
            branch,
            {
                "reason": "on_commit",
                "commit_hash": commit_hash,
                "worktree_dir": worktree_dir,
            },
        )

    def notify_pr(
        self,
        project_id: str,
        branch: str,
        pr_url: str,
        worktree_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._notify_hooks(
            TRIGGER_ON_PR,
            project_id,
            branch,
            {
                "reason": "on_pr",
                "pr_url": pr_url,
                "worktree_dir": worktree_dir,
            },
        )

    def start_scheduler(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="loop-trigger-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop_scheduler(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.tick()
            except Exception:
                continue

    def _notify_hooks(
        self,
        trigger_type: str,
        project_id: str,
        branch: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        payload_context = {key: value for key, value in context.items() if value}
        payload_context["branch"] = branch
        for trigger in self.list_triggers(
            project_id=project_id, trigger_type=trigger_type, enabled=True
        ):
            watched = (trigger["payload"] or {}).get("branch")
            if watched and watched != branch:
                continue
            try:
                result = self.fire_trigger(trigger["id"], payload_context)
                events.append(result["event"])
            except TriggerError:
                continue
        return events

    def _validate_schedule(
        self,
        trigger_type: str,
        schedule_type: str | None,
        cron_expr: str | None,
        interval_seconds: int | None,
    ) -> tuple[str | None, str | None, int | None]:
        if trigger_type != TRIGGER_SCHEDULED:
            return None, None, None
        kind = (schedule_type or "").strip()
        if kind not in SCHEDULE_TYPES:
            raise TriggerError("scheduled triggers require schedule_type cron or interval")
        if kind == SCHEDULE_CRON:
            expr = _validate_cron(str(cron_expr or ""))
            return kind, expr, None
        try:
            seconds = int(interval_seconds) if interval_seconds is not None else 0
        except (TypeError, ValueError) as exc:
            raise TriggerError("interval_seconds must be an integer") from exc
        if seconds < 1:
            raise TriggerError("interval_seconds must be >= 1")
        return kind, None, seconds

    def _is_due(self, trigger: dict[str, Any], now: datetime) -> bool:
        created = _parse_dt(trigger["created_at"])
        last = (
            _parse_dt(trigger["last_fired_at"]) if trigger["last_fired_at"] else None
        )
        base = last or created
        if trigger["schedule_type"] == SCHEDULE_INTERVAL:
            seconds = int(trigger["interval_seconds"] or 0)
            return (now - base).total_seconds() >= seconds
        if trigger["schedule_type"] == SCHEDULE_CRON:
            return _cron_is_due(str(trigger["cron_expr"] or ""), base, now)
        return False

    def _has_active_run(self, definition_id: str, project_id: str) -> bool:
        for run in self.loop_store.list_runs(definition_id):
            if run["status"] == STATUS_RUNNING and run["project_id"] == project_id:
                return True
        return False

    def _resolve_worktree(
        self, trigger: dict[str, Any], context: dict[str, Any]
    ) -> str:
        for candidate in (
            context.get("worktree_dir"),
            (trigger["payload"] or {}).get("worktree_dir"),
        ):
            if candidate:
                return str(candidate)
        if self.project_store is not None:
            project = self.project_store.get_project(trigger["project_id"])
            path = project.get("local_path")
            if path:
                return str(path)
        raise TriggerError("worktree_dir is required")

    def _touch_last_fired(self, trigger_id: str) -> None:
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                UPDATE loop_triggers
                SET last_fired_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, trigger_id),
            )
            self.conn.commit()

    def _record_event(
        self,
        trigger_id: str,
        loop_run_id: str | None,
        status: str,
        reason: str | None,
    ) -> dict[str, Any]:
        event_id = new_id()
        fired_at = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO trigger_events (
                    id, trigger_id, loop_run_id, fired_at, status, reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, trigger_id, loop_run_id, fired_at, status, reason),
            )
            self.conn.commit()
        return {
            "id": event_id,
            "trigger_id": trigger_id,
            "loop_run_id": loop_run_id,
            "fired_at": fired_at,
            "status": status,
            "reason": reason,
        }


def notify_commit(
    project_id: str,
    branch: str,
    commit_hash: str,
    worktree_dir: str | None = None,
) -> list[dict[str, Any]]:
    service = get_active_service()
    if service is None:
        return []
    return service.notify_commit(project_id, branch, commit_hash, worktree_dir)


def notify_pr(
    project_id: str,
    branch: str,
    pr_url: str,
    worktree_dir: str | None = None,
) -> list[dict[str, Any]]:
    service = get_active_service()
    if service is None:
        return []
    return service.notify_pr(project_id, branch, pr_url, worktree_dir)


def start_default_scheduler() -> LoopTriggerService:
    global _active
    if _active is None:
        _active = LoopTriggerService(
            db_path=default_db_path(),
            loop_store=LoopStore(loop_db_path()),
        )
    _active.start_scheduler()
    return _active


def _in_unit_tests() -> bool:
    return "unittest" in sys.modules or "pytest" in sys.modules


if (
    not _in_unit_tests()
    and os.environ.get("OH_LOOP_TRIGGERS_NO_SCHEDULER") != "1"
):
    try:
        start_default_scheduler()
    except Exception:
        pass
