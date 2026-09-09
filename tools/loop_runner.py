"""Generic staged loop engine with SQLite persistence.

Loop definitions, runs, and per-stage attempts live beside the other
agent-canvas stores under ``~/.openhands/agent-canvas/``. Stage commands
run as subprocesses in a worktree directory. ``cmd: null`` resolves to
``npm run <stage.name>`` from the worktree ``package.json``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import time
from typing import Any

from cost_estimator import DEFAULT_MODEL, estimate_cost_usd
from kanban import new_id, utc_now

LOOP_DB_FILENAME = "loops.sqlite"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ABORTED = "aborted"
STATUS_AWAITING_INPUT = "awaiting_input"
STATUS_SKIPPED = "skipped"
RUN_STATUSES = (
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_PASSED,
    STATUS_FAILED,
    STATUS_ABORTED,
    STATUS_AWAITING_INPUT,
)
STAGE_STATUSES = (
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_PASSED,
    STATUS_FAILED,
    STATUS_SKIPPED,
)
ON_FAILURE_STOP = "stop"
ON_FAILURE_AUTO_FIX = "auto_fix"
ON_FAILURE_NOTIFY = "notify"
ON_FAILURE_MODES = (ON_FAILURE_STOP, ON_FAILURE_AUTO_FIX, ON_FAILURE_NOTIFY)
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_COST_USD = 5.0
DEFAULT_ON_FAILURE = ON_FAILURE_AUTO_FIX
TOKENS_PER_SECOND = 50
MIN_STAGE_COST_USD = 0.000001
STAGE_HANDLERS: dict[str, Any] = {}


def register_stage_handler(definition_name: str, stage_name: str, handler: Any) -> None:
    STAGE_HANDLERS[f"{definition_name}:{stage_name}"] = handler


class LoopError(Exception):
    """Raised for invalid loop operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(LoopError):
    """Raised when a loop definition or run does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, LOOP_DB_FILENAME)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def estimate_attempt_cost(duration_seconds: float, model: str | None = None) -> float:
    tokens = max(1, int(max(duration_seconds, 0) * TOKENS_PER_SECOND))
    return max(MIN_STAGE_COST_USD, estimate_cost_usd(tokens, model or DEFAULT_MODEL))


def resolve_stage_cmd(stage: dict[str, Any], worktree_dir: str) -> str:
    cmd = stage.get("cmd")
    if cmd:
        return str(cmd)
    name = str(stage.get("name") or "").strip()
    package_path = os.path.join(worktree_dir, "package.json")
    if not os.path.isfile(package_path):
        raise LoopError(f"No command for stage {name!r} and no package.json")
    with open(package_path, encoding="utf-8") as handle:
        data = json.load(handle)
    scripts = data.get("scripts") if isinstance(data, dict) else {}
    if not isinstance(scripts, dict) or name not in scripts:
        raise LoopError(f"package.json has no script named {name!r}")
    return f"npm run {name}"


def validate_stages(stages: Any) -> list[dict[str, Any]]:
    if not isinstance(stages, list) or not stages:
        raise LoopError("stages must be a non-empty list")
    validated: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise LoopError("each stage must be an object")
        name = str(stage.get("name") or "").strip()
        if not name:
            raise LoopError("each stage needs a name")
        cmd = stage.get("cmd")
        if cmd is not None and not isinstance(cmd, str):
            raise LoopError(f"stage {name!r} cmd must be a string or null")
        validated.append(
            {
                "name": name,
                "cmd": cmd,
                "iterative": bool(stage.get("iterative", True)),
            }
        )
    return validated


class LoopStore:
    """CRUD store for loop definitions, runs, and stage attempts."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS loop_definitions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                project_id TEXT NOT NULL,
                stages TEXT NOT NULL,
                max_iterations INTEGER NOT NULL,
                max_cost_usd REAL NOT NULL,
                on_failure TEXT NOT NULL,
                config TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS loop_runs (
                id TEXT PRIMARY KEY,
                definition_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                session_id TEXT,
                worktree_dir TEXT,
                status TEXT NOT NULL,
                current_stage TEXT,
                iteration INTEGER NOT NULL DEFAULT 0,
                total_cost_usd REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (definition_id) REFERENCES loop_definitions(id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS stage_runs (
                id TEXT PRIMARY KEY,
                loop_run_id TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                last_output TEXT,
                started_at TEXT,
                finished_at TEXT,
                FOREIGN KEY (loop_run_id) REFERENCES loop_runs(id) ON DELETE CASCADE
            );
            """
        )
        columns = [
            row[1]
            for row in self.conn.execute("PRAGMA table_info(loop_definitions)")
        ]
        if "config" not in columns:
            self.conn.execute(
                "ALTER TABLE loop_definitions ADD COLUMN config TEXT NOT NULL DEFAULT '{}'"
            )
        self.conn.commit()

    def create_definition(
        self,
        name: str,
        project_id: str,
        stages: list[dict[str, Any]],
        max_iterations: int | None = None,
        max_cost_usd: float | None = None,
        on_failure: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = (name or "").strip()
        project_id = (project_id or "").strip()
        if not name:
            raise LoopError("name is required")
        if not project_id:
            raise LoopError("project_id is required")
        validated = validate_stages(stages)
        mode = on_failure or DEFAULT_ON_FAILURE
        if mode not in ON_FAILURE_MODES:
            raise LoopError(f"on_failure must be one of {', '.join(ON_FAILURE_MODES)}")
        iterations = int(max_iterations or DEFAULT_MAX_ITERATIONS)
        if iterations < 1:
            raise LoopError("max_iterations must be >= 1")
        cost_cap = float(max_cost_usd if max_cost_usd is not None else DEFAULT_MAX_COST_USD)
        if cost_cap < 0:
            raise LoopError("max_cost_usd must be >= 0")
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise LoopError("config must be an object")
        definition_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO loop_definitions (
                    id, name, project_id, stages, max_iterations, max_cost_usd,
                    on_failure, config, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    definition_id,
                    name,
                    project_id,
                    json.dumps(validated),
                    iterations,
                    cost_cap,
                    mode,
                    json.dumps(config),
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self.get_definition(definition_id)

    def list_definitions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM loop_definitions ORDER BY created_at ASC"
            ).fetchall()
        return [self.get_definition(row["id"]) for row in rows]

    def get_definition(self, definition_id: str) -> dict[str, Any]:
        with self._lock:
            row = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM loop_definitions WHERE id = ?",
                    (definition_id,),
                ).fetchone()
            )
        if row is None:
            raise NotFoundError(f"Loop definition {definition_id} not found")
        row["stages"] = json.loads(row["stages"])
        row["config"] = json.loads(row.get("config") or "{}")
        return row

    def set_definition_config(
        self, definition_id: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        self.get_definition(definition_id)
        if not isinstance(config, dict):
            raise LoopError("config must be an object")
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                UPDATE loop_definitions
                SET config = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(config), now, definition_id),
            )
            self.conn.commit()
        return self.get_definition(definition_id)

    def list_runs(self, definition_id: str) -> list[dict[str, Any]]:
        self.get_definition(definition_id)
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id FROM loop_runs
                WHERE definition_id = ?
                ORDER BY created_at ASC
                """,
                (definition_id,),
            ).fetchall()
        return [self.get_run(row["id"]) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM loop_runs WHERE id = ?", (run_id,)
                ).fetchone()
            )
            if run is None:
                raise NotFoundError(f"Loop run {run_id} not found")
            stages = self.conn.execute(
                """
                SELECT * FROM stage_runs
                WHERE loop_run_id = ?
                ORDER BY started_at ASC, rowid ASC
                """,
                (run_id,),
            ).fetchall()
        run["stages"] = [_row_to_dict(row) for row in stages]
        return run

    def start_run(
        self,
        definition_id: str,
        worktree_dir: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        definition = self.get_definition(definition_id)
        worktree_dir = os.path.abspath(worktree_dir)
        if not os.path.isdir(worktree_dir):
            raise LoopError(f"worktree_dir does not exist: {worktree_dir}")
        run_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO loop_runs (
                    id, definition_id, project_id, session_id, worktree_dir,
                    status, current_stage, iteration, total_cost_usd,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    definition_id,
                    definition["project_id"],
                    session_id,
                    worktree_dir,
                    STATUS_RUNNING,
                    None,
                    0,
                    0.0,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self._continue_run(run_id, from_index=0)

    def request_fix(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] not in (STATUS_RUNNING, STATUS_AWAITING_INPUT):
            raise LoopError("run is not waiting for a fix")
        failed = self._current_failed_stage(run)
        if failed is None:
            raise LoopError("no failed stage to fix")
        return {
            "run_id": run_id,
            "stage_name": failed["stage_name"],
            "last_output": failed["last_output"],
            "attempt": failed["attempt"],
        }

    def retry_stage(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] not in (STATUS_RUNNING, STATUS_AWAITING_INPUT, STATUS_FAILED):
            raise LoopError("run is not retryable")
        failed = self._current_failed_stage(run)
        if failed is None:
            raise LoopError("no failed stage to retry")
        definition = self.get_definition(run["definition_id"])
        names = [stage["name"] for stage in definition["stages"]]
        index = names.index(failed["stage_name"])
        self._set_run(
            run_id,
            status=STATUS_RUNNING,
            current_stage=failed["stage_name"],
        )
        return self._continue_run(run_id, from_index=index)

    def pause_for_input(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        self._set_run(
            run_id,
            status=STATUS_AWAITING_INPUT,
            current_stage=run["current_stage"],
        )
        return self.get_run(run_id)

    def submit_feedback(
        self,
        run_id: str,
        approve: bool,
        note: str | None = None,
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] not in (STATUS_RUNNING, STATUS_AWAITING_INPUT, STATUS_FAILED):
            raise LoopError("run is not waiting for feedback")
        if note:
            failed = self._current_failed_stage(run)
            if failed is not None:
                output = ((failed["last_output"] or "") + f"\nfeedback: {note}").strip()
                self._upsert_stage(
                    run_id,
                    failed["stage_name"],
                    status=failed["status"],
                    attempt=failed["attempt"],
                    last_output=output,
                )
        if approve:
            run = self.retry_stage(run_id)
            definition = self.get_definition(run["definition_id"])
            if run["status"] == STATUS_RUNNING and definition["name"] == "manual-loop":
                return self.pause_for_input(run["id"])
            return run
        return self.abort_run(run_id)

    def abort_run(self, run_id: str) -> dict[str, Any]:
        self.get_run(run_id)
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                UPDATE loop_runs
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (STATUS_ABORTED, now, run_id),
            )
            self.conn.commit()
        return self.get_run(run_id)

    def resume_checkpoint(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        definition = self.get_definition(run["definition_id"])
        return {"run": run, "definition": definition, "stages": run["stages"]}

    def _continue_run(self, run_id: str, from_index: int) -> dict[str, Any]:
        run = self.get_run(run_id)
        definition = self.get_definition(run["definition_id"])
        worktree_dir = run["worktree_dir"]
        stages = definition["stages"]
        for index in range(from_index, len(stages)):
            stage = stages[index]
            outcome = self._run_stage(run_id, definition, stage, worktree_dir)
            if outcome == STATUS_PASSED:
                continue
            if outcome == STATUS_ABORTED:
                return self.get_run(run_id)
            if (
                definition["on_failure"] == ON_FAILURE_AUTO_FIX
                and stage["iterative"]
                and outcome == STATUS_RUNNING
            ):
                return self.get_run(run_id)
            self._set_run(
                run_id,
                status=STATUS_FAILED,
                current_stage=stage["name"],
            )
            return self.get_run(run_id)
        self._set_run(run_id, status=STATUS_PASSED, current_stage=None)
        return self.get_run(run_id)

    def _run_stage(
        self,
        run_id: str,
        definition: dict[str, Any],
        stage: dict[str, Any],
        worktree_dir: str,
    ) -> str:
        run = self.get_run(run_id)
        if run["status"] == STATUS_ABORTED:
            return STATUS_ABORTED
        existing = self._stage_row(run_id, stage["name"])
        attempt = int(existing["attempt"] if existing else 0) + 1
        if attempt > int(definition["max_iterations"]):
            self._upsert_stage(
                run_id,
                stage["name"],
                status=STATUS_FAILED,
                attempt=max(attempt - 1, 1),
                last_output=existing["last_output"] if existing else "max_iterations exceeded",
            )
            self._set_run(
                run_id,
                status=STATUS_FAILED,
                current_stage=stage["name"],
                iteration=int(run["iteration"]),
            )
            return STATUS_FAILED
        if float(run["total_cost_usd"]) > float(definition["max_cost_usd"]):
            self._set_run(run_id, status=STATUS_FAILED, current_stage=stage["name"])
            return STATUS_FAILED
        started = utc_now()
        self._upsert_stage(
            run_id,
            stage["name"],
            status=STATUS_RUNNING,
            attempt=attempt,
            started_at=started,
        )
        self._set_run(
            run_id,
            status=STATUS_RUNNING,
            current_stage=stage["name"],
            iteration=int(run["iteration"]) + 1,
        )
        handler = STAGE_HANDLERS.get(f"{definition['name']}:{stage['name']}")
        t0 = time.monotonic()
        if handler is not None:
            try:
                passed, output = handler(
                    run_id=run_id,
                    stage=stage,
                    worktree_dir=worktree_dir,
                    definition=definition,
                )
            except Exception as exc:
                passed, output = False, str(exc)
            skipped = passed == STATUS_SKIPPED or passed == "skipped"
            passed_bool = skipped or passed is True
            output = (output or "").strip()
        else:
            cmd = resolve_stage_cmd(stage, worktree_dir)
            result = subprocess.run(
                cmd,
                cwd=worktree_dir,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            passed_bool = result.returncode == 0
            skipped = False
        duration = time.monotonic() - t0
        cost = estimate_attempt_cost(duration)
        self._add_cost(run_id, cost)
        updated = self.get_run(run_id)
        over_budget = float(updated["total_cost_usd"]) > float(definition["max_cost_usd"])
        if skipped and not over_budget:
            self._upsert_stage(
                run_id,
                stage["name"],
                status=STATUS_SKIPPED,
                attempt=attempt,
                last_output=output,
                started_at=started,
                finished_at=utc_now(),
            )
            return STATUS_PASSED
        if passed_bool and not over_budget:
            self._upsert_stage(
                run_id,
                stage["name"],
                status=STATUS_PASSED,
                attempt=attempt,
                last_output=output,
                started_at=started,
                finished_at=utc_now(),
            )
            return STATUS_PASSED
        self._upsert_stage(
            run_id,
            stage["name"],
            status=STATUS_FAILED,
            attempt=attempt,
            last_output=output or ("cost cap exceeded" if over_budget else ""),
            started_at=started,
            finished_at=utc_now(),
        )
        if over_budget or not stage["iterative"] or definition["on_failure"] != ON_FAILURE_AUTO_FIX:
            return STATUS_FAILED
        if attempt >= int(definition["max_iterations"]):
            return STATUS_FAILED
        return STATUS_RUNNING

    def _current_failed_stage(self, run: dict[str, Any]) -> dict[str, Any] | None:
        for stage in reversed(run["stages"]):
            if stage["status"] == STATUS_FAILED:
                return stage
        return None

    def _stage_row(self, run_id: str, stage_name: str) -> dict[str, Any] | None:
        with self._lock:
            return _row_to_dict(
                self.conn.execute(
                    """
                    SELECT * FROM stage_runs
                    WHERE loop_run_id = ? AND stage_name = ?
                    ORDER BY attempt DESC
                    LIMIT 1
                    """,
                    (run_id, stage_name),
                ).fetchone()
            )

    def _upsert_stage(
        self,
        run_id: str,
        stage_name: str,
        *,
        status: str,
        attempt: int,
        last_output: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        existing = self._stage_row(run_id, stage_name)
        with self._lock:
            if existing is None:
                self.conn.execute(
                    """
                    INSERT INTO stage_runs (
                        id, loop_run_id, stage_name, status, attempt,
                        last_output, started_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id(),
                        run_id,
                        stage_name,
                        status,
                        attempt,
                        last_output,
                        started_at,
                        finished_at,
                    ),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE stage_runs
                    SET status = ?, attempt = ?, last_output = ?,
                        started_at = COALESCE(?, started_at),
                        finished_at = ?
                    WHERE id = ?
                    """,
                    (
                        status,
                        attempt,
                        last_output if last_output is not None else existing["last_output"],
                        started_at,
                        finished_at,
                        existing["id"],
                    ),
                )
            self.conn.commit()

    def _add_cost(self, run_id: str, cost: float) -> None:
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                UPDATE loop_runs
                SET total_cost_usd = total_cost_usd + ?, updated_at = ?
                WHERE id = ?
                """,
                (cost, now, run_id),
            )
            self.conn.commit()

    def _set_run(
        self,
        run_id: str,
        *,
        status: str,
        current_stage: str | None,
        iteration: int | None = None,
    ) -> None:
        now = utc_now()
        with self._lock:
            if iteration is None:
                self.conn.execute(
                    """
                    UPDATE loop_runs
                    SET status = ?, current_stage = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, current_stage, now, run_id),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE loop_runs
                    SET status = ?, current_stage = ?, iteration = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, current_stage, iteration, now, run_id),
                )
            self.conn.commit()
