"""Text → kanban tickets → worktree sessions → commit-loop orchestrator."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from cost_estimator import estimate_task
from fleet import FleetStore
from kanban import KanbanStore, new_id, utc_now
from pr_creator import slugify
from project_bootstrap import decompose_spec, seed_board, stub_llm_complete

FEATURE_DEV_DB_FILENAME = "feature_developer.sqlite"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_PARTIAL = "partial"
STATUS_ABORTED = "aborted"
STATUS_PAUSED = "paused"
RUN_STATUSES = (
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_PASSED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_ABORTED,
    STATUS_PAUSED,
)
TICKET_PENDING = "pending"
TICKET_IN_PROGRESS = "in_progress"
TICKET_PASSED = "passed"
TICKET_FAILED = "failed"
TICKET_SKIPPED = "skipped"
DEFAULT_MAX_CONCURRENT = 1
ImplementFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class FeatureDevError(Exception):
    """Raised for invalid feature-developer operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(FeatureDevError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, FEATURE_DEV_DB_FILENAME)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class FeatureDeveloper:
    """Decompose a spec, estimate, implement tickets, and summarize."""

    def __init__(
        self,
        db_path: str = ":memory:",
        kanban_store: KanbanStore | None = None,
        fleet_store: FleetStore | None = None,
        project_store: Any | None = None,
        coordinator: Any | None = None,
        commit_loop: Any | None = None,
        llm_complete: Callable[[str], str] | None = None,
        implement_fn: ImplementFn | None = None,
        cost_cap: float | None = None,
        router_store: Any | None = None,
        dispatch_runner: Callable[..., dict[str, Any]] | None = None,
        connected_providers: list[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.kanban_store = kanban_store or KanbanStore()
        self.fleet_store = fleet_store
        self.project_store = project_store
        self.coordinator = coordinator
        self.commit_loop = commit_loop
        self.llm_complete = llm_complete or stub_llm_complete
        self.implement_fn = implement_fn
        self.cost_cap = cost_cap
        self.router_store = router_store
        self.dispatch_runner = dispatch_runner
        self.connected_providers = connected_providers
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()
        if self.kanban_store is not None:
            # ponytail: caller owns shared stores; only close ones we created
            pass

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS feature_dev_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                spec_text TEXT NOT NULL,
                board_id TEXT,
                status TEXT NOT NULL,
                current_ticket_index INTEGER NOT NULL DEFAULT 0,
                total_estimate_usd REAL NOT NULL DEFAULT 0,
                total_actual_usd REAL NOT NULL DEFAULT 0,
                max_concurrent_agents INTEGER NOT NULL,
                continue_on_failure INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature_dev_tickets (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                card_id TEXT,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                branch_name TEXT,
                session_id TEXT,
                estimate_usd REAL,
                actual_usd REAL,
                error TEXT,
                started_at TEXT,
                finished_at TEXT,
                position INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES feature_dev_runs(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def start_run(
        self,
        project_id: str,
        spec_text: str,
        max_concurrent_agents: int | None = None,
        continue_on_failure: bool = False,
        cost_cap: float | None = None,
    ) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        spec_text = (spec_text or "").strip()
        if not project_id:
            raise FeatureDevError("project_id is required")
        if not spec_text:
            raise FeatureDevError("spec_text is required")
        concurrency = int(max_concurrent_agents or DEFAULT_MAX_CONCURRENT)
        if concurrency < 1:
            raise FeatureDevError("max_concurrent_agents must be >= 1")
        run_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO feature_dev_runs (
                    id, project_id, spec_text, board_id, status,
                    current_ticket_index, total_estimate_usd, total_actual_usd,
                    max_concurrent_agents, continue_on_failure, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    project_id,
                    spec_text,
                    None,
                    STATUS_RUNNING,
                    0,
                    0.0,
                    0.0,
                    concurrency,
                    1 if continue_on_failure else 0,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        self._decompose(run_id, project_id, spec_text)
        self._estimate(run_id, cost_cap if cost_cap is not None else self._project_cap(project_id))
        run = self.get_run(run_id)
        if run["status"] == STATUS_FAILED:
            return run
        return self._implement_remaining(run_id)

    def list_runs(
        self,
        project_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            values.append(project_id)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id FROM feature_dev_runs {where} ORDER BY created_at DESC",
                values,
            ).fetchall()
        return [self.get_run(row["id"]) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM feature_dev_runs WHERE id = ?", (run_id,)
                ).fetchone()
            )
            if run is None:
                raise NotFoundError(f"Feature-dev run {run_id} not found")
            tickets = self.conn.execute(
                """
                SELECT * FROM feature_dev_tickets
                WHERE run_id = ?
                ORDER BY position ASC
                """,
                (run_id,),
            ).fetchall()
        run["tickets"] = [_row_to_dict(row) for row in tickets]
        run["continue_on_failure"] = bool(run["continue_on_failure"])
        return run

    def pause_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] not in {STATUS_RUNNING, STATUS_PAUSED}:
            raise FeatureDevError("only a running run can be paused")
        self._set_run(run_id, status=STATUS_PAUSED)
        return self.get_run(run_id)

    def resume_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] != STATUS_PAUSED:
            raise FeatureDevError("only a paused run can be resumed")
        self._set_run(run_id, status=STATUS_RUNNING)
        return self._implement_remaining(run_id)

    def abort_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        for ticket in run["tickets"]:
            if ticket["status"] in {TICKET_PENDING, TICKET_IN_PROGRESS}:
                self._patch_ticket(ticket["id"], status=TICKET_SKIPPED)
        self._set_run(run_id, status=STATUS_ABORTED)
        return self.get_run(run_id)

    def report(self, run_id: str) -> str:
        run = self.get_run(run_id)
        lines = [
            f"# Feature developer run `{run['id']}`",
            "",
            f"- Project: `{run['project_id']}`",
            f"- Status: **{run['status']}**",
            f"- Estimate: ${float(run['total_estimate_usd']):.4f}",
            f"- Actual: ${float(run['total_actual_usd']):.4f}",
            "",
            "| Title | Status | Cost | Branch |",
            "| --- | --- | --- | --- |",
        ]
        for ticket in run["tickets"]:
            cost = ticket["actual_usd"]
            if cost is None:
                cost = ticket["estimate_usd"] or 0
            branch = ticket["branch_name"] or ""
            lines.append(
                f"| {ticket['title']} | {ticket['status']} | ${float(cost):.4f} | {branch} |"
            )
        return "\n".join(lines) + "\n"

    def _decompose(self, run_id: str, project_id: str, spec_text: str) -> None:
        suggested = decompose_spec(spec_text, self.llm_complete)
        seeded = seed_board(
            self.kanban_store,
            suggested,
            name="Feature developer",
            project_id=project_id,
        )
        now = utc_now()
        with self._lock:
            self.conn.execute(
                "UPDATE feature_dev_runs SET board_id = ?, updated_at = ? WHERE id = ?",
                (seeded["board"]["id"], now, run_id),
            )
            for position, card in enumerate(seeded["cards"]):
                self.conn.execute(
                    """
                    INSERT INTO feature_dev_tickets (
                        id, run_id, card_id, title, status, branch_name,
                        session_id, estimate_usd, actual_usd, error,
                        started_at, finished_at, position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id(),
                        run_id,
                        card["id"],
                        card["title"],
                        TICKET_PENDING,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        position,
                    ),
                )
            self.conn.commit()

    def _estimate(self, run_id: str, cost_cap: float | None) -> None:
        run = self.get_run(run_id)
        total = 0.0
        for ticket in run["tickets"]:
            card = self.kanban_store.get_card(ticket["card_id"])
            estimate = estimate_task(
                title=str(card.get("title") or ticket["title"]),
                description=str(card.get("description") or ""),
                priority=str(card.get("priority") or "P2"),
            )
            self.kanban_store.update_card(
                ticket["card_id"],
                estimate_tokens=estimate["estimate_tokens"],
                estimate_cost=estimate["estimate_cost"],
            )
            self._patch_ticket(ticket["id"], estimate_usd=estimate["estimate_cost"])
            total += float(estimate["estimate_cost"])
        self._set_run(run_id, total_estimate_usd=total)
        if cost_cap is not None and total > float(cost_cap):
            for ticket in run["tickets"]:
                self._patch_ticket(ticket["id"], status=TICKET_SKIPPED, error="over cost cap")
            self._set_run(run_id, status=STATUS_FAILED)

    def _implement_remaining(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        pending = [
            ticket
            for ticket in run["tickets"]
            if ticket["status"] == TICKET_PENDING
        ]
        workers = max(1, int(run["max_concurrent_agents"]))
        if workers == 1 or len(pending) <= 1:
            for ticket in pending:
                run = self.get_run(run_id)
                if run["status"] in {STATUS_ABORTED, STATUS_PAUSED}:
                    break
                self._run_one(run_id, ticket)
                run = self.get_run(run_id)
                if run["status"] == STATUS_ABORTED:
                    break
                if run["status"] == STATUS_PAUSED:
                    break
                if (
                    not run["continue_on_failure"]
                    and self._ticket(run_id, ticket["id"])["status"] == TICKET_FAILED
                ):
                    self.abort_run(run_id)
                    self._set_run(run_id, status=STATUS_FAILED)
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(self._run_one, run_id, ticket) for ticket in pending
                ]
                for future in as_completed(futures):
                    future.result()
                    run = self.get_run(run_id)
                    if run["status"] in {STATUS_ABORTED, STATUS_PAUSED}:
                        break
        return self._finalize(run_id)

    def _run_one(self, run_id: str, ticket: dict[str, Any]) -> None:
        run = self.get_run(run_id)
        if run["status"] in {STATUS_ABORTED, STATUS_PAUSED}:
            return
        self._patch_ticket(
            ticket["id"],
            status=TICKET_IN_PROGRESS,
            started_at=utc_now(),
        )
        routing = self._resolve_ticket(run, ticket)
        try:
            if self.implement_fn is not None:
                result = self.implement_fn(self.get_run(run_id), ticket)
            else:
                result = self._implement_ticket(run, ticket, routing=routing)
        except Exception as exc:
            self._patch_ticket(
                ticket["id"],
                status=TICKET_FAILED,
                error=str(exc),
                finished_at=utc_now(),
            )
            return
        status = result.get("status") or TICKET_PASSED
        self._patch_ticket(
            ticket["id"],
            status=status,
            actual_usd=result.get("actual_usd"),
            branch_name=result.get("branch_name"),
            session_id=result.get("session_id"),
            error=result.get("error"),
            finished_at=utc_now(),
        )
        if result.get("actual_usd") is not None:
            with self._lock:
                self.conn.execute(
                    """
                    UPDATE feature_dev_runs
                    SET total_actual_usd = total_actual_usd + ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (float(result["actual_usd"]), utc_now(), run_id),
                )
                self.conn.commit()

    def _resolve_ticket(
        self, run: dict[str, Any], ticket: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self.router_store is None:
            return None
        from router_runtime import persist_dispatch_trace, resolve_for_dispatch

        card = {}
        if ticket.get("card_id"):
            try:
                card = self.kanban_store.get_card(ticket["card_id"])
            except Exception:
                card = {}
        task_text = (
            f"{ticket.get('title') or ''}\n"
            f"{card.get('description') or ticket.get('description') or ''}"
        ).strip()
        routing = resolve_for_dispatch(
            self.router_store,
            task_text=task_text,
            run_id=run["id"],
            card_id=ticket.get("card_id"),
            connected_providers=self.connected_providers,
            local_runtimes={},
        )
        persist_dispatch_trace(
            routing,
            kanban_store=self.kanban_store,
            card_id=ticket.get("card_id"),
        )
        return routing

    def _implement_ticket(
        self,
        run: dict[str, Any],
        ticket: dict[str, Any],
        routing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        branch = f"feat/{slugify(ticket['title'])}"
        worktree_dir = None
        if self.project_store is not None:
            tree = self.project_store.create_worktree(run["project_id"], branch)
            worktree_dir = tree["path"]
        if self.coordinator is not None:
            session = self.coordinator.assign_card(run["project_id"], ticket["card_id"])
        elif self.fleet_store is not None:
            session = self.fleet_store.spawn_session(
                run["project_id"],
                ticket["card_id"],
                branch_name=branch,
            )
        else:
            raise FeatureDevError("fleet store is required to implement tickets")
        actual = 0.0
        error = None
        status = TICKET_PASSED
        if self.commit_loop is not None:
            result = self.commit_loop.run_session(
                session["id"],
                worktree_dir=worktree_dir,
            )
            actual = float((result.get("run") or {}).get("total_cost_usd") or 0)
            if result.get("status") != STATUS_PASSED:
                status = TICKET_FAILED
                error = json.dumps(result.get("run") or {"status": result.get("status")})
                if self.router_store is not None and routing is not None:
                    from router_runtime import escalate_on_struggle, record_outcome

                    decision = routing.get("decision") or {}
                    record_outcome(
                        "coding",
                        str(decision.get("provider_key") or ""),
                        str(decision.get("model") or ""),
                        False,
                    )
                    if worktree_dir:
                        escalate_on_struggle(
                            self.router_store,
                            task_text=str(ticket.get("title") or ""),
                            failed_result=routing,
                            failed_output=error,
                            worktree_dir=worktree_dir,
                            branch=session.get("branch_name") or branch,
                            ticket=str(ticket.get("title") or ""),
                            run_id=run["id"],
                            card_id=ticket.get("card_id"),
                            connected_providers=self.connected_providers,
                            local_runtimes={},
                            runner=self.dispatch_runner,
                        )
            elif self.router_store is not None and routing is not None:
                from router_runtime import record_outcome

                decision = routing.get("decision") or {}
                record_outcome(
                    "coding",
                    str(decision.get("provider_key") or ""),
                    str(decision.get("model") or ""),
                    True,
                )
        return {
            "status": status,
            "actual_usd": actual,
            "branch_name": session.get("branch_name") or branch,
            "session_id": session["id"],
            "error": error,
        }

    def _finalize(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] in {STATUS_ABORTED, STATUS_PAUSED, STATUS_FAILED}:
            if run["status"] == STATUS_FAILED:
                return run
            if run["status"] == STATUS_PAUSED:
                return run
            return run
        statuses = {ticket["status"] for ticket in run["tickets"]}
        if statuses <= {TICKET_PASSED}:
            final = STATUS_PASSED
        elif TICKET_FAILED in statuses and TICKET_PASSED in statuses:
            final = STATUS_PARTIAL
        elif TICKET_FAILED in statuses:
            final = STATUS_FAILED
        else:
            final = STATUS_PARTIAL
        self._set_run(run_id, status=final)
        return self.get_run(run_id)

    def _project_cap(self, project_id: str) -> float | None:
        if self.cost_cap is not None:
            return self.cost_cap
        if self.project_store is None:
            return None
        try:
            project = self.project_store.get_project(project_id)
        except Exception:
            return None
        cap = project.get("cost_cap")
        return float(cap) if cap is not None else None

    def _ticket(self, run_id: str, ticket_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        for ticket in run["tickets"]:
            if ticket["id"] == ticket_id:
                return ticket
        raise NotFoundError(f"Ticket {ticket_id} not found")

    def _patch_ticket(self, ticket_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock:
            self.conn.execute(
                f"UPDATE feature_dev_tickets SET {assignments} WHERE id = ?",
                (*fields.values(), ticket_id),
            )
            self.conn.commit()

    def _set_run(self, run_id: str, **fields: Any) -> None:
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock:
            self.conn.execute(
                f"UPDATE feature_dev_runs SET {assignments} WHERE id = ?",
                (*fields.values(), run_id),
            )
            self.conn.commit()
