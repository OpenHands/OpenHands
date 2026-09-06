"""Pre-built commit loop: lint → typecheck → test → build, then git commit.

On iterative stage failure the run stays open. A ``fix_agent`` callback
(or the caller) applies a patch, then the service retries until green or
the loop's iteration/cost cap fires. A green run commits on the worktree
branch and moves the linked kanban card to Review.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Callable

from kanban import KanbanStore
from kanban_agent import complete_session
from loop_runner import (
    DEFAULT_MAX_COST_USD,
    DEFAULT_MAX_ITERATIONS,
    LoopStore,
    ON_FAILURE_AUTO_FIX,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_RUNNING,
)
from pr_creator import (
    CONVENTIONAL_TYPES,
    conventional_commit_message,
    create_pull_request_from_session,
)

COMMIT_LOOP_NAME = "commit-loop"
COMMIT_LOOP_STAGES: list[dict[str, Any]] = [
    {"name": "lint", "cmd": None, "iterative": True},
    {"name": "typecheck", "cmd": None, "iterative": True},
    {"name": "test", "cmd": None, "iterative": True},
    {"name": "build", "cmd": None, "iterative": True},
]
FixAgent = Callable[[dict[str, Any]], None]


class CommitLoopError(Exception):
    """Raised for invalid commit-loop operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def commit_type_for_card(card: dict[str, Any]) -> str:
    kind = str(card.get("type") or "").strip().lower()
    if kind in CONVENTIONAL_TYPES:
        return kind
    title = str(card.get("title") or "").lower()
    if "fix" in title or "bug" in title:
        return "fix"
    return "feat"


def _run_git(cwd: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise CommitLoopError(detail or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def changed_files(worktree_dir: str) -> list[str]:
    status = _run_git(worktree_dir, "status", "--porcelain")
    files: list[str] = []
    for line in status.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            files.append(path)
    return files


class CommitLoopService:
    """Registers and runs the commit loop for a fleet session worktree."""

    def __init__(
        self,
        loop_store: LoopStore | None = None,
        kanban_store: KanbanStore | None = None,
        fleet_store: Any | None = None,
        project_store: Any | None = None,
        fix_agent: FixAgent | None = None,
        router_store: Any | None = None,
        dispatch_runner: Any | None = None,
        connected_providers: list[str] | None = None,
    ) -> None:
        self.loop_store = loop_store or LoopStore()
        self.kanban_store = kanban_store
        self.fleet_store = fleet_store
        self.project_store = project_store
        self.fix_agent = fix_agent
        self.router_store = router_store
        self.dispatch_runner = dispatch_runner
        self.connected_providers = connected_providers
        self._owns_loop_store = loop_store is None

    def close(self) -> None:
        if self._owns_loop_store:
            self.loop_store.close()

    def setup(
        self,
        project_id: str,
        max_iterations: int | None = None,
        max_cost_usd: float | None = None,
    ) -> dict[str, Any]:
        project_id = (project_id or "").strip()
        if not project_id:
            raise CommitLoopError("project_id is required")
        existing = self._definition_for(project_id)
        if existing is not None:
            return existing
        return self.loop_store.create_definition(
            name=COMMIT_LOOP_NAME,
            project_id=project_id,
            stages=COMMIT_LOOP_STAGES,
            max_iterations=max_iterations or DEFAULT_MAX_ITERATIONS,
            max_cost_usd=max_cost_usd if max_cost_usd is not None else DEFAULT_MAX_COST_USD,
            on_failure=ON_FAILURE_AUTO_FIX,
        )

    def status(self, project_id: str) -> dict[str, Any]:
        definition = self._definition_for(project_id)
        if definition is None:
            raise CommitLoopError(
                f"commit-loop is not set up for project {project_id}",
                status=404,
            )
        runs = self.loop_store.list_runs(definition["id"])
        last = runs[-1] if runs else None
        return {"definition": definition, "last_run": last}

    def run_session(
        self,
        session_id: str,
        *,
        worktree_dir: str | None = None,
        push_pr: bool = False,
        fix_agent: FixAgent | None = None,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        project_id = str(session["project_id"])
        definition = self.setup(project_id)
        worktree = worktree_dir or self._worktree_dir(session)
        card = self._card(session)
        run = self.loop_store.start_run(
            definition["id"],
            worktree_dir=worktree,
            session_id=session_id,
        )
        routing = None
        if self.router_store is not None:
            from router_runtime import persist_dispatch_trace, resolve_for_dispatch

            routing = resolve_for_dispatch(
                self.router_store,
                task_text=str(
                    card.get("title") or session.get("branch_name") or "commit loop"
                ),
                run_id=run["id"],
                card_id=card.get("id"),
                connected_providers=self.connected_providers,
                local_runtimes={},
            )
            persist_dispatch_trace(
                routing,
                worktree_dir=worktree,
                kanban_store=self.kanban_store,
                card_id=card.get("id"),
            )
        fixer = fix_agent or self.fix_agent
        while run["status"] == STATUS_RUNNING:
            payload = self.loop_store.request_fix(run["id"])
            ctx = {
                **payload,
                "changed_files": changed_files(worktree),
                "worktree_dir": worktree,
                "session": session,
            }
            if (
                self.router_store is not None
                and routing is not None
                and worktree
            ):
                from router_runtime import escalate_on_struggle

                nxt = escalate_on_struggle(
                    self.router_store,
                    task_text=str(card.get("title") or ""),
                    failed_result=routing,
                    failed_output=str(payload.get("last_output") or ""),
                    worktree_dir=worktree,
                    branch=str(session.get("branch_name") or ""),
                    ticket=str(card.get("title") or ""),
                    run_id=run["id"],
                    card_id=card.get("id"),
                    connected_providers=self.connected_providers,
                    local_runtimes={},
                    runner=self.dispatch_runner,
                    stage_type=str(payload.get("stage_name") or "fix"),
                )
                if nxt.get("switched"):
                    routing = nxt
                    ctx["resume_prompt"] = nxt.get("resume_prompt")
                    ctx["routing"] = nxt
            if fixer is None:
                break
            fixer(ctx)
            run = self.loop_store.retry_stage(run["id"])
        if run["status"] != STATUS_PASSED:
            return {
                "status": run["status"] or STATUS_FAILED,
                "run": run,
                "commit_sha": None,
                "commit_message": None,
                "routing": routing,
            }
        card = self._card(session)
        message = conventional_commit_message(
            commit_type_for_card(card),
            str(card.get("title") or "change"),
        )
        sha = self._commit(worktree, message)
        from loop_triggers import notify_commit

        notify_commit(
            project_id,
            str(session.get("branch_name") or ""),
            sha,
            worktree_dir=worktree,
        )
        if self.kanban_store is not None and card.get("id"):
            complete_session(
                self.kanban_store,
                card["id"],
                tests_passed=True,
                actual_cost=float(run["total_cost_usd"] or 0),
                agent_time=float(run["iteration"] or 0),
                model_used=((routing or {}).get("decision") or {}).get("model"),
            )
            self.kanban_store.update_card(
                card["id"],
                linked_branch=session.get("branch_name"),
            )
        pr = None
        if push_pr:
            pr = create_pull_request_from_session(
                worktree,
                session_id=session_id,
                short_description=str(card.get("title") or "change"),
                session_summary=message,
                commit_type=commit_type_for_card(card),
                kanban_store=self.kanban_store,
                card_id=card.get("id"),
                project_id=project_id,
            )
        return {
            "status": STATUS_PASSED,
            "run": run,
            "commit_sha": sha,
            "commit_message": message,
            "pr": pr,
            "routing": routing,
        }

    def _definition_for(self, project_id: str) -> dict[str, Any] | None:
        for definition in self.loop_store.list_definitions():
            if (
                definition["project_id"] == project_id
                and definition["name"] == COMMIT_LOOP_NAME
            ):
                return definition
        return None

    def _session(self, session_id: str) -> dict[str, Any]:
        session_id = (session_id or "").strip()
        if not session_id:
            raise CommitLoopError("session_id is required")
        if self.fleet_store is None:
            raise CommitLoopError("fleet store is required")
        try:
            return self.fleet_store.get_session(session_id)
        except Exception as exc:
            raise CommitLoopError(f"Session {session_id} not found", status=404) from exc

    def _card(self, session: dict[str, Any]) -> dict[str, Any]:
        card_id = session.get("card_id")
        if self.kanban_store is None or not card_id:
            return {"title": session.get("branch_name") or "change"}
        return self.kanban_store.get_card(card_id)

    def _worktree_dir(self, session: dict[str, Any]) -> str:
        if self.project_store is not None:
            for tree in self.project_store.list_worktrees(session["project_id"]):
                if tree["branch_name"] == session["branch_name"]:
                    return str(tree["path"])
        raise CommitLoopError("worktree_dir is required when no project store is linked")

    def _commit(self, worktree_dir: str, message: str) -> str:
        status = _run_git(worktree_dir, "status", "--porcelain")
        if status:
            _run_git(worktree_dir, "add", "-A")
            _run_git(worktree_dir, "commit", "-m", message)
        elif "nothing to commit" in _run_git(worktree_dir, "status"):
            # ponytail: empty commit keeps the loop honest when scripts didn't dirty the tree
            env = os.environ.copy()
            result = subprocess.run(
                ["git", "commit", "--allow-empty", "-m", message],
                cwd=worktree_dir,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                raise CommitLoopError((result.stderr or result.stdout or "").strip())
        return _run_git(worktree_dir, "rev-parse", "HEAD")
