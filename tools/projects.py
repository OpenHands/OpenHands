"""Local project and git worktree persistence backed by SQLite.

Projects and their worktrees live beside the kanban store under
``~/.openhands/agent-canvas/``. Git clone/init and ``git worktree``
commands run against each project's ``local_path``.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

PROJECT_STATUSES = ("active", "idle", "error")
WORKTREE_STATUSES = ("idle", "working", "reviewing", "ci", "merged", "error")
DEFAULT_PROJECT_STATUS = "idle"
DEFAULT_WORKTREE_STATUS = "idle"
DEFAULT_BRANCH = "main"
ASSIGNED_WORKTREE_STATUS = "working"
PROJECTS_DB_FILENAME = "projects.sqlite"
WORKTREES_DIRNAME = ".worktrees"
DEFAULT_PROJECTS_DIRNAME = "openhands-projects"
PROJECT_PATCH_FIELDS = (
    "name",
    "description",
    "repo_url",
    "local_path",
    "default_branch",
    "default_agent_profile",
    "kanban_board_id",
    "cost_cap",
    "status",
)

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class ProjectError(Exception):
    """Raised for invalid project operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(ProjectError):
    """Raised when a project or worktree does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, PROJECTS_DB_FILENAME)


def default_projects_root() -> str:
    path = os.path.join(os.path.expanduser("~"), DEFAULT_PROJECTS_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip()).strip("-").lower()
    return slug or "project"


def worktree_dirname(branch_name: str) -> str:
    return branch_name.replace("/", "--")


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def run_git(args: list[str], cwd: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ProjectError(detail or f"git {' '.join(args)} failed")
    return result.stdout


class ProjectStore:
    """CRUD store for projects and git worktrees."""

    def __init__(
        self,
        db_path: str = ":memory:",
        projects_root: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.projects_root = projects_root or default_projects_root()
        os.makedirs(self.projects_root, exist_ok=True)
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
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                repo_url TEXT,
                local_path TEXT NOT NULL,
                default_branch TEXT NOT NULL,
                default_agent_profile TEXT,
                kanban_board_id TEXT,
                cost_cap REAL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS worktrees (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT NOT NULL,
                agent_session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def create_project(
        self,
        name: str,
        description: str | None = None,
        repo_url: str | None = None,
        local_path: str | None = None,
        default_branch: str | None = None,
        default_agent_profile: str | None = None,
        kanban_board_id: str | None = None,
        cost_cap: float | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ProjectError("Project name is required")
        branch = (default_branch or DEFAULT_BRANCH).strip() or DEFAULT_BRANCH
        resolved_status = status or DEFAULT_PROJECT_STATUS
        self._validate_status(resolved_status, PROJECT_STATUSES, "status")
        if cost_cap is not None:
            cost_cap = self._validate_cost_cap(cost_cap)
        path = os.path.abspath(
            os.path.expanduser(local_path)
            if local_path
            else os.path.join(self.projects_root, slugify(name))
        )
        self._materialize_repo(path, repo_url=repo_url, branch=branch)
        project_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO projects (
                    id, name, description, repo_url, local_path, default_branch,
                    default_agent_profile, kanban_board_id, cost_cap, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    name,
                    description,
                    repo_url,
                    path,
                    branch,
                    default_agent_profile,
                    kanban_board_id,
                    cost_cap,
                    resolved_status,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM projects ORDER BY created_at ASC"
            ).fetchall()
            counts = {
                row["project_id"]: row["count"]
                for row in self.conn.execute(
                    """
                    SELECT project_id, COUNT(*) AS count
                    FROM worktrees
                    GROUP BY project_id
                    """
                ).fetchall()
            }
        projects = []
        for row in rows:
            payload = _row_to_dict(row)
            assert payload is not None
            payload["worktree_count"] = int(counts.get(payload["id"], 0))
            projects.append(payload)
        return projects

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            project = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
            )
            if project is None:
                raise NotFoundError(f"Project {project_id} not found")
            worktrees = self.conn.execute(
                """
                SELECT * FROM worktrees
                WHERE project_id = ?
                ORDER BY created_at ASC
                """,
                (project_id,),
            ).fetchall()
        project["worktrees"] = [_row_to_dict(row) for row in worktrees]
        project["worktree_count"] = len(project["worktrees"])
        return project

    def update_project(self, project_id: str, **fields: Any) -> dict[str, Any]:
        self.get_project(project_id)
        updates: dict[str, Any] = {}
        for key in PROJECT_PATCH_FIELDS:
            if key not in fields:
                continue
            value = fields[key]
            if key == "name":
                value = (value or "").strip()
                if not value:
                    raise ProjectError("Project name is required")
            if key == "status" and value is not None:
                self._validate_status(value, PROJECT_STATUSES, "status")
            if key == "cost_cap" and value is not None:
                value = self._validate_cost_cap(value)
            if key == "default_branch" and value is not None:
                value = str(value).strip() or DEFAULT_BRANCH
            updates[key] = value
        if not updates:
            return self.get_project(project_id)
        updates["updated_at"] = utc_now()
        with self._lock:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            self.conn.execute(
                f"UPDATE projects SET {assignments} WHERE id = ?",
                (*updates.values(), project_id),
            )
            self.conn.commit()
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> None:
        project = self.get_project(project_id)
        for worktree in list(project["worktrees"]):
            self.remove_worktree(project_id, worktree["id"])
        with self._lock:
            self.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self.conn.commit()

    def list_worktrees(self, project_id: str) -> list[dict[str, Any]]:
        return self.get_project(project_id)["worktrees"]

    def create_worktree(
        self,
        project_id: str,
        branch_name: str,
        status: str | None = None,
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        branch_name = (branch_name or "").strip()
        if not branch_name:
            raise ProjectError("branch_name is required")
        resolved_status = status or DEFAULT_WORKTREE_STATUS
        self._validate_status(
            resolved_status, WORKTREE_STATUSES, "worktree status"
        )
        path = os.path.join(
            project["local_path"],
            WORKTREES_DIRNAME,
            worktree_dirname(branch_name),
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._add_worktree(project["local_path"], path, branch_name, project["default_branch"])
        worktree_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO worktrees (
                    id, project_id, branch_name, path, status,
                    agent_session_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    worktree_id,
                    project_id,
                    branch_name,
                    path,
                    resolved_status,
                    None,
                    now,
                    now,
                ),
            )
            self.conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                ("active", now, project_id),
            )
            self.conn.commit()
        return self.get_worktree(project_id, worktree_id)

    def get_worktree(self, project_id: str, worktree_id: str) -> dict[str, Any]:
        self.get_project(project_id)
        with self._lock:
            worktree = _row_to_dict(
                self.conn.execute(
                    """
                    SELECT * FROM worktrees
                    WHERE id = ? AND project_id = ?
                    """,
                    (worktree_id, project_id),
                ).fetchone()
            )
        if worktree is None:
            raise NotFoundError(f"Worktree {worktree_id} not found")
        return worktree

    def assign_worktree(
        self,
        project_id: str,
        worktree_id: str,
        agent_session_id: str,
    ) -> dict[str, Any]:
        agent_session_id = (agent_session_id or "").strip()
        if not agent_session_id:
            raise ProjectError("agent_session_id is required")
        self.get_worktree(project_id, worktree_id)
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                UPDATE worktrees
                SET agent_session_id = ?, status = ?, updated_at = ?
                WHERE id = ? AND project_id = ?
                """,
                (
                    agent_session_id,
                    ASSIGNED_WORKTREE_STATUS,
                    now,
                    worktree_id,
                    project_id,
                ),
            )
            self.conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                ("active", now, project_id),
            )
            self.conn.commit()
        return self.get_worktree(project_id, worktree_id)

    def remove_worktree(self, project_id: str, worktree_id: str) -> None:
        project = self.get_project(project_id)
        worktree = self.get_worktree(project_id, worktree_id)
        self._remove_worktree(project["local_path"], worktree["path"])
        with self._lock:
            self.conn.execute(
                "DELETE FROM worktrees WHERE id = ?", (worktree_id,)
            )
            remaining = self.conn.execute(
                "SELECT COUNT(*) AS count FROM worktrees WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            next_status = (
                "active" if int(remaining["count"]) > 0 else "idle"
            )
            self.conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                (next_status, utc_now(), project_id),
            )
            self.conn.commit()

    def _materialize_repo(
        self,
        path: str,
        repo_url: str | None,
        branch: str,
    ) -> None:
        git_dir = os.path.join(path, ".git")
        if repo_url:
            if os.path.exists(path) and os.listdir(path):
                if not os.path.exists(git_dir):
                    raise ProjectError(f"Refusing to clone into non-empty path {path}")
                return
            parent = os.path.dirname(path)
            os.makedirs(parent, exist_ok=True)
            run_git(["clone", "--", repo_url, path])
            return
        if os.path.exists(git_dir):
            return
        os.makedirs(path, exist_ok=True)
        run_git(["init", "-b", branch], cwd=path)

    def _branch_exists(self, repo: str, branch_name: str) -> bool:
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _add_worktree(
        self,
        repo: str,
        path: str,
        branch_name: str,
        default_branch: str,
    ) -> None:
        if os.path.exists(path):
            raise ProjectError(f"Worktree path already exists: {path}")
        if self._branch_exists(repo, branch_name):
            run_git(["worktree", "add", path, branch_name], cwd=repo)
            return
        start_point = (
            default_branch
            if self._branch_exists(repo, default_branch)
            else "HEAD"
        )
        run_git(
            ["worktree", "add", "-b", branch_name, path, start_point],
            cwd=repo,
        )

    def _remove_worktree(self, repo: str, path: str) -> None:
        if not os.path.exists(path) and not os.path.exists(repo):
            return
        try:
            run_git(["worktree", "remove", "--force", path], cwd=repo)
        except ProjectError:
            # ponytail: force-remove leftover dirs if git already dropped the worktree
            if os.path.isdir(path):
                run_git(["worktree", "prune"], cwd=repo)

    def _validate_status(
        self, status: Any, allowed: tuple[str, ...], label: str
    ) -> None:
        if status not in allowed:
            raise ProjectError(
                f"{label} must be one of {', '.join(allowed)}"
            )

    def _validate_cost_cap(self, cost_cap: Any) -> float:
        try:
            value = float(cost_cap)
        except (TypeError, ValueError) as exc:
            raise ProjectError("cost_cap must be a number") from exc
        if value < 0:
            raise ProjectError("cost_cap must be >= 0")
        return value
