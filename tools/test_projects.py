"""Unit and API tests for the local project / worktree store.

Run from the repo root:

    python3 tools/test_projects.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from threading import Thread
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from projects import (  # noqa: E402
    DEFAULT_BRANCH,
    PROJECT_STATUSES,
    NotFoundError,
    ProjectError,
    ProjectStore,
)
from projects_api import handle_request, serve_projects  # noqa: E402


def _git(args: list[str], cwd: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path: str, branch: str = "main") -> None:
    os.makedirs(path, exist_ok=True)
    _git(["init", "-b", branch], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("hello\n")
    _git(["add", "."], path)
    _git(["commit", "-m", "init"], path)


def _request(
    store: ProjectStore,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return handle_request(store, method, path, body)


class ProjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.projects_root = os.path.join(self.tmp.name, "projects")
        os.makedirs(self.projects_root)
        self.store = ProjectStore(
            db_path=":memory:",
            projects_root=self.projects_root,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_create_project_inits_git_repo(self) -> None:
        project = self.store.create_project("Alpha")
        self.assertEqual(project["name"], "Alpha")
        self.assertEqual(project["status"], "idle")
        self.assertEqual(project["default_branch"], DEFAULT_BRANCH)
        self.assertTrue(os.path.isdir(os.path.join(project["local_path"], ".git")))
        self.assertIn(project["status"], PROJECT_STATUSES)
        self.assertEqual(project["worktrees"], [])

    def test_create_project_clones_repo_url(self) -> None:
        source = os.path.join(self.tmp.name, "source")
        _init_repo(source)
        project = self.store.create_project(
            "Cloned",
            repo_url=source,
        )
        self.assertEqual(project["repo_url"], source)
        readme = os.path.join(project["local_path"], "README.md")
        self.assertTrue(os.path.isfile(readme))

    def test_create_project_registers_existing_local_path(self) -> None:
        existing = os.path.join(self.tmp.name, "existing")
        _init_repo(existing)
        project = self.store.create_project("Existing", local_path=existing)
        self.assertEqual(project["local_path"], existing)

    def test_create_project_rejects_blank_name(self) -> None:
        with self.assertRaises(ProjectError):
            self.store.create_project("  ")

    def test_list_and_get_project(self) -> None:
        first = self.store.create_project("One")
        self.store.create_project("Two")
        listed = self.store.list_projects()
        self.assertEqual({item["name"] for item in listed}, {"One", "Two"})
        loaded = self.store.get_project(first["id"])
        self.assertEqual(loaded["id"], first["id"])
        self.assertEqual(loaded["worktrees"], [])

    def test_update_project_metadata(self) -> None:
        project = self.store.create_project("Meta")
        updated = self.store.update_project(
            project["id"],
            description="A space",
            cost_cap=25.5,
            default_agent_profile="default",
            status="active",
        )
        self.assertEqual(updated["description"], "A space")
        self.assertAlmostEqual(updated["cost_cap"], 25.5)
        self.assertEqual(updated["default_agent_profile"], "default")
        self.assertEqual(updated["status"], "active")

    def test_update_rejects_invalid_status_and_cost_cap(self) -> None:
        project = self.store.create_project("Bad")
        with self.assertRaises(ProjectError):
            self.store.update_project(project["id"], status="paused")
        with self.assertRaises(ProjectError):
            self.store.update_project(project["id"], cost_cap=-1)

    def test_unknown_project_raises_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.store.get_project("missing")

    def test_create_and_list_worktree(self) -> None:
        source = os.path.join(self.tmp.name, "wt-src")
        _init_repo(source)
        project = self.store.create_project("Worktrees", local_path=source)
        worktree = self.store.create_worktree(project["id"], "feature/spaces")
        self.assertEqual(worktree["branch_name"], "feature/spaces")
        self.assertEqual(worktree["status"], "idle")
        self.assertTrue(os.path.isdir(worktree["path"]))
        loaded = self.store.get_project(project["id"])
        self.assertEqual(len(loaded["worktrees"]), 1)
        listed = self.store.list_worktrees(project["id"])
        self.assertEqual(listed[0]["id"], worktree["id"])

    def test_assign_agent_to_worktree(self) -> None:
        source = os.path.join(self.tmp.name, "assign-src")
        _init_repo(source)
        project = self.store.create_project("Assign", local_path=source)
        worktree = self.store.create_worktree(project["id"], "agent-branch")
        assigned = self.store.assign_worktree(
            project["id"], worktree["id"], "session-123"
        )
        self.assertEqual(assigned["agent_session_id"], "session-123")
        self.assertEqual(assigned["status"], "working")

    def test_remove_worktree(self) -> None:
        source = os.path.join(self.tmp.name, "rm-src")
        _init_repo(source)
        project = self.store.create_project("Remove", local_path=source)
        worktree = self.store.create_worktree(project["id"], "temp-branch")
        path = worktree["path"]
        self.store.remove_worktree(project["id"], worktree["id"])
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.store.list_worktrees(project["id"]), [])

    def test_delete_project_cleans_worktrees(self) -> None:
        source = os.path.join(self.tmp.name, "del-src")
        _init_repo(source)
        project = self.store.create_project("Delete me", local_path=source)
        worktree = self.store.create_worktree(project["id"], "gone")
        path = worktree["path"]
        self.store.delete_project(project["id"])
        self.assertFalse(os.path.exists(path))
        with self.assertRaises(NotFoundError):
            self.store.get_project(project["id"])

    def test_file_db_survives_reopen(self) -> None:
        db_path = os.path.join(self.tmp.name, "projects.sqlite")
        store = ProjectStore(db_path=db_path, projects_root=self.projects_root)
        project = store.create_project("Persisted")
        store.close()
        reopened = ProjectStore(
            db_path=db_path, projects_root=self.projects_root
        )
        loaded = reopened.get_project(project["id"])
        self.assertEqual(loaded["name"], "Persisted")
        reopened.close()


class ProjectApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.projects_root = os.path.join(self.tmp.name, "projects")
        os.makedirs(self.projects_root)
        self.store = ProjectStore(
            db_path=":memory:",
            projects_root=self.projects_root,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_list_and_create_projects(self) -> None:
        status, listed = _request(self.store, "GET", "/api/projects")
        self.assertEqual(status, 200)
        self.assertEqual(listed, [])

        status, created = _request(
            self.store,
            "POST",
            "/api/projects",
            {"name": "Roadmap", "description": "Spaces", "cost_cap": 10},
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["name"], "Roadmap")
        self.assertAlmostEqual(created["cost_cap"], 10)

        status, listed = _request(self.store, "GET", "/api/projects")
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

    def test_get_patch_delete_project(self) -> None:
        _, created = _request(
            self.store, "POST", "/api/projects", {"name": "Patch"}
        )
        project_id = created["id"]
        status, loaded = _request(
            self.store, "GET", f"/api/projects/{project_id}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(loaded["worktrees"], [])

        status, updated = _request(
            self.store,
            "PATCH",
            f"/api/projects/{project_id}",
            {"description": "updated", "kanban_board_id": "board-1"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["description"], "updated")
        self.assertEqual(updated["kanban_board_id"], "board-1")

        status, _ = _request(
            self.store, "DELETE", f"/api/projects/{project_id}"
        )
        self.assertEqual(status, 204)
        status, payload = _request(
            self.store, "GET", f"/api/projects/{project_id}"
        )
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_worktree_lifecycle_endpoints(self) -> None:
        source = os.path.join(self.tmp.name, "api-src")
        _init_repo(source)
        _, project = _request(
            self.store,
            "POST",
            "/api/projects",
            {"name": "API WT", "local_path": source},
        )
        project_id = project["id"]
        status, worktree = _request(
            self.store,
            "POST",
            f"/api/projects/{project_id}/worktrees",
            {"branch_name": "feature/pr"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(worktree["branch_name"], "feature/pr")

        status, listed = _request(
            self.store, "GET", f"/api/projects/{project_id}/worktrees"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

        status, assigned = _request(
            self.store,
            "POST",
            f"/api/projects/{project_id}/worktrees/{worktree['id']}/assign",
            {"agent_session_id": "sess-9"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(assigned["agent_session_id"], "sess-9")

        status, _ = _request(
            self.store,
            "DELETE",
            f"/api/projects/{project_id}/worktrees/{worktree['id']}",
        )
        self.assertEqual(status, 204)

    def test_missing_routes_and_validation(self) -> None:
        status, payload = _request(self.store, "GET", "/api/projects/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

        status, payload = _request(self.store, "POST", "/api/projects", {})
        self.assertEqual(status, 400)

        status, payload = _request(self.store, "GET", "/api/unknown")
        self.assertEqual(status, 404)


class ProjectHttpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(
            db_path=":memory:",
            projects_root=os.path.join(self.tmp.name, "projects"),
        )
        self.server = serve_projects("127.0.0.1", 0, store=self.store)
        self.port = self.server.server_address[1]
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.store.close()
        self.tmp.cleanup()

    def _http(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int, Any]:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        parsed = json.loads(raw) if raw else None
        return response.status, parsed

    def test_http_roundtrip(self) -> None:
        status, project = self._http("POST", "/api/projects", {"name": "HTTP"})
        self.assertEqual(status, 201)
        status, listed = self._http("GET", "/api/projects")
        self.assertEqual(status, 200)
        self.assertEqual(listed[0]["id"], project["id"])


if __name__ == "__main__":
    unittest.main()
