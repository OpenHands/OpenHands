"""Unit and API tests for fleet sessions and cost caps.

Run from the repo root:

    python3 tools/test_fleet.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from threading import Thread
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fleet import (  # noqa: E402
    ACTIVE_STATUS,
    ARCHIVED_STATUS,
    DEFAULT_RESET_PERIOD,
    SESSION_STATUSES,
    CapExceededError,
    FleetError,
    FleetStore,
    NotFoundError,
)
from fleet_api import handle_request, serve_fleet  # noqa: E402
from kanban import KanbanStore  # noqa: E402
from projects import ProjectStore  # noqa: E402


def _request(
    store: FleetStore,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return handle_request(store, method, path, body)


def _git(args: list[str], cwd: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    _git(["init", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("hello\n")
    _git(["add", "."], path)
    _git(["commit", "-m", "init"], path)


class FleetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kanban = KanbanStore(":memory:")
        self.store = FleetStore(":memory:", kanban_store=self.kanban)
        self.board = self.kanban.create_board("Work", project_id="proj-1")
        self.card = self.kanban.create_card(
            self.board["columns"][0]["id"],
            title="Ship fleet",
            actual_cost=1.25,
            model_used="claude-sonnet",
            agent_time=90,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.kanban.close()

    def test_spawn_creates_session_and_links_card(self) -> None:
        session = self.store.spawn_session(
            project_id="proj-1",
            card_id=self.card["id"],
            model="claude-sonnet",
        )
        self.assertEqual(session["project_id"], "proj-1")
        self.assertEqual(session["card_id"], self.card["id"])
        self.assertEqual(session["status"], ACTIVE_STATUS)
        self.assertEqual(session["model"], "claude-sonnet")
        self.assertIn(session["status"], SESSION_STATUSES)
        self.assertTrue(session["branch_name"])
        self.assertTrue(session["agent_session_id"])
        self.assertAlmostEqual(session["cost"], 1.25)
        self.assertEqual(session["duration_seconds"], 90)
        linked = self.kanban.get_card(self.card["id"])
        self.assertEqual(linked["agent_session_id"], session["agent_session_id"])
        self.assertEqual(linked["linked_branch"], session["branch_name"])

    def test_list_filters_by_project_status_model_and_date(self) -> None:
        first = self.store.spawn_session("proj-1", self.card["id"], model="claude-sonnet")
        other_card = self.kanban.create_card(
            self.board["columns"][0]["id"], title="Other"
        )
        self.store.spawn_session("proj-2", other_card["id"], model="glm-5.2")
        self.store.update_session(first["id"], status="done")

        active = self.store.list_sessions()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["project_id"], "proj-2")

        by_project = self.store.list_sessions(project_id="proj-1", status="done")
        self.assertEqual(len(by_project), 1)
        self.assertEqual(by_project[0]["id"], first["id"])

        by_model = self.store.list_sessions(status="done", model="claude-sonnet")
        self.assertEqual(len(by_model), 1)

        future = (
            datetime.now(timezone.utc) + timedelta(days=1)
        ).replace(microsecond=0).isoformat()
        none = self.store.list_sessions(status="done", created_after=future)
        self.assertEqual(none, [])

    def test_cost_cap_rejects_new_sessions(self) -> None:
        self.store.set_cost_cap("proj-1", cap_usd=1.0)
        with self.assertRaises(CapExceededError):
            self.store.spawn_session("proj-1", self.card["id"])
        self.assertEqual(self.store.list_sessions(project_id="proj-1"), [])

    def test_set_and_list_cost_caps(self) -> None:
        cap = self.store.set_cost_cap("proj-1", cap_usd=50, reset_period="weekly")
        self.assertEqual(cap["project_id"], "proj-1")
        self.assertAlmostEqual(cap["cap_usd"], 50)
        self.assertEqual(cap["reset_period"], "weekly")
        listed = self.store.list_cost_caps()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], cap["id"])
        self.assertEqual(DEFAULT_RESET_PERIOD, "never")

    def test_stats_and_archive(self) -> None:
        session = self.store.spawn_session("proj-1", self.card["id"])
        stats = self.store.stats()
        self.assertEqual(stats["total_sessions"], 1)
        self.assertEqual(stats["active_sessions"], 1)
        self.assertAlmostEqual(stats["total_cost"], 1.25)
        self.assertEqual(stats["total_duration_seconds"], 90)

        archived = self.store.archive_session(session["id"])
        self.assertEqual(archived["status"], ARCHIVED_STATUS)
        self.assertEqual(self.store.list_sessions(), [])
        stats = self.store.stats()
        self.assertEqual(stats["active_sessions"], 0)
        self.assertEqual(stats["total_sessions"], 1)

    def test_unknown_session_raises_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.store.get_session("missing")

    def test_spawn_creates_project_worktree(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = os.path.join(tmp.name, "repo")
        _init_repo(source)
        projects = ProjectStore(
            db_path=":memory:",
            projects_root=os.path.join(tmp.name, "projects"),
        )
        self.addCleanup(projects.close)
        project = projects.create_project("Alpha", local_path=source)
        self.store.project_store = projects
        session = self.store.spawn_session(project["id"], self.card["id"])
        worktrees = projects.list_worktrees(project["id"])
        self.assertEqual(len(worktrees), 1)
        self.assertEqual(worktrees[0]["branch_name"], session["branch_name"])
        self.assertEqual(worktrees[0]["agent_session_id"], session["agent_session_id"])
        self.assertTrue(os.path.isdir(worktrees[0]["path"]))


class FleetApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kanban = KanbanStore(":memory:")
        self.store = FleetStore(":memory:", kanban_store=self.kanban)
        self.board = self.kanban.create_board("Work", project_id="proj-1")
        self.card = self.kanban.create_card(
            self.board["columns"][0]["id"], title="API card"
        )

    def tearDown(self) -> None:
        self.store.close()
        self.kanban.close()

    def test_session_and_cap_endpoints(self) -> None:
        status, session = _request(
            self.store,
            "POST",
            "/api/fleet/sessions",
            {"project_id": "proj-1", "card_id": self.card["id"], "model": "glm-5.2"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(session["model"], "glm-5.2")

        status, listed = _request(self.store, "GET", "/api/fleet/sessions")
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

        status, listed = _request(
            self.store,
            "GET",
            f"/api/fleet/sessions?project_id=proj-1&status=active&model=glm-5.2",
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

        status, loaded = _request(
            self.store, "GET", f"/api/fleet/sessions/{session['id']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(loaded["id"], session["id"])

        status, updated = _request(
            self.store,
            "PATCH",
            f"/api/fleet/sessions/{session['id']}",
            {"status": "done", "cost": 2.5},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["status"], "done")
        self.assertAlmostEqual(updated["cost"], 2.5)

        status, cap = _request(
            self.store,
            "POST",
            "/api/fleet/caps",
            {"project_id": "proj-1", "cap_usd": 10, "reset_period": "daily"},
        )
        self.assertEqual(status, 201)
        self.assertAlmostEqual(cap["cap_usd"], 10)

        status, caps = _request(self.store, "GET", "/api/fleet/caps")
        self.assertEqual(status, 200)
        self.assertEqual(len(caps), 1)

        status, stats = _request(self.store, "GET", "/api/fleet/stats")
        self.assertEqual(status, 200)
        self.assertEqual(stats["total_sessions"], 1)
        self.assertAlmostEqual(stats["total_cost"], 2.5)

        status, _ = _request(
            self.store, "DELETE", f"/api/fleet/sessions/{session['id']}"
        )
        self.assertEqual(status, 204)
        status, listed = _request(self.store, "GET", "/api/fleet/sessions?status=archived")
        self.assertEqual(status, 200)
        self.assertEqual(listed[0]["status"], ARCHIVED_STATUS)

    def test_spawn_rejected_when_cap_exceeded(self) -> None:
        _request(
            self.store,
            "POST",
            "/api/fleet/caps",
            {"project_id": "proj-1", "cap_usd": 0},
        )
        status, payload = _request(
            self.store,
            "POST",
            "/api/fleet/sessions",
            {"project_id": "proj-1", "card_id": self.card["id"]},
        )
        self.assertEqual(status, 409)
        self.assertIn("error", payload)

    def test_missing_routes_and_validation(self) -> None:
        status, payload = _request(self.store, "GET", "/api/fleet/sessions/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)
        status, payload = _request(self.store, "POST", "/api/fleet/sessions", {})
        self.assertEqual(status, 400)
        status, payload = _request(self.store, "GET", "/api/unknown")
        self.assertEqual(status, 404)


class FleetHttpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FleetStore(":memory:")
        self.server = serve_fleet("127.0.0.1", 0, store=self.store)
        self.port = self.server.server_address[1]
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.store.close()

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
        status, cap = self._http(
            "POST", "/api/fleet/caps", {"project_id": "p1", "cap_usd": 5}
        )
        self.assertEqual(status, 201)
        status, listed = self._http("GET", "/api/fleet/caps")
        self.assertEqual(status, 200)
        self.assertEqual(listed[0]["id"], cap["id"])


if __name__ == "__main__":
    unittest.main()
