"""Unit and API tests for loop triggers.

Run from the repo root:

    python3 tools/test_loop_triggers.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from loop_runner import LoopStore, STATUS_RUNNING  # noqa: E402
from loop_triggers import (  # noqa: E402
    TRIGGER_ON_COMMIT,
    TRIGGER_ON_PR,
    TRIGGER_MANUAL,
    TRIGGER_SCHEDULED,
    EVENT_ERROR,
    EVENT_FIRED,
    EVENT_SKIPPED,
    LoopTriggerService,
    TriggerError,
    notify_commit,
    notify_pr,
    set_active_service,
)
from loop_triggers_api import handle_request  # noqa: E402


def _now() -> datetime:
    return datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)


class LoopTriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp()
        self.loops = LoopStore(":memory:")
        self.service = LoopTriggerService(
            db_path=":memory:", loop_store=self.loops
        )
        set_active_service(self.service)
        self.definition = self.loops.create_definition(
            name="echo",
            project_id="proj-1",
            stages=[
                {
                    "name": "echo",
                    "cmd": "python3 -c \"print('ok')\"",
                    "iterative": False,
                }
            ],
        )

    def tearDown(self) -> None:
        set_active_service(None)
        self.service.close()
        self.loops.close()

    def _scheduled(self, **overrides: Any) -> dict[str, Any]:
        payload = {
            "project_id": "proj-1",
            "loop_definition_id": self.definition["id"],
            "trigger_type": TRIGGER_SCHEDULED,
            "schedule_type": "interval",
            "interval_seconds": 1,
        }
        payload.update(overrides)
        return self.service.create_trigger(**payload)

    def test_interval_trigger_fires_when_due(self) -> None:
        trigger = self._scheduled()
        created = datetime.fromisoformat(trigger["created_at"])
        fired = self.service.tick(
            now=created,
            worktree_dir=self.workdir,
        )
        self.assertEqual(fired, [])
        later = created + timedelta(seconds=1)
        fired = self.service.tick(now=later, worktree_dir=self.workdir)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0]["status"], EVENT_FIRED)
        self.assertIsNotNone(fired[0]["loop_run_id"])
        updated = self.service.get_trigger(trigger["id"])
        self.assertIsNotNone(updated["last_fired_at"])

    def test_disabled_trigger_does_not_fire(self) -> None:
        trigger = self._scheduled(enabled=False)
        created = datetime.fromisoformat(trigger["created_at"])
        fired = self.service.tick(
            now=created + timedelta(seconds=5),
            worktree_dir=self.workdir,
        )
        self.assertEqual(fired, [])
        events = self.service.list_events(trigger_id=trigger["id"])
        self.assertEqual(events, [])

    def test_cron_expression_parses(self) -> None:
        trigger = self._scheduled(
            schedule_type="cron",
            cron_expr="*/5 * * * *",
            interval_seconds=None,
        )
        self.assertEqual(trigger["cron_expr"], "*/5 * * * *")
        with self.assertRaises(TriggerError):
            self._scheduled(schedule_type="cron", cron_expr="not a cron")
        with self.assertRaises(TriggerError):
            self.service.create_trigger(
                project_id="proj-1",
                loop_definition_id=self.definition["id"],
                trigger_type=TRIGGER_SCHEDULED,
            )

    def test_manual_fire_ignores_enabled(self) -> None:
        trigger = self._scheduled(enabled=False)
        result = self.service.fire_trigger(
            trigger["id"],
            {"worktree_dir": self.workdir},
            ignore_enabled=True,
        )
        self.assertEqual(result["event"]["status"], EVENT_FIRED)
        self.assertEqual(result["run"]["status"], "passed")

    def test_on_commit_and_on_pr_helpers(self) -> None:
        commit_trigger = self.service.create_trigger(
            project_id="proj-1",
            loop_definition_id=self.definition["id"],
            trigger_type=TRIGGER_ON_COMMIT,
            payload={"branch": "main"},
        )
        other_branch = self.service.create_trigger(
            project_id="proj-1",
            loop_definition_id=self.definition["id"],
            trigger_type=TRIGGER_ON_COMMIT,
            payload={"branch": "other"},
        )
        pr_trigger = self.service.create_trigger(
            project_id="proj-1",
            loop_definition_id=self.definition["id"],
            trigger_type=TRIGGER_ON_PR,
        )
        commit_events = notify_commit(
            "proj-1",
            "main",
            "abc123",
            worktree_dir=self.workdir,
        )
        self.assertEqual(len(commit_events), 1)
        self.assertEqual(commit_events[0]["trigger_id"], commit_trigger["id"])
        self.assertEqual(commit_events[0]["status"], EVENT_FIRED)
        skipped = notify_commit(
            "proj-1",
            "feature",
            "def456",
            worktree_dir=self.workdir,
        )
        self.assertEqual(skipped, [])
        self.assertEqual(
            self.service.list_events(trigger_id=other_branch["id"]), []
        )
        pr_events = notify_pr(
            "proj-1",
            "main",
            "https://example.com/pr/1",
            worktree_dir=self.workdir,
        )
        self.assertEqual(len(pr_events), 1)
        self.assertEqual(pr_events[0]["trigger_id"], pr_trigger["id"])

    def test_no_double_fire_while_run_active(self) -> None:
        failing = self.loops.create_definition(
            name="fail",
            project_id="proj-1",
            stages=[
                {
                    "name": "lint",
                    "cmd": "python3 -c \"raise SystemExit(1)\"",
                    "iterative": True,
                }
            ],
        )
        trigger = self.service.create_trigger(
            project_id="proj-1",
            loop_definition_id=failing["id"],
            trigger_type=TRIGGER_MANUAL,
        )
        first = self.service.fire_trigger(
            trigger["id"],
            {"worktree_dir": self.workdir},
            ignore_enabled=True,
        )
        self.assertEqual(first["run"]["status"], STATUS_RUNNING)
        with self.assertRaises(TriggerError) as raised:
            self.service.fire_trigger(
                trigger["id"],
                {"worktree_dir": self.workdir},
                ignore_enabled=True,
            )
        self.assertEqual(raised.exception.status, 409)
        events = self.service.list_events(trigger_id=trigger["id"])
        self.assertEqual(events[0]["status"], EVENT_SKIPPED)
        self.assertEqual(len(events), 2)

    def test_event_history_round_trip(self) -> None:
        trigger = self.service.create_trigger(
            project_id="proj-1",
            loop_definition_id=self.definition["id"],
            trigger_type=TRIGGER_MANUAL,
        )
        self.service.fire_trigger(
            trigger["id"],
            {"worktree_dir": self.workdir},
            ignore_enabled=True,
        )
        self.service.fire_trigger(
            trigger["id"],
            {"worktree_dir": self.workdir},
            ignore_enabled=True,
        )
        events = self.service.list_events(trigger_id=trigger["id"])
        self.assertEqual(len(events), 2)
        self.assertGreaterEqual(events[0]["fired_at"], events[1]["fired_at"])
        page = self.service.list_events(trigger_id=trigger["id"], limit=1)
        self.assertEqual(len(page), 1)
        self.assertEqual(page[0]["id"], events[0]["id"])
        all_events = self.service.list_events()
        self.assertEqual(len(all_events), 2)


class LoopTriggerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp()
        self.loops = LoopStore(":memory:")
        self.service = LoopTriggerService(
            db_path=":memory:", loop_store=self.loops
        )
        self.definition = self.loops.create_definition(
            name="echo",
            project_id="proj-1",
            stages=[
                {
                    "name": "echo",
                    "cmd": "python3 -c \"print('ok')\"",
                    "iterative": False,
                }
            ],
        )

    def tearDown(self) -> None:
        self.service.close()
        self.loops.close()

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int, Any]:
        return handle_request(self.service, method, path, body)

    def test_crud_and_fire_routes(self) -> None:
        status, created = self._request(
            "POST",
            "/api/loops/triggers",
            {
                "project_id": "proj-1",
                "loop_definition_id": self.definition["id"],
                "trigger_type": TRIGGER_SCHEDULED,
                "schedule_type": "interval",
                "interval_seconds": 30,
            },
        )
        self.assertEqual(status, 201)
        status, missing_def = self._request(
            "POST",
            "/api/loops/triggers",
            {
                "project_id": "proj-1",
                "loop_definition_id": "missing",
                "trigger_type": TRIGGER_MANUAL,
            },
        )
        self.assertEqual(status, 404)
        self.assertIn("error", missing_def)

        status, listed = self._request(
            "GET", "/api/loops/triggers?project_id=proj-1&enabled=true"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

        status, updated = self._request(
            "PATCH",
            f"/api/loops/triggers/{created['id']}",
            {"enabled": False, "interval_seconds": 60},
        )
        self.assertEqual(status, 200)
        self.assertFalse(updated["enabled"])
        self.assertEqual(updated["interval_seconds"], 60)

        status, fired = self._request(
            "POST",
            f"/api/loops/triggers/{created['id']}/fire",
            {"worktree_dir": self.workdir},
        )
        self.assertEqual(status, 201)
        self.assertEqual(fired["event"]["status"], EVENT_FIRED)

        status, history = self._request(
            "GET", f"/api/loops/triggers/{created['id']}/events?limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(history), 1)

        status, feed = self._request("GET", "/api/loops/triggers/events")
        self.assertEqual(status, 200)
        self.assertEqual(len(feed), 1)

        status, _ = self._request(
            "DELETE", f"/api/loops/triggers/{created['id']}"
        )
        self.assertEqual(status, 204)
        status, listed = self._request("GET", "/api/loops/triggers")
        self.assertEqual(status, 200)
        self.assertEqual(listed, [])


if __name__ == "__main__":
    unittest.main()
