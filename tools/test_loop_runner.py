"""Unit and API tests for the loop runner store.

Run from the repo root:

    python3 tools/test_loop_runner.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from loop_runner import (  # noqa: E402
    LoopError,
    LoopStore,
    NotFoundError,
    ON_FAILURE_AUTO_FIX,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_RUNNING,
)
from loop_runner_api import handle_request  # noqa: E402


def _request(
    store: LoopStore,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return handle_request(store, method, path, body)


def _write_scripts(root: str) -> None:
    os.makedirs(os.path.join(root, "bin"), exist_ok=True)
    with open(os.path.join(root, "package.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "scripts": {
                    "lint": "python3 bin/lint.py",
                    "typecheck": "python3 bin/ok.py",
                    "test": "python3 bin/ok.py",
                    "build": "python3 bin/ok.py",
                }
            },
            handle,
        )
    with open(os.path.join(root, "bin", "ok.py"), "w", encoding="utf-8") as handle:
        handle.write("print('ok')\n")
    with open(os.path.join(root, "bin", "lint.py"), "w", encoding="utf-8") as handle:
        handle.write(
            "import pathlib, sys\n"
            "flag = pathlib.Path(__file__).resolve().parent.parent / 'FAIL_LINT'\n"
            "if flag.exists():\n"
            "    print('lint failed', file=sys.stderr)\n"
            "    raise SystemExit(1)\n"
            "print('lint ok')\n"
        )


COMMIT_STAGES = [
    {"name": "lint", "cmd": None, "iterative": True},
    {"name": "typecheck", "cmd": None, "iterative": True},
    {"name": "test", "cmd": None, "iterative": True},
    {"name": "build", "cmd": None, "iterative": False},
]


class LoopStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = LoopStore(":memory:")
        self.workdir = tempfile.mkdtemp()
        _write_scripts(self.workdir)

    def tearDown(self) -> None:
        self.store.close()

    def _definition(self, **overrides: Any) -> dict[str, Any]:
        payload = {
            "name": "commit-loop",
            "project_id": "proj-1",
            "stages": COMMIT_STAGES,
            "max_iterations": 10,
            "max_cost_usd": 5.0,
            "on_failure": ON_FAILURE_AUTO_FIX,
        }
        payload.update(overrides)
        return self.store.create_definition(**payload)

    def test_create_definition_validates_schema(self) -> None:
        definition = self._definition()
        self.assertEqual(definition["name"], "commit-loop")
        self.assertEqual(len(definition["stages"]), 4)
        self.assertEqual(definition["on_failure"], ON_FAILURE_AUTO_FIX)
        with self.assertRaises(LoopError):
            self.store.create_definition(
                name="bad",
                project_id="proj-1",
                stages=[],
            )
        with self.assertRaises(LoopError):
            self.store.create_definition(
                name="bad",
                project_id="proj-1",
                stages=COMMIT_STAGES,
                on_failure="explode",
            )

    def test_start_run_executes_stages_in_order(self) -> None:
        definition = self._definition()
        run = self.store.start_run(
            definition["id"], worktree_dir=self.workdir, session_id="sess-1"
        )
        self.assertEqual(run["status"], STATUS_PASSED)
        self.assertEqual(run["session_id"], "sess-1")
        names = [stage["stage_name"] for stage in run["stages"]]
        self.assertEqual(names, ["lint", "typecheck", "test", "build"])
        self.assertTrue(all(stage["status"] == STATUS_PASSED for stage in run["stages"]))
        self.assertGreaterEqual(run["total_cost_usd"], 0)

    def test_explicit_cmd_overrides_package_scripts(self) -> None:
        definition = self._definition(
            stages=[
                {
                    "name": "echo",
                    "cmd": "python3 -c \"print('hello-stage')\"",
                    "iterative": False,
                }
            ]
        )
        run = self.store.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(run["status"], STATUS_PASSED)
        self.assertIn("hello-stage", run["stages"][0]["last_output"])

    def test_auto_fix_pauses_on_iterative_failure(self) -> None:
        open(os.path.join(self.workdir, "FAIL_LINT"), "w", encoding="utf-8").close()
        definition = self._definition()
        run = self.store.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(run["status"], STATUS_RUNNING)
        self.assertEqual(run["current_stage"], "lint")
        self.assertEqual(run["stages"][0]["status"], STATUS_FAILED)
        self.assertIn("lint failed", run["stages"][0]["last_output"])
        self.assertEqual(len(run["stages"]), 1)

        fix = self.store.request_fix(run["id"])
        self.assertIn("lint failed", fix["last_output"])
        os.remove(os.path.join(self.workdir, "FAIL_LINT"))
        run = self.store.retry_stage(run["id"])
        self.assertEqual(run["status"], STATUS_PASSED)
        self.assertEqual(run["stages"][0]["attempt"], 2)
        self.assertEqual(len(run["stages"]), 4)

    def test_non_iterative_failure_stops_run(self) -> None:
        definition = self._definition(
            stages=[
                {
                    "name": "build",
                    "cmd": "python3 -c \"raise SystemExit(1)\"",
                    "iterative": False,
                }
            ]
        )
        run = self.store.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(run["status"], STATUS_FAILED)
        self.assertEqual(run["stages"][0]["status"], STATUS_FAILED)

    def test_iteration_cap_stops_retries(self) -> None:
        open(os.path.join(self.workdir, "FAIL_LINT"), "w", encoding="utf-8").close()
        definition = self._definition(max_iterations=2)
        run = self.store.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(run["status"], STATUS_RUNNING)
        run = self.store.retry_stage(run["id"])
        self.assertEqual(run["status"], STATUS_FAILED)
        self.assertEqual(run["stages"][0]["attempt"], 2)
        self.assertEqual(run["iteration"], 2)

    def test_cost_guard_stops_run(self) -> None:
        definition = self._definition(max_cost_usd=0.0)
        run = self.store.start_run(definition["id"], worktree_dir=self.workdir)
        self.assertEqual(run["status"], STATUS_FAILED)
        self.assertGreaterEqual(run["total_cost_usd"], 0)

    def test_abort_and_resume_checkpoint_round_trip(self) -> None:
        open(os.path.join(self.workdir, "FAIL_LINT"), "w", encoding="utf-8").close()
        definition = self._definition()
        run = self.store.start_run(definition["id"], worktree_dir=self.workdir)
        aborted = self.store.abort_run(run["id"])
        self.assertEqual(aborted["status"], STATUS_ABORTED)
        checkpoint = self.store.resume_checkpoint(run["id"])
        self.assertEqual(checkpoint["run"]["id"], run["id"])
        self.assertEqual(checkpoint["run"]["status"], STATUS_ABORTED)
        self.assertEqual(checkpoint["definition"]["id"], definition["id"])
        self.assertEqual(checkpoint["stages"][0]["stage_name"], "lint")

        db_path = os.path.join(self.workdir, "loops.sqlite")
        persistent = LoopStore(db_path)
        created = persistent.create_definition(
            name="persist",
            project_id="proj-1",
            stages=[{"name": "ok", "cmd": "python3 -c \"print(1)\"", "iterative": False}],
        )
        persistent.start_run(created["id"], worktree_dir=self.workdir)
        persistent.close()
        reloaded = LoopStore(db_path)
        listed = reloaded.list_runs(created["id"])
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["status"], STATUS_PASSED)
        reloaded.close()

    def test_unknown_run_raises_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.store.get_run("missing")


class LoopApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = LoopStore(":memory:")
        self.workdir = tempfile.mkdtemp()
        _write_scripts(self.workdir)

    def tearDown(self) -> None:
        self.store.close()

    def test_create_list_and_run_via_api(self) -> None:
        status, definition = _request(
            self.store,
            "POST",
            "/api/loops",
            {
                "name": "commit-loop",
                "project_id": "proj-1",
                "stages": COMMIT_STAGES,
                "max_iterations": 5,
                "max_cost_usd": 5.0,
                "on_failure": "auto_fix",
            },
        )
        self.assertEqual(status, 201)
        listed_status, listed = _request(self.store, "GET", "/api/loops")
        self.assertEqual(listed_status, 200)
        self.assertEqual(len(listed), 1)

        status, run = _request(
            self.store,
            "POST",
            f"/api/loops/{definition['id']}/runs",
            {"worktree_dir": self.workdir, "session_id": "sess-1"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(run["status"], STATUS_PASSED)

        status, runs = _request(
            self.store, "GET", f"/api/loops/{definition['id']}/runs"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(runs), 1)

        status, detail = _request(self.store, "GET", f"/api/loops/runs/{run['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(len(detail["stages"]), 4)

    def test_fix_retry_abort_and_checkpoint_routes(self) -> None:
        open(os.path.join(self.workdir, "FAIL_LINT"), "w", encoding="utf-8").close()
        _, definition = _request(
            self.store,
            "POST",
            "/api/loops",
            {
                "name": "commit-loop",
                "project_id": "proj-1",
                "stages": COMMIT_STAGES,
            },
        )
        _, run = _request(
            self.store,
            "POST",
            f"/api/loops/{definition['id']}/runs",
            {"worktree_dir": self.workdir},
        )
        status, fix = _request(
            self.store, "POST", f"/api/loops/runs/{run['id']}/request-fix"
        )
        self.assertEqual(status, 200)
        self.assertIn("lint failed", fix["last_output"])

        os.remove(os.path.join(self.workdir, "FAIL_LINT"))
        status, retried = _request(
            self.store, "POST", f"/api/loops/runs/{run['id']}/retry-stage"
        )
        self.assertEqual(status, 200)
        self.assertEqual(retried["status"], STATUS_PASSED)

        _, failed_def = _request(
            self.store,
            "POST",
            "/api/loops",
            {
                "name": "fail-loop",
                "project_id": "proj-1",
                "stages": COMMIT_STAGES,
            },
        )
        open(os.path.join(self.workdir, "FAIL_LINT"), "w", encoding="utf-8").close()
        _, running = _request(
            self.store,
            "POST",
            f"/api/loops/{failed_def['id']}/runs",
            {"worktree_dir": self.workdir},
        )
        status, aborted = _request(
            self.store, "POST", f"/api/loops/runs/{running['id']}/abort"
        )
        self.assertEqual(status, 200)
        self.assertEqual(aborted["status"], STATUS_ABORTED)
        status, checkpoint = _request(
            self.store,
            "GET",
            f"/api/loops/runs/{running['id']}/resume-checkpoint",
        )
        self.assertEqual(status, 200)
        self.assertEqual(checkpoint["run"]["status"], STATUS_ABORTED)


if __name__ == "__main__":
    unittest.main()
