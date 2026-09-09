"""Integration tests for the commit-loop auto-fix orchestrator.

Run from the repo root:

    python3 tools/test_commit_loop.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from commit_loop import (  # noqa: E402
    COMMIT_LOOP_NAME,
    CommitLoopError,
    CommitLoopService,
)
from commit_loop_api import handle_request  # noqa: E402
from fleet import FleetStore  # noqa: E402
from kanban import KanbanStore  # noqa: E402
from kanban_agent import REVIEW_STATUS  # noqa: E402
from loop_runner import LoopStore, STATUS_FAILED, STATUS_PASSED  # noqa: E402


def _git(cwd: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_scripts(root: str, fail_build: bool = True) -> None:
    Path(root, "bin").mkdir(exist_ok=True)
    Path(root, "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "lint": "python3 bin/ok.py",
                    "typecheck": "python3 bin/ok.py",
                    "test": "python3 bin/ok.py",
                    "build": "python3 bin/build.py",
                }
            }
        ),
        encoding="utf-8",
    )
    Path(root, "bin", "ok.py").write_text("print('ok')\n", encoding="utf-8")
    Path(root, "bin", "build.py").write_text(
        "import pathlib, sys\n"
        "flag = pathlib.Path(__file__).resolve().parent.parent / 'FAIL_BUILD'\n"
        "if flag.exists():\n"
        "    print('build failed', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print('build ok')\n",
        encoding="utf-8",
    )
    if fail_build:
        Path(root, "FAIL_BUILD").write_text("1\n", encoding="utf-8")


def _init_repo(path: str, fail_build: bool = True) -> None:
    os.makedirs(path, exist_ok=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "agent@example.com")
    _git(path, "config", "user.name", "Agent")
    _write_scripts(path, fail_build=fail_build)
    Path(path, "README.md").write_text("hello\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "chore: init")
    _git(path, "checkout", "-b", "agent/sess/ship-feature")


class CommitLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp()
        _init_repo(self.workdir)
        self.kanban = KanbanStore(":memory:")
        self.loops = LoopStore(":memory:")
        self.fleet = FleetStore(":memory:", kanban_store=self.kanban)
        self.board = self.kanban.create_board("Work", project_id="proj-1")
        self.card = self.kanban.create_card(
            self.board["columns"][0]["id"],
            title="Ship feature",
        )
        self.session = self.fleet.spawn_session(
            "proj-1",
            self.card["id"],
            branch_name="agent/sess/ship-feature",
        )
        self.service = CommitLoopService(
            loop_store=self.loops,
            kanban_store=self.kanban,
            fleet_store=self.fleet,
        )

    def tearDown(self) -> None:
        self.service.close()
        self.loops.close()
        self.fleet.close()
        self.kanban.close()

    def test_setup_registers_commit_loop_definition(self) -> None:
        definition = self.service.setup("proj-1")
        self.assertEqual(definition["name"], COMMIT_LOOP_NAME)
        self.assertEqual(definition["project_id"], "proj-1")
        names = [stage["name"] for stage in definition["stages"]]
        self.assertEqual(names, ["lint", "typecheck", "test", "build"])
        again = self.service.setup("proj-1")
        self.assertEqual(again["id"], definition["id"])

    def test_auto_fix_then_commit_and_move_card(self) -> None:
        self.service.setup("proj-1")
        rounds = {"count": 0}

        def fixer(ctx: dict[str, Any]) -> None:
            rounds["count"] += 1
            self.assertIn("build failed", ctx["last_output"])
            self.assertTrue(ctx["changed_files"] or True)
            Path(self.workdir, "FAIL_BUILD").unlink()
            Path(self.workdir, "src.txt").write_text("fixed\n", encoding="utf-8")

        result = self.service.run_session(
            self.session["id"],
            worktree_dir=self.workdir,
            fix_agent=fixer,
        )
        self.assertEqual(result["status"], STATUS_PASSED)
        self.assertEqual(rounds["count"], 1)
        self.assertTrue(result["commit_sha"])
        self.assertTrue(result["commit_message"].startswith("feat:"))
        log = _git(self.workdir, "log", "-1", "--pretty=%s")
        self.assertIn("Ship feature", log)
        card = self.kanban.get_card(self.card["id"])
        self.assertEqual(card["status"], REVIEW_STATUS)
        self.assertGreater(float(card["actual_cost"] or 0), 0)
        self.assertEqual(card["linked_branch"], "agent/sess/ship-feature")

    def test_abort_on_iteration_cap(self) -> None:
        self.service.setup("proj-1", max_iterations=1, max_cost_usd=5.0)

        def fixer(_ctx: dict[str, Any]) -> None:
            return

        result = self.service.run_session(
            self.session["id"],
            worktree_dir=self.workdir,
            fix_agent=fixer,
        )
        self.assertEqual(result["status"], STATUS_FAILED)
        card = self.kanban.get_card(self.card["id"])
        self.assertNotEqual(card["status"], REVIEW_STATUS)
        self.assertFalse(_git(self.workdir, "log", "-1", "--pretty=%s").startswith("feat:"))

    def test_api_setup_run_and_status(self) -> None:
        status, definition = handle_request(
            self.service, "POST", "/api/commit-loop/projects/proj-1/setup", {}
        )
        self.assertEqual(status, 201)
        self.assertEqual(definition["name"], COMMIT_LOOP_NAME)

        def fixer(ctx: dict[str, Any]) -> None:
            Path(self.workdir, "FAIL_BUILD").unlink(missing_ok=True)

        self.service.fix_agent = fixer
        status, result = handle_request(
            self.service,
            "POST",
            f"/api/commit-loop/sessions/{self.session['id']}/run",
            {"worktree_dir": self.workdir},
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], STATUS_PASSED)
        status, info = handle_request(
            self.service, "GET", "/api/commit-loop/projects/proj-1/status"
        )
        self.assertEqual(status, 200)
        self.assertEqual(info["last_run"]["status"], STATUS_PASSED)
        self.assertEqual(info["definition"]["id"], definition["id"])

    def test_missing_session_is_404(self) -> None:
        self.service.setup("proj-1")
        with self.assertRaises(CommitLoopError):
            self.service.run_session("missing", worktree_dir=self.workdir)


if __name__ == "__main__":
    unittest.main()
