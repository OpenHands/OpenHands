"""Tests for agent-session PR creation and cross-worktree reads.

Run from the repo root:

    python3 tools/test_pr_creator.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kanban import KanbanStore  # noqa: E402
from pr_creator import (  # noqa: E402
    AGENT_BRANCH_PREFIX,
    BRANCH_PANEL_STATUSES,
    PrCreatorError,
    agent_branch_name,
    conventional_commit_message,
    create_pull_request_from_session,
    generate_pr_description,
    read_worktree_file,
)


def _run_git(cwd: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(path: str) -> None:
    _run_git(path, "init", "-b", "main")
    _run_git(path, "config", "user.email", "agent@example.com")
    _run_git(path, "config", "user.name", "Agent")
    Path(path, "README.md").write_text("hello\n", encoding="utf-8")
    _run_git(path, "add", "README.md")
    _run_git(path, "commit", "-m", "chore: init")


class _GitHubHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode("utf-8"))
        parsed = urlparse(self.path)
        self.server.requests.append(  # type: ignore[attr-defined]
            {"path": parsed.path, "body": body, "auth": self.headers.get("Authorization")}
        )
        payload = {
            "html_url": "https://github.com/acme/demo/pull/7",
            "number": 7,
            "title": body.get("title", ""),
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class _GitHubServer(ThreadingHTTPServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[dict[str, Any]] = []


class NamingTests(unittest.TestCase):
    def test_agent_branch_name(self) -> None:
        name = agent_branch_name("sess-9", "Add Login Form!")
        self.assertEqual(name, f"{AGENT_BRANCH_PREFIX}/sess-9/add-login-form")

    def test_agent_branch_rejects_empty_session(self) -> None:
        with self.assertRaises(PrCreatorError):
            agent_branch_name("  ", "login")

    def test_conventional_commit_message(self) -> None:
        message = conventional_commit_message("feat", "add login", scope="auth")
        self.assertEqual(message, "feat(auth): add login")

    def test_conventional_commit_rejects_unknown_type(self) -> None:
        with self.assertRaises(PrCreatorError):
            conventional_commit_message("ship", "stuff")

    def test_pr_description_includes_session_and_qa(self) -> None:
        body = generate_pr_description(
            session_summary="Added login.",
            test_results="npm test passed",
            visual_qa="screenshot: login.png",
        )
        self.assertIn("Added login.", body)
        self.assertIn("npm test passed", body)
        self.assertIn("screenshot: login.png", body)

    def test_branch_panel_statuses(self) -> None:
        self.assertEqual(
            BRANCH_PANEL_STATUSES,
            ("working", "reviewing", "ci", "merged"),
        )


class CrossWorktreeReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.other = os.path.join(self.tmpdir.name, "other")
        os.makedirs(self.other)
        Path(self.other, "src").mkdir()
        Path(self.other, "src", "app.py").write_text("print('ok')\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_reads_relative_file(self) -> None:
        self.assertEqual(
            read_worktree_file(self.other, "src/app.py"),
            "print('ok')\n",
        )

    def test_rejects_path_escape(self) -> None:
        with self.assertRaises(PrCreatorError):
            read_worktree_file(self.other, "../secret")

    def test_rejects_absolute_path(self) -> None:
        with self.assertRaises(PrCreatorError):
            read_worktree_file(self.other, "/etc/passwd")


class PullRequestIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.repo = os.path.join(self.tmpdir.name, "repo")
        self.bare = os.path.join(self.tmpdir.name, "remote.git")
        os.makedirs(self.repo)
        _init_repo(self.repo)
        _run_git(self.tmpdir.name, "init", "--bare", self.bare)
        _run_git(self.repo, "remote", "add", "origin", self.bare)
        Path(self.repo, "login.py").write_text("def login():\n    return True\n", encoding="utf-8")
        self.store = KanbanStore(":memory:")
        board = self.store.create_board("Work")
        self.card = self.store.create_card(board["columns"][0]["id"], title="Add login")
        self.server = _GitHubServer(("127.0.0.1", 0), _GitHubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.api_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.store.close()
        self.tmpdir.cleanup()

    def test_creates_branch_commit_push_and_pr(self) -> None:
        result = create_pull_request_from_session(
            self.repo,
            session_id="abc123",
            short_description="add login",
            session_summary="Implemented login helper.",
            test_results="python3 -m unittest: ok",
            visual_qa="No UI changes",
            commit_type="feat",
            github_api_url=self.api_url,
            github_token="test-token",
            owner="acme",
            repo="demo",
            kanban_store=self.store,
            card_id=self.card["id"],
        )
        self.assertEqual(result["branch"], "agent/abc123/add-login")
        self.assertEqual(result["pr_url"], "https://github.com/acme/demo/pull/7")
        self.assertEqual(result["pr_number"], 7)
        self.assertEqual(
            _run_git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "agent/abc123/add-login",
        )
        log = _run_git(self.repo, "log", "-1", "--pretty=%s")
        self.assertEqual(log, "feat: add login")
        remote_branch = _run_git(self.bare, "rev-parse", "--verify", "refs/heads/agent/abc123/add-login")
        self.assertTrue(remote_branch)
        self.assertEqual(len(self.server.requests), 1)
        posted = self.server.requests[0]
        self.assertEqual(posted["path"], "/repos/acme/demo/pulls")
        self.assertEqual(posted["auth"], "Bearer test-token")
        self.assertEqual(posted["body"]["head"], "agent/abc123/add-login")
        self.assertEqual(posted["body"]["base"], "main")
        self.assertIn("Implemented login helper.", posted["body"]["body"])
        self.assertIn("python3 -m unittest: ok", posted["body"]["body"])
        self.assertIn("No UI changes", posted["body"]["body"])
        card = self.store.get_card(self.card["id"])
        self.assertEqual(card["linked_pr"], result["pr_url"])
        self.assertEqual(card["linked_branch"], result["branch"])

    def test_requires_changes(self) -> None:
        clean = os.path.join(self.tmpdir.name, "clean")
        os.makedirs(clean)
        _init_repo(clean)
        with self.assertRaises(PrCreatorError):
            create_pull_request_from_session(
                clean,
                session_id="abc123",
                short_description="noop",
                session_summary="Nothing.",
            )


if __name__ == "__main__":
    unittest.main()
