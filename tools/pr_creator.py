"""Create git branches, commits, and pull requests from agent sessions."""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kanban import KanbanStore

AGENT_BRANCH_PREFIX = "agent"
DEFAULT_BASE_BRANCH = "main"
DEFAULT_GITHUB_API_URL = "https://api.github.com"
BRANCH_PANEL_STATUSES = ("working", "reviewing", "ci", "merged")
CONVENTIONAL_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
)
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_GITHUB_SSH_RE = re.compile(r"git@github\.com:([^/]+)/(.+?)(?:\.git)?$")
_GITHUB_HTTP_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([^/]+)/(.+?)(?:\.git)?$"
)


class PrCreatorError(Exception):
    """Raised when a pull-request or worktree-read step fails."""


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", (text or "").strip()).strip("-.").lower()
    return slug or "change"


def agent_branch_name(session_id: str, short_description: str) -> str:
    session = (session_id or "").strip()
    if not session:
        raise PrCreatorError("session_id is required")
    return f"{AGENT_BRANCH_PREFIX}/{session}/{slugify(short_description)}"


def conventional_commit_message(
    commit_type: str,
    description: str,
    scope: str | None = None,
) -> str:
    kind = (commit_type or "").strip().lower()
    if kind not in CONVENTIONAL_TYPES:
        raise PrCreatorError(f"Unknown conventional commit type: {commit_type}")
    summary = (description or "").strip()
    if not summary:
        raise PrCreatorError("commit description is required")
    if scope and scope.strip():
        return f"{kind}({scope.strip()}): {summary}"
    return f"{kind}: {summary}"


def generate_pr_description(
    session_summary: str,
    test_results: str | None = None,
    visual_qa: str | None = None,
) -> str:
    sections = [
        "## Summary",
        (session_summary or "").strip() or "Agent session completed.",
        "",
        "## Test results",
        (test_results or "").strip() or "Not reported",
        "",
        "## Visual QA",
        (visual_qa or "").strip() or "Not reported",
    ]
    return "\n".join(sections).strip() + "\n"


def read_worktree_file(worktree_root: str, relative_path: str) -> str:
    if not relative_path or relative_path.startswith("/") or os.path.isabs(relative_path):
        raise PrCreatorError("path must be relative to the worktree")
    root = os.path.realpath(worktree_root)
    target = os.path.realpath(os.path.join(root, relative_path))
    if target != root and not target.startswith(root + os.sep):
        raise PrCreatorError("path escapes the worktree")
    if not os.path.isfile(target):
        raise PrCreatorError(f"file not found: {relative_path}")
    with open(target, encoding="utf-8") as handle:
        return handle.read()


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
        raise PrCreatorError(detail or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _has_changes(repo_path: str) -> bool:
    status = _run_git(repo_path, "status", "--porcelain")
    return bool(status)


def parse_github_owner_repo(remote_url: str) -> tuple[str, str]:
    url = (remote_url or "").strip()
    ssh = _GITHUB_SSH_RE.match(url)
    if ssh:
        return ssh.group(1), ssh.group(2)
    http = _GITHUB_HTTP_RE.match(url)
    if http:
        return http.group(1), http.group(2)
    raise PrCreatorError("Could not parse GitHub owner/repo from remote URL")


def _post_pull_request(
    *,
    github_api_url: str,
    owner: str,
    repo: str,
    token: str | None,
    title: str,
    body: str,
    head: str,
    base: str,
) -> dict[str, Any]:
    url = f"{github_api_url.rstrip('/')}/repos/{owner}/{repo}/pulls"
    payload = json.dumps(
        {"title": title, "body": body, "head": head, "base": base}
    ).encode("utf-8")
    headers = {
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PrCreatorError(f"GitHub PR create failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise PrCreatorError(f"GitHub PR create failed: {exc.reason}") from exc
    data = json.loads(raw) if raw else {}
    return data


def create_pull_request_from_session(
    repo_path: str,
    *,
    session_id: str,
    short_description: str,
    session_summary: str,
    commit_type: str = "feat",
    test_results: str | None = None,
    visual_qa: str | None = None,
    base_branch: str = DEFAULT_BASE_BRANCH,
    remote: str = "origin",
    github_api_url: str = DEFAULT_GITHUB_API_URL,
    github_token: str | None = None,
    owner: str | None = None,
    repo: str | None = None,
    kanban_store: KanbanStore | None = None,
    card_id: str | None = None,
) -> dict[str, Any]:
    if not _has_changes(repo_path):
        raise PrCreatorError("No changes to commit")
    branch = agent_branch_name(session_id, short_description)
    message = conventional_commit_message(commit_type, short_description)
    body = generate_pr_description(session_summary, test_results, visual_qa)
    _run_git(repo_path, "checkout", "-b", branch, base_branch)
    _run_git(repo_path, "add", "-A")
    _run_git(repo_path, "commit", "-m", message)
    _run_git(repo_path, "push", "-u", remote, branch)
    if not owner or not repo:
        owner, repo = parse_github_owner_repo(_run_git(repo_path, "remote", "get-url", remote))
    pr = _post_pull_request(
        github_api_url=github_api_url,
        owner=owner,
        repo=repo,
        token=github_token,
        title=message,
        body=body,
        head=branch,
        base=base_branch,
    )
    pr_url = str(pr.get("html_url") or "")
    pr_number = pr.get("number")
    if kanban_store is not None and card_id:
        kanban_store.update_card(card_id, linked_pr=pr_url, linked_branch=branch)
    return {
        "branch": branch,
        "commit_message": message,
        "pr_url": pr_url,
        "pr_number": pr_number,
        "description": body,
    }
