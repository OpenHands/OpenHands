"""
PR Analyzer — fetches PR metadata, diff, and file list from GitHub.

Supports both GitHub Actions context and local testing via environment variables.
"""

import os
from typing import Any, Optional

import requests


class PRAnalyzer:
    """Analyze a GitHub pull request and extract review inputs."""

    def __init__(self, token: str, repo: str, pr_number: int):
        self.token = token
        self.repo = repo  # "owner/repo"
        self.pr_number = pr_number
        self.api_base = "https://api.github.com"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3.diff",
        })

    def get_pr_metadata(self) -> dict[str, Any]:
        """Fetch PR details from GitHub API."""
        url = f"{self.api_base}/repos/{self.repo}/pulls/{self.pr_number}"
        resp = self.session.get(url, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        data = resp.json()

        return {
            "title": data["title"],
            "body": data.get("body", "") or "",
            "author": data["user"]["login"],
            "base_branch": data["base"]["ref"],
            "head_branch": data["head"]["ref"],
            "base_sha": data["base"]["sha"],
            "head_sha": data["head"]["sha"],
            "changed_files": data["changed_files"],
            "additions": data["additions"],
            "deletions": data["deletions"],
            "draft": data["draft"],
            "labels": [label["name"] for label in data["labels"]],
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }

    def get_diff(self) -> str:
        """Fetch the unified diff for the PR."""
        url = f"{self.api_base}/repos/{self.repo}/pulls/{self.pr_number}"
        # The default Accept header includes diff
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.text

    def get_files(self) -> list[dict[str, Any]]:
        """Fetch the list of changed files with patch."""
        url = f"{self.api_base}/repos/{self.repo}/pulls/{self.pr_number}/files"
        resp = self.session.get(url, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        return resp.json()

    def get_pr_description_sections(self) -> dict[str, str]:
        """Parse PR body into sections based on template markers."""
        body = self.get_pr_metadata()["body"]
        sections = {}

        current_section = None
        current_lines = []

        for line in body.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = stripped[3:].strip()
                current_lines = []
            else:
                if current_section:
                    current_lines.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        return sections

    def check_template_compliance(self) -> dict:
        """Check PR description for required template fields."""
        metadata = self.get_pr_metadata()
        body = metadata["body"]

        required_fields = ["Why", "Summary", "How to Test", "Type"]
        missing = []
        present = []

        for field in required_fields:
            if f"## {field}" in body:
                present.append(field)
            else:
                missing.append(field)

        # Check that Type has at least one checkbox selected
        if "## Type" in body:
            type_section = self._extract_section(body, "Type")
            has_selection = any(
                line.strip().startswith(("- [x]", "- [X]"))
                for line in type_section.split("\n")
            )
            if not has_selection:
                if "Type" not in missing and "Type" not in present:
                    pass  # already handled
                missing.append("Type (no checkbox selected)")
                present.remove("Type")

        return {
            "passed": len(missing) == 0,
            "missing": missing,
            "present": present,
        }

    def _extract_section(self, body: str, section_name: str) -> str:
        """Extract a section from the PR body by its heading."""
        lines = body.split("\n")
        in_section = False
        section_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"## {section_name}"):
                in_section = True
                continue
            if stripped.startswith("## ") and in_section:
                break
            if in_section:
                section_lines.append(line)
        return "\n".join(section_lines).strip()

    @classmethod
    def from_env(cls) -> Optional["PRAnalyzer"]:
        """Create an analyzer from GitHub Actions environment variables."""
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("REVIEWER_GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY")
        pr_number = os.environ.get("PR_NUMBER")

        if not token:
            # Try to read from file (GitHub Actions provides this path)
            token_path = os.environ.get("GITHUB_TOKEN_PATH")
            if token_path and os.path.exists(token_path):
                with open(token_path) as f:
                    token = f.read().strip()

        if not token:
            return None

        if not pr_number and os.environ.get("GITHUB_EVENT_PATH"):
            # Parse PR number from the GitHub event payload
            try:
                import json
                with open(os.environ["GITHUB_EVENT_PATH"]) as f:
                    event = json.load(f)
                pr_number = str(
                    event.get("pull_request", {}).get("number")
                    or event.get("issue", {}).get("number")
                )
            except (json.JSONDecodeError, FileNotFoundError, KeyError):
                pass

        if not token or not repo or not pr_number:
            return None

        try:
            return cls(token=token, repo=repo, pr_number=int(pr_number))
        except (ValueError, requests.RequestException) as e:
            print(f"[Reviewer] Failed to create PRAnalyzer: {e}")
            return None
