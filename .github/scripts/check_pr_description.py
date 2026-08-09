"""Validate PR description readiness before a PR is reviewed.

Required template fields: Why, Summary, and How to Test.

Additional checks:
- If the "A human has tested these changes" checkbox is present, it must be
  checked.
- If frontend code was touched, the description must include a screenshot or
  video.
- If the PR is marked as a Bug fix, the description must include a screenshot
  or video showing reproduction evidence — regardless of whether frontend
  code was touched.
- The body must reference at least one issue (e.g. `Fixes #123`) and at least
  one referenced issue must carry the `ready-for-dev` label. The API lookup is
  only performed in CI (when GITHUB_EVENT_PATH and GITHUB_TOKEN are available).
- The PR's Type checkbox must match the linked issue's labels: a "Bug fix" PR
  should link an issue with the `bug` label; a "Feature" PR should link one
  with the `enhancement` label. This prevents a contributor from bypassing
  bug-specific requirements by mislabeling the PR type.

Local usage example:
    python .github/scripts/check_pr_description.py --body-file /tmp/pr-body.md \
        --files-file /tmp/pr-files.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# Reject placeholders while allowing a concise human-written sentence.
MIN_HUMAN_NOTE_CHARS = 20
# These are the only PR-template sections that must remain and contain content.
REQUIRED_TEMPLATE_FIELDS: tuple[str, ...] = ("Why", "Summary", "How to Test")

HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
HEADING_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
HUMAN_HEADING_RE = re.compile(r"(?im)^\s*HUMAN:\s*$")
AGENT_HEADING_RE = re.compile(r"(?im)^\s*AGENT:\s*$")

# A file counts as frontend code if its path is under one of these prefixes or
# has one of these extensions. This mirrors the paths the E2E workflows treat as
# stack-affecting (src/**, public/**) plus component/style/test extensions.
FRONTEND_PATH_PREFIXES: tuple[str, ...] = ("src/", "__tests__/", "public/")
FRONTEND_FILE_EXTENSIONS: tuple[str, ...] = (
    ".tsx",
    ".jsx",
    ".vue",
    ".svelte",
    ".css",
    ".scss",
    ".sass",
    ".less",
)
FRONTEND_CONFIG_GLOBS: tuple[str, ...] = (
    "tailwind.config.*",
    "vite.config.*",
    "postcss.config.*",
)

# The human-tested checkbox the HUMAN section asks contributors to tick.
HUMAN_TESTED_RE = re.compile(
    r"(?im)^\s*[-*]\s*\[(?P<box>[ xX])\]\s*.*?human has tested these changes"
)

# Issue references in the PR body: "Fixes #123", "Closes #123", "Resolves #123",
# or a bare "#123" in the Issue Number section. We capture the issue number.
# GitHub auto-linking keywords: close, closes, closed, fix, fixes, fixed,
# resolve, resolves, resolved (case-insensitive).
ISSUE_REF_RE = re.compile(
    r"(?i)(?:fix|clos|resolv)(?:e?(?:s|d)?|ing)?\s+#(\d+)"
)
BARE_ISSUE_REF_RE = re.compile(r"(?<!\w)#(\d+)")

READY_FOR_DEV_LABEL = "ready-for-dev"
BUG_LABEL = "bug"
ENHANCEMENT_LABEL = "enhancement"

# PR Type checkboxes in the `## Type` section. We capture which type the author
# checked so we can cross-check it against the linked issue's labels.
PR_TYPE_BUG_RE = re.compile(
    r"(?im)^\s*[-*]\s*\[(?P<box>[ xX])\]\s*Bug fix"
)
PR_TYPE_FEATURE_RE = re.compile(
    r"(?im)^\s*[-*]\s*\[(?P<box>[ xX])\]\s*Feature"
)

# Markdown image: ![alt](url)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
# HTML <img ...> and <video ...> tags.
HTML_IMG_RE = re.compile(r"<img\b[^>]*\bsrc\s*=", re.IGNORECASE)
HTML_VIDEO_RE = re.compile(r"<video\b", re.IGNORECASE)
# GitHub-uploaded assets (images and videos both use this URL shape).
GITHUB_ATTACHMENT_RE = re.compile(
    r"https?://(?:www\.)?github\.com/user-attachments/assets/"
)
# Direct links to video files.
VIDEO_FILE_RE = re.compile(
    r"https?://\S+\.(?:mp4|webm|mov|avi|mkv|ogv|3gp|m4v)(?:\S*)",
    re.IGNORECASE,
)
# Known video-hosting services.
VIDEO_HOST_RE = re.compile(
    r"https?://(?:[a-z0-9-]+\.)?(?:youtube\.com|youtu\.be|loom\.com|vimeo\.com|asciinema\.org|streamable\.com)/",
    re.IGNORECASE,
)


def visible_text(text: str) -> str:
    """Return PR body content that should count as author-provided text."""
    lines = []
    for line in HTML_COMMENT_RE.sub("", text).splitlines():
        stripped = line.strip()
        if stripped and stripped != "-":
            lines.append(stripped)
    return "\n".join(lines).strip()


def first_visible_line(text: str) -> str:
    for line in HTML_COMMENT_RE.sub("", text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_sections(body: str) -> dict[str, str]:
    matches = list(HEADING_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip()] = body[start:end]
    return sections


def extract_human_note(body: str) -> str:
    """Return human-written text in the required location before `AGENT:`."""
    human_match = HUMAN_HEADING_RE.search(body)
    if human_match is None:
        return ""

    agent_match = AGENT_HEADING_RE.search(body, human_match.end())
    if agent_match is None:
        return ""

    return visible_text(body[human_match.end() : agent_match.start()])


def is_frontend_file(path: str) -> bool:
    """Return True if a changed file should be treated as frontend code."""
    normalized = path.lstrip("./")
    if any(normalized.startswith(prefix) for prefix in FRONTEND_PATH_PREFIXES):
        return True
    lower = normalized.lower()
    if any(lower.endswith(ext) for ext in FRONTEND_FILE_EXTENSIONS):
        return True
    name = normalized.split("/")[-1]
    return any(
        re.fullmatch(glob.replace(r".", r"\.").replace(r"*", r".*"), name)
        for glob in FRONTEND_CONFIG_GLOBS
    )


def touches_frontend(files: list[str]) -> bool:
    """Return True if any changed file is frontend code."""
    return any(is_frontend_file(path) for path in files if path)


def has_screenshot_or_video(body: str) -> bool:
    """Return True if the PR body embeds a screenshot or video."""
    if MARKDOWN_IMAGE_RE.search(body):
        return True
    if HTML_IMG_RE.search(body):
        return True
    if HTML_VIDEO_RE.search(body):
        return True
    if GITHUB_ATTACHMENT_RE.search(body):
        return True
    if VIDEO_FILE_RE.search(body):
        return True
    if VIDEO_HOST_RE.search(body):
        return True
    return False


def validate_human_tested_checkbox(body: str) -> list[str]:
    """Require the human-tested checkbox to be checked when it is present."""
    errors: list[str] = []
    matches = list(HUMAN_TESTED_RE.finditer(body))
    for match in matches:
        if match.group("box").strip().lower() != "x":
            errors.append(
                "The `A human has tested these changes` checkbox is present but "
                "unchecked. Tick it (`- [x]`) or remove the line if it does not apply."
            )
            break
    return errors


def validate_frontend_screenshot(body: str, files: list[str]) -> list[str]:
    """Require a screenshot/video in the body when frontend code was touched."""
    if not touches_frontend(files):
        return []
    if has_screenshot_or_video(body):
        return []
    return [
        "This PR touches frontend code but the description has no screenshot or "
        "video. Add one under `## Video/Screenshots` (drag a file into the editor "
        "or paste a video link)."
    ]


def validate_bug_fix_evidence(body: str) -> list[str]:
    """Require reproduction evidence when the PR is marked as a Bug fix.

    A bug-fix PR must include a screenshot or video showing the bug reproduced
    and then fixed. This applies regardless of whether frontend code was touched
    — a terminal screenshot showing the error before and the fix after is just
    as valid as a UI screenshot.
    """
    pr_type = extract_pr_type(body)
    if pr_type != BUG_LABEL:
        return []
    if has_screenshot_or_video(body):
        return []
    return [
        "This PR is marked as a Bug fix but the description has no screenshot or "
        "video. Add reproduction evidence under `## Video/Screenshots` showing the "
        "bug before the fix and the result after (drag a file into the editor or "
        "paste a video link)."
    ]


def extract_linked_issue_numbers(body: str) -> list[int]:
    """Return issue numbers referenced in the PR body.

    Looks for GitHub auto-close keywords (`Fixes #N`, `Closes #N`,
    `Resolves #N`) anywhere in the body, plus bare `#N` references inside the
    `## Issue Number` section.
    """
    numbers: list[int] = []
    seen: set[int] = set()

    for match in ISSUE_REF_RE.finditer(body):
        number = int(match.group(1))
        if number not in seen:
            numbers.append(number)
            seen.add(number)

    sections = extract_sections(body)
    issue_section = sections.get("Issue Number", "")
    if visible_text(issue_section):
        for match in BARE_ISSUE_REF_RE.finditer(issue_section):
            number = int(match.group(1))
            if number not in seen:
                numbers.append(number)
                seen.add(number)

    return numbers


def extract_pr_type(body: str) -> str | None:
    """Return the checked PR type from the `## Type` section.

    Returns "bug" if the "Bug fix" checkbox is checked, "enhancement" if the
    "Feature" checkbox is checked, or None if neither is checked or the section
    is absent.
    """
    bug_match = PR_TYPE_BUG_RE.search(body)
    if bug_match and bug_match.group("box").strip().lower() == "x":
        return BUG_LABEL
    feature_match = PR_TYPE_FEATURE_RE.search(body)
    if feature_match and feature_match.group("box").strip().lower() == "x":
        return ENHANCEMENT_LABEL
    return None


def fetch_issue_labels(repo: str, issue_number: int, token: str) -> list[str]:
    """Fetch label names for an issue via the GitHub REST API."""
    import urllib.request

    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/labels"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted HTTPS API
        data = json.loads(resp.read().decode())
    return [label["name"] for label in data if isinstance(label, dict)]


def validate_linked_issue_ready(
    body: str, repo: str | None = None, token: str | None = None
) -> list[str]:
    """Require a linked issue carrying the `ready-for-dev` label.

    Also cross-checks the PR's Type checkbox against the linked issue's labels:
    a "Bug fix" PR must link an issue with the `bug` label, and a "Feature" PR
    must link one with the `enhancement` label.

    When `repo` and `token` are not provided (local `--body-file` mode), only
    checks that the body references at least one issue — the API lookup is
    skipped.
    """
    errors: list[str] = []

    numbers = extract_linked_issue_numbers(body)
    if not numbers:
        errors.append(
            "Link an issue in the `## Issue Number` section (e.g. `Fixes #123`). "
            "The issue must carry the `ready-for-dev` label."
        )
        return errors

    if not repo or not token:
        return errors

    import urllib.error

    pr_type = extract_pr_type(body)

    checked: list[int] = []
    found_ready = False
    found_type_match = False

    for number in numbers:
        try:
            labels = fetch_issue_labels(repo, number, token)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        checked.append(number)

        if READY_FOR_DEV_LABEL in labels:
            found_ready = True

        if pr_type and pr_type in [label.lower() for label in labels]:
            found_type_match = True

    if not found_ready:
        if checked:
            ref = ", ".join(f"#{n}" for n in checked)
            errors.append(
                f"None of the linked issues ({ref}) carry the `ready-for-dev` label. "
                "The issue must meet the type-specific readiness criteria before a PR "
                "can be opened against it."
            )
        else:
            errors.append(
                f"Referenced issue(s) {', '.join(f'#{n}' for n in numbers)} could not "
                "be found in this repository. Link an issue in this repo."
            )

    if pr_type and not found_type_match and checked:
        type_name = "bug" if pr_type == BUG_LABEL else "enhancement"
        ref = ", ".join(f"#{n}" for n in checked)
        errors.append(
            f"This PR is marked as `{type_name}` but none of the linked issues "
            f"({ref}) carry the `{pr_type}` label. Ensure the linked issue has the "
            f"correct type label (`{pr_type}`)."
        )

    return errors


def validate_pr_body(body: str, files: list[str] | None = None) -> list[str]:
    errors: list[str] = []

    if first_visible_line(body) != "HUMAN:":
        errors.append("The first visible line of the PR description must be `HUMAN:`.")

    human_note = extract_human_note(body)
    if len(human_note) < MIN_HUMAN_NOTE_CHARS:
        errors.append("Add a short human-written note between `HUMAN:` and `AGENT:`.")

    if AGENT_HEADING_RE.search(body) is None:
        errors.append("Keep the `AGENT:` marker from the PR template.")

    sections = extract_sections(body)
    for section in REQUIRED_TEMPLATE_FIELDS:
        if section not in sections:
            errors.append(f"Keep the `## {section}` section from the PR template.")
        elif not visible_text(sections[section]):
            errors.append(f"Fill in the `## {section}` section of the PR template.")

    errors.extend(validate_human_tested_checkbox(body))
    errors.extend(validate_frontend_screenshot(body, files or []))
    errors.extend(validate_bug_fix_evidence(body))

    return errors


def body_from_event(event_path: Path) -> str:
    payload = json.loads(event_path.read_text())
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        raise ValueError("GitHub event payload does not contain a pull_request object")
    body = pull_request.get("body")
    return body if isinstance(body, str) else ""


def pr_number_from_event(event_path: Path) -> int | None:
    payload = json.loads(event_path.read_text())
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return None
    number = pull_request.get("number")
    return int(number) if number is not None else None


def files_from_api(event_path: Path) -> list[str]:
    """Fetch the changed file paths for the PR via the GitHub REST API.

    Used in CI when --files-file is not provided. Relies on GITHUB_TOKEN and the
    repository being available in the event payload.
    """
    import urllib.request

    payload = json.loads(event_path.read_text())
    repo = payload.get("repository", {}).get("full_name")
    number = pr_number_from_event(event_path)
    if not repo or number is None:
        return []

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return []

    files: list[str] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/pulls/{number}/files?per_page=100&page={page}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted HTTPS API
            data = json.loads(resp.read().decode())
        if not data:
            break
        files.extend(item.get("filename", "") for item in data if isinstance(item, dict))
        if len(data) < 100:
            break
        page += 1
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pull request description readiness from --body-file "
            "or a GitHub event payload."
        )
    )
    parser.add_argument(
        "--body-file", type=Path, help="Read a PR description body from a file."
    )
    parser.add_argument(
        "--event-path",
        type=Path,
        default=Path(os.environ["GITHUB_EVENT_PATH"])
        if "GITHUB_EVENT_PATH" in os.environ
        else None,
        help="Read the PR description body from a GitHub event payload.",
    )
    parser.add_argument(
        "--files-file",
        type=Path,
        help=(
            "Read changed file paths (one per line) from a file. If omitted in CI, "
            "the script fetches them from the GitHub API."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.body_file is not None:
        body = args.body_file.read_text()
        files: list[str] = []
    elif args.event_path is not None:
        body = body_from_event(args.event_path)
    else:
        raise SystemExit("Pass --body-file or set GITHUB_EVENT_PATH.")

    if args.files_file is not None:
        files = [
            line.strip()
            for line in args.files_file.read_text().splitlines()
            if line.strip()
        ]
    elif args.event_path is not None and args.body_file is None:
        files = files_from_api(args.event_path)
    else:
        files = []

    errors = validate_pr_body(body, files)

    repo = None
    token = os.environ.get("GITHUB_TOKEN")
    if args.event_path is not None and args.body_file is None:
        payload = json.loads(args.event_path.read_text())
        repo = payload.get("repository", {}).get("full_name")

    errors.extend(validate_linked_issue_ready(body, repo, token))

    for error in errors:
        print(f"::error::{error}")

    if errors:
        print(f"PR description validation failed with {len(errors)} error(s).")
        return 1

    print("PR description validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
