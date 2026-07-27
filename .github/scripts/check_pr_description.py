"""Validate PR description readiness before a PR is reviewed.

Required template fields: Why, Summary, and How to Test.

Local usage example:
    python .github/scripts/check_pr_description.py --body-file /tmp/pr-body.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


HUMAN_TESTED_TEXT = "A human has tested these changes."
# Reject placeholders while allowing a concise human-written sentence.
MIN_HUMAN_NOTE_CHARS = 20
# These are the only PR-template sections that must remain and contain content.
REQUIRED_TEMPLATE_FIELDS: tuple[str, ...] = ("Why", "Summary", "How to Test")

HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
HEADING_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
HUMAN_HEADING_RE = re.compile(r"(?im)^\s*HUMAN:\s*$")
AGENT_HEADING_RE = re.compile(r"(?im)^\s*AGENT:\s*$")


def checkbox_re(label: str) -> re.Pattern[str]:
    return re.compile(rf"(?im)^\s*[-*]\s+\[(?P<mark>[ xX])]\s+{re.escape(label)}\s*$")


def checkbox_is_checked(pattern: re.Pattern[str], text: str) -> bool:
    match = pattern.search(text)
    return match is not None and match.group("mark").lower() == "x"


HUMAN_TESTED_RE = checkbox_re(HUMAN_TESTED_TEXT)


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
    """Return human-written text in the required location before the checkbox."""
    human_match = HUMAN_HEADING_RE.search(body)
    if human_match is None:
        return ""

    checkbox_match = HUMAN_TESTED_RE.search(body, human_match.end())
    if checkbox_match is None:
        return ""

    return visible_text(body[human_match.end() : checkbox_match.start()])


def validate_pr_body(body: str) -> list[str]:
    errors: list[str] = []

    if first_visible_line(body) != "HUMAN:":
        errors.append("The first visible line of the PR description must be `HUMAN:`.")

    human_note = extract_human_note(body)
    if len(human_note) < MIN_HUMAN_NOTE_CHARS:
        errors.append(
            "Add a short human-written note between `HUMAN:` and "
            "the human-tested checkbox."
        )

    human_tested = HUMAN_TESTED_RE.search(body)
    if human_tested is None:
        errors.append(
            f"Keep the `- [ ] {HUMAN_TESTED_TEXT}` checkbox in the PR description."
        )
    elif not checkbox_is_checked(HUMAN_TESTED_RE, body):
        errors.append(
            "A human must check `A human has tested these changes.` before review."
        )

    if AGENT_HEADING_RE.search(body) is None:
        errors.append("Keep the `AGENT:` marker from the PR template.")

    sections = extract_sections(body)
    for section in REQUIRED_TEMPLATE_FIELDS:
        if section not in sections:
            errors.append(f"Keep the `## {section}` section from the PR template.")
        elif not visible_text(sections[section]):
            errors.append(f"Fill in the `## {section}` section of the PR template.")

    return errors


def body_from_event(event_path: Path) -> str:
    payload = json.loads(event_path.read_text())
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        raise ValueError("GitHub event payload does not contain a pull_request object")
    body = pull_request.get("body")
    return body if isinstance(body, str) else ""


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.body_file is not None:
        body = args.body_file.read_text()
    elif args.event_path is not None:
        body = body_from_event(args.event_path)
    else:
        raise SystemExit("Pass --body-file or set GITHUB_EVENT_PATH.")

    errors = validate_pr_body(body)
    for error in errors:
        print(f"::error::{error}")

    if errors:
        print(f"PR description validation failed with {len(errors)} error(s).")
        return 1

    print("PR description validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
