"""Determine whether an issue meets the `ready-for-dev` readiness criteria.

The criteria are type-specific:

- Bug reports (labeled `bug`): the Steps to Reproduce section must reference at
  least one supported run method (`agent-canvas`, `npm run`, or
  `app.all-hands.dev/canvas`), the Actual Behavior section must embed a
  screenshot or video, and there must be a non-empty Acceptance Criteria
  section with at least one checklist item.

- Enhancements (labeled `enhancement`): the body must contain non-empty
  Desired Behavior and Acceptance Criteria sections, the latter with at least
  one checklist item.

The type normally comes from the `bug`/`enhancement` label. When neither label
is present (common for agent-filed issues that cannot set labels), the type is
inferred from the body structure instead: a `### Steps to Reproduce` +
`### Actual Behavior` combination marks a bug report, a `### Desired Behavior`
section marks a feature request. Only when the body matches neither template is
the issue treated as not-ready.

GitHub issue forms render each field as an `### <Label>` (h3) heading followed
by the field text, with empty optional fields rendered as `_No response_`. The
parser also accepts `## <Label>` headings used by hand-edited and free-form
issues, and checks each criterion against the corresponding section.

Local usage:

    python .github/scripts/check_issue_readiness.py --body-file /tmp/issue.md \
        --labels bug
    python .github/scripts/check_issue_readiness.py --event-path "$GITHUB_EVENT_PATH"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from markdown_sections import find_headings

BUG_LABEL = "bug"
ENHANCEMENT_LABEL = "enhancement"

# Issue-form fields render as h3 headings, while hand-edited and free-form
# issues commonly use h2. Capture the level so nested h3 headings can remain
# part of an h2 section.
HEADING_RE = re.compile(r"(?m)^(?P<level>#{2,3})\s+(?P<title>.+?)\s*$")

READINESS_SECTION_LABELS = {
    "steps to reproduce",
    "reproduction",
    "actual behavior",
    "actual",
    "desired behavior",
    "desired",
    "acceptance criteria",
    "acceptance",
}

# `_No response_` is what GitHub writes for an empty optional form field.
NO_RESPONSE = "_No response_"

# A supported way to run Agent Canvas. The Reproduction section must mention at
# least one. `agent-canvas` covers `npx @openhands/agent-canvas`,
# `agent-canvas --version`, the published binary, etc. `npm run` covers
# `npm run dev` / `npm run dev:minimal`. The hosted canvas URL is matched
# literally.
RUN_METHOD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"agent-canvas", re.IGNORECASE),
    re.compile(r"npm\s+run", re.IGNORECASE),
    re.compile(r"app\.all-hands\.dev/canvas", re.IGNORECASE),
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

# An Acceptance Criteria item is a markdown checklist bullet (`- [ ]` or
# `- [x]`). We require at least one so the section is verifiable.
CHECKLIST_ITEM_RE = re.compile(r"(?m)^\s*[-*]\s*\[[ xX]\]")


@dataclass
class ReadinessResult:
    """Outcome of a readiness check."""

    ready: bool
    reasons: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        self.reasons.append(reason)
        self.ready = False


def visible_text(text: str) -> str:
    """Return field text with HTML comments stripped and emptiness normalized."""
    cleaned = re.sub(r"<!--[\s\S]*?-->", "", text).strip()
    if cleaned == NO_RESPONSE:
        return ""
    return cleaned


def extract_sections(body: str) -> dict[str, str]:
    """Split the body into a {heading: text} map using h2 or h3 boundaries.

    Issue forms render fields as h3 headings. Free-form and hand-edited issues
    may use h2 headings instead, including a mix of both levels.
    """
    matches = find_headings(body, HEADING_RE)
    has_h2 = any(match.group("level") == "##" for match in matches)
    boundaries = [
        match
        for match in matches
        if match.group("level") == "##"
        or not has_h2
        or match.group("title").strip().lower() in READINESS_SECTION_LABELS
    ]

    sections: dict[str, str] = {}
    for index, match in enumerate(boundaries):
        start = match.end()
        end = (
            boundaries[index + 1].start()
            if index + 1 < len(boundaries)
            else len(body)
        )
        sections[match.group("title").strip().lower()] = body[start:end]
    return sections


def find_section(sections: dict[str, str], *labels: str) -> str:
    """Return the first matching section text by case-insensitive label."""
    for label in labels:
        if label in sections:
            return sections[label]
    return ""


def has_screenshot_or_video(text: str) -> bool:
    if MARKDOWN_IMAGE_RE.search(text):
        return True
    if HTML_IMG_RE.search(text):
        return True
    if HTML_VIDEO_RE.search(text):
        return True
    if GITHUB_ATTACHMENT_RE.search(text):
        return True
    if VIDEO_FILE_RE.search(text):
        return True
    if VIDEO_HOST_RE.search(text):
        return True
    return False


def references_run_method(text: str) -> bool:
    return any(pattern.search(text) for pattern in RUN_METHOD_PATTERNS)


def has_checklist_item(text: str) -> bool:
    return bool(CHECKLIST_ITEM_RE.search(text))


def check_bug(sections: dict[str, str]) -> ReadinessResult:
    result = ReadinessResult(ready=True)

    reproduction = visible_text(find_section(sections, "steps to reproduce", "reproduction"))
    if not reproduction:
        result.add(
            "Fill in the `### Steps to Reproduce` section showing how you reproduced "
            "the bug in a live Agent Canvas session."
        )
    elif not references_run_method(reproduction):
        result.add(
            "The Steps to Reproduce section must reference a supported run method: "
            "`agent-canvas`, `npm run`, or `app.all-hands.dev/canvas`."
        )

    actual = visible_text(find_section(sections, "actual behavior", "actual"))
    if not actual:
        result.add("Fill in the `### Actual Behavior` section describing the observed bug.")
    elif not has_screenshot_or_video(actual):
        result.add(
            "The Actual Behavior section must include a screenshot or video of "
            "the bug (drag a file into the field or paste a link)."
        )

    acceptance = visible_text(find_section(sections, "acceptance criteria", "acceptance"))
    if not acceptance:
        result.add("Add an `### Acceptance Criteria` section with testable checklist items.")
    elif not has_checklist_item(acceptance):
        result.add(
            "The Acceptance Criteria section must contain at least one checklist item "
            "(`- [ ] …`)."
        )

    return result


def check_enhancement(sections: dict[str, str]) -> ReadinessResult:
    result = ReadinessResult(ready=True)

    desired = visible_text(find_section(sections, "desired behavior", "desired"))
    if not desired:
        result.add(
            "Add a `### Desired Behavior` section describing the behavior you want."
        )

    acceptance = visible_text(find_section(sections, "acceptance criteria", "acceptance"))
    if not acceptance:
        result.add("Add an `### Acceptance Criteria` section with testable checklist items.")
    elif not has_checklist_item(acceptance):
        result.add(
            "The Acceptance Criteria section must contain at least one checklist item "
            "(`- [ ] …`)."
        )

    return result


def infer_issue_type(sections: dict[str, str]) -> str | None:
    """Infer the issue type from its body structure when no type label exists.

    Bug reports use the `### Steps to Reproduce` + `### Actual Behavior`
    sections; feature requests use `### Desired Behavior`. Agent-filed issues
    may be unable to set labels, so the gate falls back to these markers.
    """
    if "steps to reproduce" in sections and "actual behavior" in sections:
        return BUG_LABEL
    if "desired behavior" in sections:
        return ENHANCEMENT_LABEL
    return None


def evaluate_readiness(body: str, labels: list[str]) -> ReadinessResult:
    """Return the readiness result for an issue body + label set.

    The type is taken from the `bug`/`enhancement` label when present.
    Otherwise it is inferred from the body structure (see `infer_issue_type`),
    so agent-filed issues that cannot set labels are still evaluated. Only when
    neither a type label nor a recognizable template is present is the issue
    treated as not-ready.
    """
    label_set = {label.lower() for label in labels}
    sections = extract_sections(body or "")

    if BUG_LABEL in label_set or ENHANCEMENT_LABEL in label_set:
        if BUG_LABEL in label_set:
            return check_bug(sections)
        return check_enhancement(sections)

    inferred = infer_issue_type(sections)
    if inferred == BUG_LABEL:
        return check_bug(sections)
    if inferred == ENHANCEMENT_LABEL:
        return check_enhancement(sections)

    return ReadinessResult(
        ready=False,
        reasons=[
            "The issue has neither a `bug` nor `enhancement` label, and its "
            "body does not match a bug or feature template, so the readiness "
            "criteria cannot be evaluated."
        ],
    )


def body_and_labels_from_event(event_path: Path) -> tuple[str, list[str]]:
    payload = json.loads(event_path.read_text())
    issue = payload.get("issue") or payload.get("pull_request")
    if not isinstance(issue, dict):
        raise ValueError("GitHub event payload does not contain an issue object")
    body = issue.get("body")
    body = body if isinstance(body, str) else ""
    labels = [label["name"] for label in issue.get("labels", []) if isinstance(label, dict)]
    return body, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether an issue meets the ready-for-dev criteria."
    )
    parser.add_argument("--body-file", type=Path, help="Read the issue body from a file.")
    parser.add_argument(
        "--labels",
        help="Comma-separated issue labels (e.g. 'bug,frontend').",
        default="",
    )
    parser.add_argument(
        "--event-path",
        type=Path,
        default=Path(os.environ["GITHUB_EVENT_PATH"])
        if "GITHUB_EVENT_PATH" in os.environ
        else None,
        help="Read body and labels from a GitHub event payload.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON result instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.body_file is not None:
        body = args.body_file.read_text()
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    elif args.event_path is not None:
        body, labels = body_and_labels_from_event(args.event_path)
    else:
        raise SystemExit("Pass --body-file or set GITHUB_EVENT_PATH.")

    result = evaluate_readiness(body, labels)

    if args.json:
        print(json.dumps({"ready": result.ready, "reasons": result.reasons}))
    else:
        if result.ready:
            print("Issue meets ready-for-dev criteria.")
        else:
            print("Issue does not meet ready-for-dev criteria:")
            for reason in result.reasons:
                print(f"  - {reason}")

    return 0 if result.ready else 1


if __name__ == "__main__":
    sys.exit(main())
