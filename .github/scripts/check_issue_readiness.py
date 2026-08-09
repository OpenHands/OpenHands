"""Determine whether an issue meets the `ready-for-dev` readiness criteria.

The criteria are type-specific:

- Bug reports (labeled `bug`): the Actual Behavior section must reference at
  least one supported run method (`agent-canvas`, `npm run`, or
  `app.all-hands.dev/canvas`) *and* embed a screenshot or video, plus a
  non-empty Acceptance Criteria section with at least one checklist item.

- Enhancements (labeled `enhancement`): the body must contain non-empty
  Desired Behavior and Acceptance Criteria sections, the latter with at least
  one checklist item.

GitHub issue forms render each field as an `### <Label>` (h3) heading followed
by the field text, with empty optional fields rendered as `_No response_`. This
parser splits the body on those headings so each criterion is checked against
the right field rather than the whole body.

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

BUG_LABEL = "bug"
ENHANCEMENT_LABEL = "enhancement"

# Issue-form fields render as `### Label` h3 headings. Match case-insensitively
# and tolerate trailing whitespace/colons. `^###\s+` is specific enough because
# h1/h2 are not produced by issue forms.
HEADING_RE = re.compile(r"(?m)^###\s+(.+?)\s*$")

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
    """Split the body into a {heading: text} map using `### <heading>` boundaries.

    Issue forms render every field this way. Free-form issues (not created via a
    form) may still use `###` headings; if they don't, the map is empty and the
    caller falls back to whole-body checks.
    """
    matches = list(HEADING_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip().lower()] = body[start:end]
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

    actual = visible_text(find_section(sections, "actual behavior", "actual"))
    if not actual:
        result.add(
            "Fill in the `### Actual Behavior` section showing how you reproduced "
            "the bug in a live Agent Canvas session."
        )
    else:
        if not references_run_method(actual):
            result.add(
                "The Actual Behavior section must reference a supported run method: "
                "`agent-canvas`, `npm run`, or `app.all-hands.dev/canvas`."
            )
        if not has_screenshot_or_video(actual):
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


def evaluate_readiness(body: str, labels: list[str]) -> ReadinessResult:
    """Return the readiness result for an issue body + label set.

    An issue is only a candidate when it carries the `bug` or `enhancement`
    label. If it has neither, it is treated as not-ready-for-dev (the gate does
    not apply a label it cannot validate).
    """
    label_set = {label.lower() for label in labels}
    sections = extract_sections(body or "")

    if BUG_LABEL in label_set:
        return check_bug(sections)
    if ENHANCEMENT_LABEL in label_set:
        return check_enhancement(sections)

    return ReadinessResult(
        ready=False,
        reasons=[
            "The issue has neither the `bug` nor `enhancement` label, so its "
            "readiness criteria cannot be evaluated. Add the appropriate label."
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
