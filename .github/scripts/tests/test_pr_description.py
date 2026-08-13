"""Tests for check_pr_description.py — PR description validation logic."""

import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_pr_description import (
    extract_linked_issue_numbers,
    extract_pr_type,
    extract_sections,
    mask_fenced_code,
    validate_linked_issue_ready,
    validate_bug_fix_evidence,
    validate_pr_body,
    BUG_LABEL,
    ENHANCEMENT_LABEL,
    READY_FOR_DEV_LABEL,
)

# Built at runtime so these fixtures stay readable and quotable.
FENCE = "`" * 3

# ---------------------------------------------------------------------------
# extract_linked_issue_numbers
# ---------------------------------------------------------------------------

def test_extract_fixes():
    assert extract_linked_issue_numbers("## Issue Number\n\nFixes #123\n") == [123]

def test_extract_fix_singular():
    assert extract_linked_issue_numbers("## Issue Number\n\nFix #123\n") == [123]

def test_extract_closes():
    assert extract_linked_issue_numbers("## Issue Number\n\nCloses #456\n") == [456]

def test_extract_close_singular():
    assert extract_linked_issue_numbers("## Issue Number\n\nClose #456\n") == [456]

def test_extract_resolves():
    assert extract_linked_issue_numbers("## Issue Number\n\nResolves #789\n") == [789]

def test_extract_resolve_singular():
    assert extract_linked_issue_numbers("## Issue Number\n\nResolve #789\n") == [789]

def test_extract_bare_in_issue_section():
    assert extract_linked_issue_numbers("## Issue Number\n\n#42\n") == [42]

def test_extract_none_when_no_ref():
    assert extract_linked_issue_numbers("## Issue Number\n\nNo existing issue.\n") == []

def test_extract_multiple():
    assert extract_linked_issue_numbers("## Issue Number\n\nFixes #1 and Closes #2\n") == [1, 2]

def test_bare_outside_section_ignored():
    body = "Some text #99 here\n## Issue Number\n\nNo issue\n"
    assert extract_linked_issue_numbers(body) == []

def test_fixes_anywhere_in_body():
    body = "See Fixes #55 in the summary\n## Issue Number\n\n"
    assert extract_linked_issue_numbers(body) == [55]


# ---------------------------------------------------------------------------
# validate_linked_issue_ready — local mode (no API)
# ---------------------------------------------------------------------------

def test_local_mode_has_issue_ref_no_errors():
    assert validate_linked_issue_ready("## Issue Number\n\nFixes #123\n") == []

def test_local_mode_no_issue_ref_errors():
    errors = validate_linked_issue_ready("## Issue Number\n\nNo issue\n")
    assert len(errors) == 1
    assert "Link an issue" in errors[0]


# ---------------------------------------------------------------------------
# validate_linked_issue_ready — API mode (mocked)
# ---------------------------------------------------------------------------

def test_api_issue_has_ready_for_dev():
    with patch("check_pr_description.fetch_issue_labels", return_value=["enhancement", "ready-for-dev"]):
        errors = validate_linked_issue_ready("Fixes #123\n", "owner/repo", "fake-token")
    assert errors == []

def test_api_issue_missing_ready_for_dev():
    with patch("check_pr_description.fetch_issue_labels", return_value=["enhancement"]):
        errors = validate_linked_issue_ready("Fixes #123\n", "owner/repo", "fake-token")
    assert len(errors) == 1
    assert "ready-for-dev" in errors[0]

def test_api_one_of_multiple_has_ready_for_dev():
    labels_map = {123: ["bug"], 456: ["enhancement", "ready-for-dev"]}
    def mock_labels(repo, number, token):
        return labels_map.get(number, [])
    with patch("check_pr_description.fetch_issue_labels", side_effect=mock_labels):
        errors = validate_linked_issue_ready("Fixes #123 and Closes #456\n", "owner/repo", "fake-token")
    assert errors == []

def test_api_issue_404():
    with patch("check_pr_description.fetch_issue_labels", side_effect=urllib.error.HTTPError(None, 404, "Not Found", {}, None)):
        errors = validate_linked_issue_ready("Fixes #999\n", "owner/repo", "fake-token")
    assert len(errors) == 1
    assert "could not" in errors[0]


# ---------------------------------------------------------------------------
# extract_pr_type
# ---------------------------------------------------------------------------

def test_pr_type_bug_checked():
    body = "## Type\n\n- [x] Bug fix\n- [ ] Feature\n"
    assert extract_pr_type(body) == BUG_LABEL

def test_pr_type_feature_checked():
    body = "## Type\n\n- [ ] Bug fix\n- [x] Feature\n"
    assert extract_pr_type(body) == ENHANCEMENT_LABEL

def test_pr_type_nothing_checked():
    body = "## Type\n\n- [ ] Bug fix\n- [ ] Feature\n"
    assert extract_pr_type(body) is None

def test_pr_type_no_section():
    body = "## Summary\n\nNo type section\n"
    assert extract_pr_type(body) is None


# ---------------------------------------------------------------------------
# Type cross-check (mocked API)
# ---------------------------------------------------------------------------

def test_type_check_bug_pr_bug_issue_no_errors():
    body = "## Issue Number\n\nFixes #123\n\n## Type\n\n- [x] Bug fix\n- [ ] Feature\n"
    with patch("check_pr_description.fetch_issue_labels", return_value=["bug", "ready-for-dev"]):
        errors = validate_linked_issue_ready(body, "owner/repo", "fake-token")
    assert errors == []

def test_type_check_bug_pr_enhancement_issue_mismatch():
    body = "## Issue Number\n\nFixes #123\n\n## Type\n\n- [x] Bug fix\n- [ ] Feature\n"
    with patch("check_pr_description.fetch_issue_labels", return_value=["enhancement", "ready-for-dev"]):
        errors = validate_linked_issue_ready(body, "owner/repo", "fake-token")
    assert any("bug" in e for e in errors)

def test_type_check_feature_pr_enhancement_issue_no_errors():
    body = "## Issue Number\n\nFixes #123\n\n## Type\n\n- [ ] Bug fix\n- [x] Feature\n"
    with patch("check_pr_description.fetch_issue_labels", return_value=["enhancement", "ready-for-dev"]):
        errors = validate_linked_issue_ready(body, "owner/repo", "fake-token")
    assert errors == []

def test_type_check_feature_pr_bug_issue_mismatch():
    body = "## Issue Number\n\nFixes #123\n\n## Type\n\n- [ ] Bug fix\n- [x] Feature\n"
    with patch("check_pr_description.fetch_issue_labels", return_value=["bug", "ready-for-dev"]):
        errors = validate_linked_issue_ready(body, "owner/repo", "fake-token")
    assert any("enhancement" in e for e in errors)

def test_type_check_no_type_checked_no_type_error():
    body = "## Issue Number\n\nFixes #123\n"
    with patch("check_pr_description.fetch_issue_labels", return_value=["ready-for-dev"]):
        errors = validate_linked_issue_ready(body, "owner/repo", "fake-token")
    assert errors == []


# ---------------------------------------------------------------------------
# validate_bug_fix_evidence
# ---------------------------------------------------------------------------

BUG_FIX_BODY_NO_EVIDENCE = """## Type

- [x] Bug fix
- [ ] Feature
"""

BUG_FIX_BODY_WITH_SCREENSHOT = """## Type

- [x] Bug fix
- [ ] Feature

## Video/Screenshots

![before](https://example.com/before.png) → ![after](https://example.com/after.png)
"""

FEATURE_BODY = """## Type

- [ ] Bug fix
- [x] Feature
"""

def test_bug_fix_no_screenshot_errors():
    errors = validate_bug_fix_evidence(BUG_FIX_BODY_NO_EVIDENCE)
    assert len(errors) == 1
    assert "reproduction evidence" in errors[0].lower()

def test_bug_fix_with_screenshot_no_errors():
    errors = validate_bug_fix_evidence(BUG_FIX_BODY_WITH_SCREENSHOT)
    assert errors == []

def test_feature_no_bug_evidence_check():
    errors = validate_bug_fix_evidence(FEATURE_BODY)
    assert errors == []

def test_no_type_no_bug_evidence_check():
    errors = validate_bug_fix_evidence("## Summary\n\nNo type\n")
    assert errors == []

def test_bug_fix_with_github_attachment_no_errors():
    body = """## Type

- [x] Bug fix

## Video/Screenshots

https://github.com/user-attachments/assets/abc123
"""
    errors = validate_bug_fix_evidence(body)
    assert errors == []

def test_bug_fix_with_video_link_no_errors():
    body = """## Type

- [x] Bug fix

## Video/Screenshots

https://youtube.com/watch?v=abc123
"""
    errors = validate_bug_fix_evidence(body)
    assert errors == []


# ---------------------------------------------------------------------------
# Fenced code blocks are not section boundaries
# ---------------------------------------------------------------------------

def test_fenced_heading_is_not_a_section():
    """A template quoted inside a fence must not manufacture a section."""
    body = f"## Why\n\nreal reason\n\n{FENCE}\n## How to Test\n1. npm ci\n{FENCE}\n"
    assert "How to Test" not in extract_sections(body)


def test_quoted_required_section_does_not_satisfy_template_check():
    """Before the fix, a fenced `## How to Test` satisfied the required-field check."""
    body = (
        "HUMAN:\n\nI tested this locally on Node 24.\n\nAGENT:\n\n"
        "## Why\n\nreal reason\n\n"
        "## Summary\n\nExample of the template we follow:\n\n"
        f"{FENCE}\n## How to Test\n1. npm ci\n{FENCE}\n\n"
        "## Issue Number\n\nFixes #123\n"
    )
    errors = validate_pr_body(body, [])
    assert any("How to Test" in e for e in errors)


def test_fenced_heading_does_not_truncate_enclosing_section():
    """Command output containing `## ` must not cut its section short."""
    body = (
        "## How to Test\n\n1. npm ci\n\n"
        f"{FENCE}\n## Output\nall green\n{FENCE}\n\n"
        "2. npm run build\n\n## Type\n\n- [x] Refactor\n"
    )
    sections = extract_sections(body)
    assert "Output" not in sections
    assert "npm run build" in sections["How to Test"]


def test_real_sections_still_parsed_around_fences():
    body = (
        "## Why\n\nbecause\n\n"
        f"{FENCE}\nsome log\n{FENCE}\n\n"
        "## Summary\n\nchanged things\n\n## How to Test\n\nrun it\n"
    )
    assert set(extract_sections(body)) == {"Why", "Summary", "How to Test"}


def test_mask_fenced_code_preserves_offsets():
    body = f"## A\ntext\n{FENCE}\n## B\n{FENCE}\n## C\ntail\n"
    masked = mask_fenced_code(body)
    assert len(masked) == len(body)
    assert masked.count("\n") == body.count("\n")


def test_mask_fenced_code_leaves_unfenced_text_untouched():
    body = "## A\nplain text\n## B\nmore text\n"
    assert mask_fenced_code(body) == body


def test_fence_directly_after_heading_is_kept_in_the_section():
    """`HEADING_RE` ends in `\\s*$`; a masked fence must not be swallowed by it."""
    body = f"## How to Test\n{FENCE}\nnpm ci && npm test\n{FENCE}\n\n## Type\n\n- [x] Refactor\n"
    section = extract_sections(body)["How to Test"]
    assert "npm ci && npm test" in section


def test_how_to_test_with_only_a_fenced_command_is_not_empty():
    """A section whose entire content is a fenced block must still count as filled."""
    body = (
        "HUMAN:\n\nI ran this locally on Node 24.\n\nAGENT:\n\n"
        "## Why\n\nreason\n\n## Summary\n\nchanges\n\n"
        f"## How to Test\n{FENCE}\nnpm ci && npm test\n{FENCE}\n\n"
        "## Issue Number\n\nFixes #123\n"
    )
    errors = validate_pr_body(body, [])
    assert not any("How to Test" in e for e in errors), errors


def test_form_feed_does_not_open_a_fence():
    body = f"## Why\nreason\x0c{FENCE}\n\n## Summary\nchanges\n"
    assert "Summary" in extract_sections(body)


def test_crlf_body_sections_are_found():
    body = (
        "## Why\r\nreason\r\n\r\n"
        f"{FENCE}\r\n## Quoted\r\n{FENCE}\r\n\r\n"
        "## Summary\r\nchanges\r\n"
    )
    sections = extract_sections(body)
    assert "Quoted" not in sections
    assert set(sections) == {"Why", "Summary"}


def test_closing_marker_with_trailing_text_does_not_close():
    """CommonMark: a closing fence is followed only by spaces or tabs."""
    body = f"## Why\n\nreason\n\n{FENCE}\n{FENCE}not a close\n## How to Test\n1. npm ci\n"
    assert "How to Test" not in extract_sections(body)


def test_tab_indented_marker_is_not_a_fence():
    body = f"## Why\n\nreason\n\t{FENCE}\n\n## How to Test\n\nrun it\n"
    assert "How to Test" in extract_sections(body)


def test_backtick_in_backtick_fence_info_string_is_not_an_opener():
    body = f"## Why\n\nreason\n{FENCE}lang`bad\n\n## How to Test\n\nrun it\n"
    assert "How to Test" in extract_sections(body)
