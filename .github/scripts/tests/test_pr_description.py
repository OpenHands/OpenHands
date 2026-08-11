"""Tests for check_pr_description.py — PR description validation logic."""

import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_pr_description import (
    extract_human_note,
    extract_sections,
    validate_pr_body,
    is_frontend_file,
    touches_frontend,
    extract_linked_issue_numbers,
    extract_pr_type,
    validate_linked_issue_ready,
    validate_bug_fix_evidence,
    BUG_LABEL,
    ENHANCEMENT_LABEL,
)


def test_extract_sections_ignores_heading_inside_fence():
    body = """## Summary
The template says:

```markdown
## How to Test
Describe testing here
```
"""
    assert set(extract_sections(body)) == {"Summary"}


def test_fenced_heading_does_not_truncate_how_to_test():
    body = """## How to Test
Run the focused checker tests.

~~~text
## Example output
all tests passed
~~~

Then run the reproduction script.

## Issue Number
Fixes #16553
"""
    sections = extract_sections(body)
    assert "Example output" not in sections
    assert "Then run the reproduction script." in sections["How to Test"]

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


def test_markdown_under_frontend_prefix_is_not_frontend():
    assert not is_frontend_file("__tests__/router.md")
    assert not is_frontend_file("src/notes.md")
    assert not is_frontend_file("public/README.mdx")

def test_markdown_outside_frontend_prefix_still_not_frontend():
    assert not is_frontend_file("docs/README.md")

def test_frontend_code_under_prefix_still_frontend():
    assert is_frontend_file("src/app.tsx")
    assert is_frontend_file("__tests__/routes/launch.test.tsx")
    assert is_frontend_file("src/styles/main.css")

def test_docs_only_change_does_not_require_frontend_evidence():
    assert not touches_frontend(["__tests__/router.md", "docs/README.md"])

def test_mixed_change_still_requires_frontend_evidence():
    assert touches_frontend(["__tests__/router.md", "src/app.tsx"])


# ---------------------------------------------------------------------------
# HUMAN: / AGENT: markers
# ---------------------------------------------------------------------------


FENCED_TEMPLATE_BODY = """HUMAN:

Quoting the template I was asked to fill in:

```
HUMAN:
I really did test this thoroughly by hand, twice, on my own machine.
AGENT:
```
"""


def test_fenced_human_marker_does_not_supply_the_note():
    assert extract_human_note(FENCED_TEMPLATE_BODY) == ""


def test_fenced_agent_marker_does_not_satisfy_validation():
    errors = validate_pr_body(FENCED_TEMPLATE_BODY)
    assert "Add a short human-written note between `HUMAN:` and `AGENT:`." in errors
    assert "Keep the `AGENT:` marker from the PR template." in errors


def test_real_markers_still_supply_the_note():
    body = """HUMAN:

I ran the checker suite and reproduced the fenced-heading case by hand.

AGENT:

## Why
x
"""
    assert "reproduced the fenced-heading case" in extract_human_note(body)
    errors = validate_pr_body(body)
    assert "Keep the `AGENT:` marker from the PR template." not in errors
    assert "Add a short human-written note between `HUMAN:` and `AGENT:`." not in errors
