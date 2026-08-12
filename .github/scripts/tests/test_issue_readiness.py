"""Tests for check_issue_readiness.py — the ready-for-dev gate logic."""

import sys
from pathlib import Path

# Make the sibling script importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_issue_readiness import (
    evaluate_readiness,
    extract_sections,
    has_screenshot_or_video,
    mask_fenced_code,
    references_run_method,
    has_checklist_item,
    visible_text,
    BUG_LABEL,
    ENHANCEMENT_LABEL,
)

# Built at runtime so these fixtures stay readable and quotable.
FENCE = "`" * 3
TILDE_FENCE = "~" * 3

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

BUG_BODY_READY = """### Actual Behavior
I ran `npm run dev` and saw this:

![screenshot](https://github.com/user-attachments/assets/abc123)

The button was misaligned.

### Expected Behavior
The button should be centered.

### Acceptance Criteria
- [ ] Button is centered
- [ ] No layout shift on resize
"""

BUG_BODY_NO_RUN_METHOD = """### Actual Behavior
The button was misaligned.

### Acceptance Criteria
- [ ] Button is centered
"""

BUG_BODY_NO_SCREENSHOT = """### Actual Behavior
I ran `npm run dev` and saw the button was misaligned.

### Acceptance Criteria
- [ ] Button is centered
"""

BUG_BODY_NO_ACCEPTANCE = """### Actual Behavior
I ran `npm run dev` and saw this:

![screenshot](https://github.com/user-attachments/assets/abc123)
"""

BUG_BODY_EMPTY_ACTUAL = """### Actual Behavior
_No response_

### Acceptance Criteria
- [ ] Button is centered
"""

BUG_BODY_AGENT_CANVAS = """### Actual Behavior
I used agent-canvas to reproduce this.

![screenshot](https://example.com/screenshot.png)

### Acceptance Criteria
- [ ] Fixed
"""

BUG_BODY_HOSTED_URL = """### Actual Behavior
Reproduced on app.all-hands.dev/canvas — see video below.

<video src="https://example.com/bug.mp4"></video>

### Acceptance Criteria
- [ ] Fixed
"""

ENHANCEMENT_BODY_READY = """### Desired Behavior
The button should animate on hover.

### Acceptance Criteria
- [ ] Hover animation works
- [ ] No perf regression
"""

ENHANCEMENT_BODY_NO_DESIRED = """### Acceptance Criteria
- [ ] Something
"""

ENHANCEMENT_BODY_NO_ACCEPTANCE = """### Desired Behavior
The button should animate on hover.
"""

ENHANCEMENT_BODY_PROSE_ACCEPTANCE = """### Desired Behavior
The button should animate on hover.

### Acceptance Criteria
Make it look nice.
"""


# ---------------------------------------------------------------------------
# Bug readiness
# ---------------------------------------------------------------------------

def test_bug_ready_npm_run_screenshot():
    result = evaluate_readiness(BUG_BODY_READY, [BUG_LABEL])
    assert result.ready, result.reasons

def test_bug_ready_agent_canvas():
    result = evaluate_readiness(BUG_BODY_AGENT_CANVAS, [BUG_LABEL])
    assert result.ready, result.reasons

def test_bug_ready_hosted_url():
    result = evaluate_readiness(BUG_BODY_HOSTED_URL, [BUG_LABEL])
    assert result.ready, result.reasons

def test_bug_not_ready_no_run_method():
    result = evaluate_readiness(BUG_BODY_NO_RUN_METHOD, [BUG_LABEL])
    assert not result.ready
    assert any("run method" in r for r in result.reasons)

def test_bug_not_ready_no_screenshot():
    result = evaluate_readiness(BUG_BODY_NO_SCREENSHOT, [BUG_LABEL])
    assert not result.ready
    assert any("screenshot" in r for r in result.reasons)

def test_bug_not_ready_no_acceptance():
    result = evaluate_readiness(BUG_BODY_NO_ACCEPTANCE, [BUG_LABEL])
    assert not result.ready
    assert any("Acceptance Criteria" in r for r in result.reasons)

def test_bug_not_ready_empty_actual():
    result = evaluate_readiness(BUG_BODY_EMPTY_ACTUAL, [BUG_LABEL])
    assert not result.ready
    assert any("Actual Behavior" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Enhancement readiness
# ---------------------------------------------------------------------------

def test_enhancement_ready():
    result = evaluate_readiness(ENHANCEMENT_BODY_READY, [ENHANCEMENT_LABEL])
    assert result.ready, result.reasons

def test_enhancement_not_ready_no_desired():
    result = evaluate_readiness(ENHANCEMENT_BODY_NO_DESIRED, [ENHANCEMENT_LABEL])
    assert not result.ready
    assert any("Desired Behavior" in r for r in result.reasons)

def test_enhancement_not_ready_no_acceptance():
    result = evaluate_readiness(ENHANCEMENT_BODY_NO_ACCEPTANCE, [ENHANCEMENT_LABEL])
    assert not result.ready
    assert any("Acceptance Criteria" in r for r in result.reasons)

def test_enhancement_not_ready_prose_acceptance():
    result = evaluate_readiness(ENHANCEMENT_BODY_PROSE_ACCEPTANCE, [ENHANCEMENT_LABEL])
    assert not result.ready
    assert any("checklist" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# No type label
# ---------------------------------------------------------------------------

def test_no_type_label_not_ready():
    result = evaluate_readiness("### Something\nSome text", ["frontend"])
    assert not result.ready
    assert any("neither" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Unit-level helpers
# ---------------------------------------------------------------------------

def test_has_screenshot_markdown_image():
    assert has_screenshot_or_video("![alt](https://example.com/img.png)")

def test_has_screenshot_github_attachment():
    assert has_screenshot_or_video("https://github.com/user-attachments/assets/abc123")

def test_has_screenshot_html_video():
    assert has_screenshot_or_video('<video src="bug.mp4"></video>')

def test_has_screenshot_youtube():
    assert has_screenshot_or_video("https://youtube.com/watch?v=abc123")

def test_has_screenshot_none():
    assert not has_screenshot_or_video("Just text, no media")

def test_references_run_method_npm():
    assert references_run_method("I ran npm run dev")

def test_references_run_method_agent_canvas():
    assert references_run_method("Used agent-canvas to test")

def test_references_run_method_hosted():
    assert references_run_method("Reproduced on app.all-hands.dev/canvas")

def test_references_run_method_none():
    assert not references_run_method("I clicked the button")

def test_has_checklist_item():
    assert has_checklist_item("- [ ] Do something")
    assert has_checklist_item("- [x] Done")
    assert has_checklist_item("  * [ ] Indented")

def test_has_checklist_item_none():
    assert not has_checklist_item("Just prose, no checklist")

def test_visible_text_strips_html_comments():
    assert visible_text("<!-- hidden -->visible text") == "visible text"

def test_visible_text_no_response():
    assert visible_text("_No response_") == ""

def test_extract_sections():
    sections = extract_sections("### Title One\nText 1\n### Title Two\nText 2")
    assert "title one" in sections
    assert "title two" in sections
    assert "Text 1" in sections["title one"]
    assert "Text 2" in sections["title two"]


# ---------------------------------------------------------------------------
# Fenced code blocks are not section boundaries
# ---------------------------------------------------------------------------

def test_fenced_heading_is_not_a_section():
    """A heading quoted inside a fence must not become a section."""
    body = (
        "### Desired Behavior\n\nMake it work.\n\n"
        f"{FENCE}markdown\n### Acceptance Criteria\n- [ ] quoted, not real\n{FENCE}\n"
    )
    assert "acceptance criteria" not in extract_sections(body)


def test_enhancement_not_ready_when_acceptance_only_quoted():
    """Quoting the template must not satisfy the acceptance-criteria rule."""
    body = (
        "### Desired Behavior\n\nMake it work.\n\n"
        f"{FENCE}\n### Acceptance Criteria\n- [ ] quoted, not real\n{FENCE}\n"
    )
    result = evaluate_readiness(body, [ENHANCEMENT_LABEL])
    assert not result.ready
    assert any("Acceptance Criteria" in r for r in result.reasons)


def test_fenced_heading_does_not_truncate_enclosing_section():
    """A pasted log containing `### ` must not cut its section short."""
    body = (
        "### Actual Behavior\n\nI ran `npm run dev` and the server said:\n\n"
        f"{FENCE}\n### Error detail\nboom\n{FENCE}\n\n"
        "![screenshot](https://github.com/user-attachments/assets/abc123)\n\n"
        "### Acceptance Criteria\n- [ ] fixed\n"
    )
    sections = extract_sections(body)
    assert "error detail" not in sections
    assert "user-attachments" in sections["actual behavior"]


def test_bug_ready_with_fenced_log_before_screenshot():
    """End-to-end: the report above is ready; before the fix it was rejected."""
    body = (
        "### Actual Behavior\n\nI ran `npm run dev` and the server said:\n\n"
        f"{FENCE}\n### Error detail\nboom\n{FENCE}\n\n"
        "![screenshot](https://github.com/user-attachments/assets/abc123)\n\n"
        "### Acceptance Criteria\n- [ ] fixed\n"
    )
    result = evaluate_readiness(body, [BUG_LABEL])
    assert result.ready, result.reasons


def test_tilde_fence_is_masked():
    body = (
        "### Desired Behavior\n\nMake it work.\n\n"
        f"{TILDE_FENCE}\n### Acceptance Criteria\n- [ ] quoted\n{TILDE_FENCE}\n"
    )
    assert "acceptance criteria" not in extract_sections(body)


def test_backticks_do_not_close_a_tilde_fence():
    """Fences only close on their own marker character."""
    body = (
        "### Desired Behavior\n\ntext\n\n"
        f"{TILDE_FENCE}\n{FENCE}\n### Acceptance Criteria\n- [ ] still inside\n"
        f"{TILDE_FENCE}\n"
    )
    assert "acceptance criteria" not in extract_sections(body)


def test_indented_fence_is_masked():
    """CommonMark allows a fence indented up to three spaces."""
    body = (
        "### Desired Behavior\n\ntext\n\n"
        f"   {FENCE}\n   ### Acceptance Criteria\n   - [ ] quoted\n   {FENCE}\n"
    )
    assert "acceptance criteria" not in extract_sections(body)


def test_unterminated_fence_masks_to_end_of_body():
    """An unclosed fence runs to the end, matching CommonMark and GitHub."""
    body = f"### Desired Behavior\n\ntext\n\n{FENCE}\n### Acceptance Criteria\n- [ ] x\n"
    sections = extract_sections(body)
    assert "desired behavior" in sections
    assert "acceptance criteria" not in sections


def test_longer_marker_closes_shorter_fence():
    body = (
        "### Desired Behavior\n\ntext\n\n"
        f"{FENCE}\n### Quoted\n{FENCE}`\n\n### Acceptance Criteria\n- [ ] real\n"
    )
    sections = extract_sections(body)
    assert "quoted" not in sections
    assert "acceptance criteria" in sections


def test_sections_after_a_fence_are_still_found():
    """Regression: masking must not swallow real headings that follow."""
    body = (
        "### Actual Behavior\n\nran `npm run dev`\n\n"
        f"{FENCE}\nsome log\n{FENCE}\n\n"
        "![shot](https://example.com/a.png)\n\n"
        "### Acceptance Criteria\n\n- [ ] fixed\n"
    )
    sections = extract_sections(body)
    assert set(sections) == {"actual behavior", "acceptance criteria"}


def test_mask_fenced_code_preserves_offsets():
    """Section slicing reads from the original body, so length must not shift."""
    body = f"### A\ntext\n{FENCE}\n### B\n{FENCE}\n### C\ntail\n"
    masked = mask_fenced_code(body)
    assert len(masked) == len(body)
    assert masked.count("\n") == body.count("\n")


def test_mask_fenced_code_leaves_unfenced_text_untouched():
    body = "### A\nplain text\n### B\nmore text\n"
    assert mask_fenced_code(body) == body


def test_fence_directly_after_heading_is_kept_in_the_section():
    """`HEADING_RE` ends in `\\s*$`; a masked fence must not be swallowed by it."""
    body = f"### Actual Behavior\n{FENCE}\n$ npm run dev\nboom\n{FENCE}\n\ntail\n"
    section = extract_sections(body)["actual behavior"]
    assert "npm run dev" in section
    assert "boom" in section


def test_bug_ready_when_section_opens_with_a_fenced_log():
    """The run method lives only in a fence that opens the section."""
    body = (
        f"### Actual Behavior\n{FENCE}\n$ npm run dev\nboom\n{FENCE}\n\n"
        "![shot](https://github.com/user-attachments/assets/abc123)\n\n"
        "### Acceptance Criteria\n- [ ] fixed\n"
    )
    result = evaluate_readiness(body, [BUG_LABEL])
    assert result.ready, result.reasons


def test_blank_lines_after_heading_do_not_eat_the_section():
    body = "### Actual Behavior\n\n\n\nran `npm run dev`\n\n### Expected Behavior\nx\n"
    assert "npm run dev" in extract_sections(body)["actual behavior"]


def test_form_feed_does_not_open_a_fence():
    """`str.splitlines()` breaks on \\x0c; CommonMark does not."""
    body = (
        f"### Actual Behavior\npage one\x0c{FENCE}\n\n"
        "### Acceptance Criteria\n- [ ] x\n"
    )
    assert "acceptance criteria" in extract_sections(body)


def test_crlf_body_sections_are_found():
    body = (
        "### Actual Behavior\r\nran `npm run dev`\r\n\r\n"
        f"{FENCE}\r\n### Quoted\r\n{FENCE}\r\n\r\n"
        "![shot](https://example.com/a.png)\r\n\r\n"
        "### Acceptance Criteria\r\n- [ ] fixed\r\n"
    )
    sections = extract_sections(body)
    assert "quoted" not in sections
    assert set(sections) == {"actual behavior", "acceptance criteria"}


def test_crlf_body_is_ready():
    body = (
        "### Actual Behavior\r\nran `npm run dev`\r\n\r\n"
        "![shot](https://example.com/a.png)\r\n\r\n"
        "### Acceptance Criteria\r\n- [ ] fixed\r\n"
    )
    result = evaluate_readiness(body, [BUG_LABEL])
    assert result.ready, result.reasons
