"""Tests for check_issue_readiness.py — the ready-for-dev gate logic."""

import sys
from pathlib import Path

# Make the sibling script importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_issue_readiness import (
    evaluate_readiness,
    extract_sections,
    has_screenshot_or_video,
    references_run_method,
    has_checklist_item,
    visible_text,
    BUG_LABEL,
    ENHANCEMENT_LABEL,
)

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
# Fenced-block regression coverage for issues #16553 and #16583.
#
# A heading inside a fence must not become a section; quoted template text
# inside a fence must not satisfy run-method / screenshot / checklist rules.
# ---------------------------------------------------------------------------


def test_extract_sections_ignores_fenced_heading():
    sections = extract_sections("### One\n```\n### Two\n```\n### Three\n")
    assert "one" in sections
    assert "two" not in sections
    assert "three" in sections
    # The fenced "### Two" should be present in the prior section's text.
    assert "### Two" in sections["one"]


def test_extract_sections_does_not_truncate_at_invalid_close():
    # A close with trailing text is invalid per CommonMark; the section that
    # contains it must keep the content that follows.
    body = (
        "### Actual Behavior\n"
        "Run method: npm run dev\n\n"
        "```\n"
        "### Error detail\n"
        "garbage closer: ``` more text\n"
        "```\n\n"
        "<img src='https://github.com/user-attachments/assets/abc123' />\n\n"
        "### Acceptance Criteria\n"
        "- [ ] bug is fixed\n"
    )
    sections = extract_sections(body)
    assert "actual behavior" in sections
    # The "### Error detail" line is inside a fence, so it must not be treated
    # as a section header. The "more text" closer is invalid, so the fenced
    # block runs through it.
    assert "error detail" not in sections
    # The screenshot is preserved inside the actual-behavior section.
    assert "user-attachments/assets/abc123" in sections["actual behavior"]


def test_references_run_method_ignores_fenced_npm_run():
    # The script must not count `npm run` written inside a code fence.
    assert not references_run_method("```\nnpm run dev\n```")
    assert not references_run_method("```bash\nnpm run dev\n```\n")


def test_references_run_method_keeps_real_npm_run_beside_fenced_quote():
    actual = (
        "Real reproduction: I ran npm run dev and saw the bug.\n"
        "```\nExample only: npm run something-else\n```\n"
    )
    assert references_run_method(actual)


def test_has_screenshot_ignores_fenced_image():
    assert not has_screenshot_or_video("```\n![shot](https://example.com/x.png)\n```")


def test_has_screenshot_keeps_real_image_beside_fenced_quote():
    actual = (
        "Real screenshot:\n\n"
        "![real](https://github.com/user-attachments/assets/real123)\n"
        "```\n![example](https://github.com/user-attachments/assets/example)\n```\n"
    )
    assert has_screenshot_or_video(actual)


def test_has_checklist_ignores_fenced_checklist():
    assert not has_checklist_item("```\n- [ ] quoted template item\n```")


def test_has_checklist_keeps_real_item_beside_fenced_quote():
    text = (
        "- [ ] real criterion\n"
        "```\n- [ ] template item\n```\n"
    )
    assert has_checklist_item(text)


def test_bug_ready_with_fenced_heading_in_actual():
    # The bug body in #16553's reproduction: a log with a `### Error detail`
    # line inside a fence. The real `### Acceptance Criteria` lives outside
    # the fence, so the bug report should still pass the gate.
    body = """### Actual Behavior

Run method: `npm run dev`

```
### Error detail
something went wrong
```

<img width="800" alt="Image" src="https://github.com/user-attachments/assets/abc123" />

### Acceptance Criteria

- [ ] the bug is fixed
"""
    result = evaluate_readiness(body, [BUG_LABEL])
    assert result.ready, result.reasons


def test_enhancement_not_ready_with_only_fenced_acceptance():
    # The enhancement body in #16553's reproduction: the only acceptance
    # criteria are inside a fence. The gate must still reject it.
    body = """### Desired Behavior

Make the thing work.

### Notes

The template asks for:

```markdown
### Acceptance Criteria
- [ ] checklist items go here
```
"""
    result = evaluate_readiness(body, [ENHANCEMENT_LABEL])
    assert not result.ready
    assert any("Acceptance Criteria" in r for r in result.reasons)


def test_bug_not_ready_with_only_fenced_examples():
    body = """### Actual Behavior

Nothing here describes a real reproduction.

```
Run method: `npm run dev`
![shot](https://github.com/user-attachments/assets/abc123)
```

### Acceptance Criteria

```
- [ ] a checklist item quoted from the template, not a real criterion
```
"""
    result = evaluate_readiness(body, [BUG_LABEL])
    assert not result.ready
    assert any("run method" in r for r in result.reasons)
    assert any("screenshot" in r for r in result.reasons)
    assert any("Acceptance Criteria" in r for r in result.reasons)
