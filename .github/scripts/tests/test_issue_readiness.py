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
# Fenced code block edge cases (issue #16553)
# ---------------------------------------------------------------------------

def test_fenced_heading_not_parsed_as_section():
    """A ### heading inside a fenced code block should NOT become a section."""
    body = """### Actual Behavior

Run method: `npm run dev`

The server returned:

```
### Error detail
something went wrong
```

<img width="800" alt="Image" src="https://github.com/user-attachments/assets/abc123" />

### Acceptance Criteria

- [ ] the bug is fixed
"""
    sections = extract_sections(body)
    assert "error detail" not in sections, f"Unexpected section: {sections}"
    assert "actual behavior" in sections
    assert "acceptance criteria" in sections

def test_fenced_acceptance_criteria_not_counted():
    """Only Acceptance Criteria inside a fence should not grant readiness."""
    body = """### Actual Behavior

Run method: `npm run dev`

![screenshot](https://github.com/user-attachments/assets/abc123)

### Notes

The template asks for:

```markdown
### Acceptance Criteria
- [ ] checklist items go here
```
"""
    result = evaluate_readiness(body, [BUG_LABEL])
    assert not result.ready, f"Should not be ready: {result.reasons}"

def test_actual_behavior_with_fenced_block_preserved():
    """A fenced block inside Actual Behavior should not truncate the section."""
    body = """### Actual Behavior

Run method: `npm run dev`

The server returned:

```
### Error detail
something went wrong
```

<img width="800" alt="Image" src="https://github.com/user-attachments/assets/abc123" />

### Acceptance Criteria

- [ ] the bug is fixed
"""
    result = evaluate_readiness(body, [BUG_LABEL])
    assert result.ready, f"Should be ready: {result.reasons}"

def test_different_fence_types():
    """Fences with tildes and backticks should both work."""
    body_tilde = """### Actual Behavior

Run method: `npm run dev`

Some text:

~~~
### Not a section
~~~

More text here.

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria

- [ ] the bug is fixed
"""
    body_backtick = """### Actual Behavior

Run method: `npm run dev`

Some text:

```
### Not a section
```

More text here.

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria

- [ ] the bug is fixed
"""
    result_tilde = evaluate_readiness(body_tilde, [BUG_LABEL])
    result_backtick = evaluate_readiness(body_backtick, [BUG_LABEL])
    assert result_tilde.ready, f"Tilde fence failed: {result_tilde.reasons}"
    assert result_backtick.ready, f"Backtick fence failed: {result_backtick.reasons}"

def test_nested_fences_dont_break():
    """Closing a fence, then having more text should work correctly."""
    body = """### Actual Behavior

Run method: `npm run dev`

Some text:

```
Line one
```
More text after the fence.

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria

- [ ] the bug is fixed
"""
    result = evaluate_readiness(body, [BUG_LABEL])
    assert result.ready, f"Should be ready: {result.reasons}"

def test_fence_content_includes_caret():
    """Content after a closing fence should belong to the correct section."""
    body = """### Actual Behavior

Run method: `npm run dev`

Before fence.

```
### Inside fence
```

After fence with screenshot.

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria

- [ ] the bug is fixed
"""
    sections = extract_sections(body)
    actual = sections.get("actual behavior", "")
    assert "Before fence." in actual, f"Missing text in actual behavior: {actual}"
    assert "After fence with screenshot." in actual, f"Missing text in actual behavior: {actual}"

def test_real_issue_16553_body_passes():
    """The actual issue body from #16553 should pass readiness after the fix."""
    # Build the body without nested triple quotes to avoid syntax issues
    body = """### Operating System

macOS

### Installation Method

From source (`npm run dev` / `npm run dev:minimal`)

### Agent Canvas Version

main (1.12.0)

### Bug Description

`extract_sections()` in `.github/scripts/check_issue_readiness.py` splits an issue body on `###` with no awareness of fenced code blocks.

This cuts both ways.

**A quoted heading grants readiness it should not.** An issue whose only `### Acceptance Criteria` appears inside a fence — for example when quoting the issue template — passes the gate despite having no real acceptance criteria.

**A fenced heading inside a real section silently truncates it.** Because the next `###` match ends the previous section, a bug report that pastes a log or markdown snippet containing a `###` line loses everything after that fence. If the screenshot sits below the paste, the checker never sees it and rejects a valid report.

### Steps to Reproduce

From a source checkout, run against the repo's own script.

1. Save this as `repro.py` in the repo root:

```python
import sys; sys.path.insert(0, ".github/scripts")
from check_issue_readiness import evaluate_readiness, extract_sections
```

2. Run: `python3 repro.py`

The reproduction demonstrates the defect. See output below.

### Actual Behavior

Run method: `npm run dev` source checkout — the defect is in the repo's own CI scripts, reproduced directly against them with the snippet above.

`extract_sections()` reports `acceptance criteria` as a parsed section when that heading exists only inside a fenced block, and drops the tail of `### Actual Behavior` when a fence inside it contains a `###` line. Neither script skips fenced regions:

```
check_issue_readiness.py:42   HEADING_RE = re.compile(r"(?m)^###\\s+(.+?)\\s*$")
check_issue_readiness.py:110  matches = list(HEADING_RE.finditer(body))
```

There is no fence handling anywhere in either parser, and none of the 26 tests in `.github/scripts/tests/test_issue_readiness.py` exercise a fenced code block.

Output of the reproduction:

<img width="3600" height="1100" alt="Image" src="https://github.com/user-attachments/assets/25b0887b-42bc-4452-b74b-2799247033af" />

### Expected Behavior

Headings inside fenced code blocks are not treated as section boundaries, so quoting the template does not manufacture a section, and pasting a log into Actual Behavior does not truncate it.

### Relevant Logs

```
.github/scripts/check_issue_readiness.py
  42:  HEADING_RE = re.compile(r"(?m)^###\\s+(.+?)\\s*$")
 110:  matches = list(HEADING_RE.finditer(body))

.github/scripts/check_pr_description.py
  70:  HEADING_RE = re.compile(r"(?m)^##\\s+(.+?)\\s*$")

Sibling script, same root cause — "## How to Test" exists only inside a fence
yet satisfies the required-section check:

  >>> import check_pr_description as p
  >>> list(p.extract_sections(body).keys())
  ['Why', 'Summary', 'How to Test', 'Issue Number']
```

### Acceptance Criteria

- [ ] A `###` heading inside a fenced code block is not parsed as a section by `check_issue_readiness.py`.
- [ ] An issue whose only Acceptance Criteria heading is inside a fence is not granted `ready-for-dev`.
- [ ] A bug report whose Actual Behavior contains a fenced block with a `###` line is still evaluated against the full section, including a screenshot placed after the fence.
- [ ] The same fence handling applies to `##` headings in `check_pr_description.py`.
- [ ] Tests in `.github/scripts/tests/` cover fenced headings in both directions — a spurious section and a truncated one.

### Screenshots

_No response_

### Additional Context

This issue body demonstrates the defect on itself. Running `extract_sections()` over the draft of it yields two sections that do not exist — `notes` and `error detail` — both lifted from inside the fenced snippets above.

It still evaluates correctly here only because the real `### Acceptance Criteria` appears later than the fenced one and overwrites it in the section dict. Reorder the body and it would not.

Related but distinct: #16513 covers the workflow failing on not-ready issues via the script's exit code. This is a parsing defect in the same file and is independent of that.

Both scripts arrived recently in #16449, so this is unlikely to have affected many issues yet — but quoting the issue template inside a fence is a natural thing for a reporter to do, and pasting logs into Actual Behavior is exactly what the template asks for.
"""
    result = evaluate_readiness(body, ["bug"])
    assert result.ready, f"Should be ready: {result.reasons}"
