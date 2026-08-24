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
    main,
    BUG_LABEL,
    ENHANCEMENT_LABEL,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

BUG_BODY_READY = """### Steps to Reproduce
Run `npm run dev` and click the button.

### Actual Behavior
The button was misaligned.

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria
- [ ] Button is centered
"""

BUG_BODY_NO_RUN_METHOD = """### Steps to Reproduce
Click the button.

### Actual Behavior
The button was misaligned.

![screenshot](https://github.com/user-attachments/assets/abc123)

"""

BUG_BODY_NO_SCREENSHOT = """### Steps to Reproduce
I ran `npm run dev` and clicked the button.

### Actual Behavior
The button was misaligned.

"""

BUG_BODY_NO_ACCEPTANCE = """### Steps to Reproduce
I ran `npm run dev` and clicked the button.

### Actual Behavior
The button was misaligned.

![screenshot](https://github.com/user-attachments/assets/abc123)
"""

BUG_BODY_EMPTY_ACTUAL = """### Steps to Reproduce
I ran `npm run dev` and clicked the button.

### Actual Behavior
_No response_

### Acceptance Criteria
- [ ] Button is centered
"""

BUG_BODY_AGENT_CANVAS = """### Steps to Reproduce
I used agent-canvas to reproduce this.

### Actual Behavior
The button was misaligned.

![screenshot](https://example.com/screenshot.png)

### Acceptance Criteria
- [ ] Fixed
"""

BUG_BODY_HOSTED_URL = """### Steps to Reproduce
Reproduced on app.all-hands.dev/canvas.

### Actual Behavior
The button was misaligned.

<video src="https://example.com/bug.mp4"></video>

### Acceptance Criteria
- [ ] Fixed
"""

BUG_BODY_MISSING_REPRODUCTION = """### Actual Behavior
The button was misaligned.

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria
- [ ] Button is centered
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



def test_bug_not_ready_missing_reproduction():
    result = evaluate_readiness(BUG_BODY_MISSING_REPRODUCTION, [BUG_LABEL])
    assert not result.ready
    assert any("Steps to Reproduce" in r for r in result.reasons)

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

def test_extract_sections_ignores_heading_inside_fence():
    body = """### Notes
The template says:

```markdown
### Acceptance Criteria
- [ ] Add criteria here
```
"""
    sections = extract_sections(body)
    assert set(sections) == {"notes"}

def test_fenced_heading_does_not_truncate_actual_behavior():
    body = """### Steps to Reproduce
Run `npm run dev` and reproduce the error.

### Actual Behavior
I ran `npm run dev` and saw:

~~~text
### Error detail
something went wrong
~~~

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria
- [ ] The bug is fixed
"""
    sections = extract_sections(body)
    assert "error detail" not in sections
    assert "user-attachments" in sections["actual behavior"]
    assert evaluate_readiness(body, [BUG_LABEL]).ready


def test_unclosed_fence_does_not_swallow_later_sections():
    """One stray marker in a log paste must not reject an otherwise-ready report."""
    body = """### Steps to Reproduce
Run `npm run dev` and reproduce the crash.

### Relevant Logs
```shell
Traceback (most recent call last):
  the paste was cut off before the closing fence

### Actual Behavior
I ran `npm run dev` and saw the crash above.

![screenshot](https://github.com/user-attachments/assets/abc123)

### Acceptance Criteria
- [ ] The bug is fixed
"""
    sections = extract_sections(body)
    assert {"relevant logs", "actual behavior", "acceptance criteria"} <= set(sections)
    assert evaluate_readiness(body, [BUG_LABEL]).ready

def test_main_json_not_ready(tmp_path, capsys, monkeypatch):
    import json
    body_file = tmp_path / "issue.md"
    body_file.write_text(BUG_BODY_NO_SCREENSHOT)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_issue_readiness.py",
            "--body-file",
            str(body_file),
            "--labels",
            "bug",
            "--json",
        ],
    )
    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ready"] is False
    assert len(data["reasons"]) > 0


def test_main_json_ready(tmp_path, capsys, monkeypatch):
    import json

    body_file = tmp_path / "issue.md"
    body_file.write_text(BUG_BODY_READY)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_issue_readiness.py",
            "--body-file",
            str(body_file),
            "--labels",
            "bug",
            "--json",
        ],
    )
    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ready"] is True
    assert len(data["reasons"]) == 0


def test_main_text_ready(tmp_path, capsys, monkeypatch):
    body_file = tmp_path / "issue.md"
    body_file.write_text(BUG_BODY_READY)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_issue_readiness.py",
            "--body-file",
            str(body_file),
            "--labels",
            "bug",
        ],
    )
    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Issue meets ready-for-dev criteria." in captured.out


def test_main_text_not_ready(tmp_path, capsys, monkeypatch):
    body_file = tmp_path / "issue.md"
    body_file.write_text(BUG_BODY_NO_SCREENSHOT)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_issue_readiness.py",
            "--body-file",
            str(body_file),
            "--labels",
            "bug",
        ],
    )
    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Issue does not meet ready-for-dev criteria:" in captured.out


def test_main_event_path_json_ready(tmp_path, capsys, monkeypatch):
    import json

    event_file = tmp_path / "event.json"
    event_file.write_text(
        json.dumps(
            {
                "issue": {
                    "body": BUG_BODY_READY,
                    "labels": [{"name": "bug"}],
                }
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_issue_readiness.py",
            "--event-path",
            str(event_file),
            "--json",
        ],
    )
    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ready"] is True
    assert len(data["reasons"]) == 0


