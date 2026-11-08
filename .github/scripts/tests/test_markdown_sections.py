"""Tests for markdown_sections.py — the fence-aware heading scanner."""

import re
import sys
from pathlib import Path

# Make the sibling script importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_sections import find_headings, without_fenced_code_blocks

H3_RE = re.compile(r"(?m)^###\s+(.+?)\s*$")


def headings(body: str) -> list[str]:
    return [match.group(1) for match in find_headings(body, H3_RE)]


# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------


def test_masking_preserves_offsets():
    body = "### One\n```py\n### Two\n```\n### Three\n"
    masked = without_fenced_code_blocks(body)
    assert len(masked) == len(body)
    assert masked.splitlines() == [
        "### One",
        "     ",
        "       ",
        "   ",
        "### Three",
    ]


def test_masking_preserves_crlf_offsets():
    body = "### One\r\n```py\r\n### Two\r\n```\r\n### Three\r\n"
    masked = without_fenced_code_blocks(body)
    assert len(masked) == len(body)
    assert headings(body) == ["One", "Three"]


# ---------------------------------------------------------------------------
# Unclosed fences
# ---------------------------------------------------------------------------


def test_unclosed_fence_does_not_hide_later_headings():
    """A stray marker in a pasted log must not swallow the rest of the body."""
    body = """### Relevant Logs
```shell
Traceback (most recent call last):
  the paste was cut off here

### Actual Behavior
It broke.
"""
    assert headings(body) == ["Relevant Logs", "Actual Behavior"]


def test_unclosed_fence_leaves_text_untouched():
    body = "### One\n~~~\nstill here\n"
    assert without_fenced_code_blocks(body) == body


def test_second_fence_reopens_after_a_balanced_one():
    body = """### One
```
### Masked
```
### Two
```
### Not masked, fence never closes
"""
    assert headings(body) == ["One", "Two", "Not masked, fence never closes"]


# ---------------------------------------------------------------------------
# Fence markers
# ---------------------------------------------------------------------------


def test_tilde_fence_is_not_closed_by_backticks():
    body = "### One\n~~~\n### Masked\n```\n### Also masked\n~~~\n### Two\n"
    assert headings(body) == ["One", "Two"]


def test_longer_fence_is_not_closed_by_a_shorter_one():
    body = "### One\n````\n### Masked\n```\n### Also masked\n````\n### Two\n"
    assert headings(body) == ["One", "Two"]


def test_info_string_does_not_close_a_fence():
    body = "### One\n```\n### Masked\n```py\n### Also masked\n```\n### Two\n"
    assert headings(body) == ["One", "Two"]


# ---------------------------------------------------------------------------
# Indentation
# ---------------------------------------------------------------------------


def test_fence_indented_inside_a_list_item_is_masked():
    """CommonMark measures a nested fence from the list's content column."""
    body = """### Steps to Reproduce

1.  Quote the template:

    ```markdown
### Acceptance Criteria
    - [ ] not a real criterion
    ```

### Actual Behavior
It broke.
"""
    assert headings(body) == ["Steps to Reproduce", "Actual Behavior"]


def test_closing_fence_indented_past_the_limit_does_not_close():
    body = """### One
```text
### Masked
        ```
### Also masked
```
### Two
"""
    assert headings(body) == ["One", "Two"]


def test_closing_fence_may_be_indented_three_spaces():
    body = "### One\n```\n### Masked\n   ```\n### Two\n"
    assert headings(body) == ["One", "Two"]
