"""Tests for markdown_utils.py — fenced code-block detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_utils import (
    find_fenced_regions,
    is_inside_fenced_region,
    strip_fenced_regions,
)


# ---------------------------------------------------------------------------
# find_fenced_regions
# ---------------------------------------------------------------------------


def test_find_fenced_regions_backticks():
    text = "### Heading\n```\n### Not a heading\n```\n### After"
    regions = find_fenced_regions(text)
    assert len(regions) == 1
    start, end = regions[0]
    assert text[start:end] == "```\n### Not a heading\n```\n"


def test_find_fenced_regions_tildes():
    text = "### Heading\n~~~python\n### Not a heading\n~~~\n### After"
    regions = find_fenced_regions(text)
    assert len(regions) == 1
    start, end = regions[0]
    assert text[start:end] == "~~~python\n### Not a heading\n~~~\n"


def test_find_fenced_regions_multiple_blocks():
    text = "```\nfirst\n```\n\n```\nsecond\n```"
    regions = find_fenced_regions(text)
    assert len(regions) == 2


def test_find_fenced_regions_unclosed():
    text = "### Heading\n```\n### Not a heading"
    regions = find_fenced_regions(text)
    assert len(regions) == 1
    assert regions[0][1] == len(text)


def test_find_fenced_regions_length_mismatch():
    # A 4-backtick fence needs at least 4 backticks to close; 3 should not close it.
    text = "````\n### Not a heading\n```\n### After"
    regions = find_fenced_regions(text)
    assert len(regions) == 1
    assert regions[0][1] == len(text)


# ---------------------------------------------------------------------------
# Bug cases identified by code review of an earlier revision. Each input
# follows CommonMark's grammar: a tab-indented opener is *not* a fence, a
# backtick-fence info string cannot contain backticks, and a closing marker
# must be followed by only whitespace.
# ---------------------------------------------------------------------------


def test_tab_indented_opener_is_not_a_fence():
    # A tab-indented ``` is not a fence per CommonMark. The text should fall
    # back to whole-body checks, not be hidden inside a phantom fence.
    text = "\t```python\n### real heading\n```\n"
    regions = find_fenced_regions(text)
    # The tab line is not a fence. The ``` after "### real heading" is a new
    # fence that runs to EOF (it has no closer).
    assert len(regions) == 1
    start, _ = regions[0]
    # The fence must not start at the tab line (offset 0).
    assert start > 0
    # The real heading is *outside* the fence.
    assert not is_inside_fenced_region(text.find("### real heading"), regions)


def test_backtick_fence_info_string_cannot_contain_backticks():
    # `` ````code``` `` is not a valid 4-backtick fence because the info
    # string after the 4 backticks contains more backticks. CommonMark says
    # the info string for a backtick fence cannot contain backticks.
    text = "````code```\n### real heading\n"
    regions = find_fenced_regions(text)
    # The opener is invalid — no fence should start. The real heading must be
    # visible (not inside a fence).
    assert regions == [] or not is_inside_fenced_region(
        text.find("### real heading"), regions
    )


def test_closing_fence_with_trailing_text_is_not_a_close():
    # A closer must be followed by only whitespace. `` ``` garbage`` is not a
    # valid closer; the fence must stay open until a real close (or EOF).
    text = "```python\n### inside fence\n``` garbage\n### outside\n"
    regions = find_fenced_regions(text)
    assert len(regions) == 1
    # The fence runs to EOF because the trailing-text "closer" is invalid.
    assert regions[0][1] == len(text)
    # The "outside" heading is inside the still-open fence.
    assert is_inside_fenced_region(text.find("### outside"), regions)


# ---------------------------------------------------------------------------
# is_inside_fenced_region
# ---------------------------------------------------------------------------


def test_is_inside_fenced_region():
    text = "before\n```\ninside\n```\nafter"
    regions = find_fenced_regions(text)
    assert is_inside_fenced_region(text.find("inside"), regions)
    assert not is_inside_fenced_region(text.find("before"), regions)
    assert not is_inside_fenced_region(text.find("after"), regions)


# ---------------------------------------------------------------------------
# strip_fenced_regions
# ---------------------------------------------------------------------------


def test_strip_fenced_regions_removes_block():
    text = "before\n```\nnpm run dev\n![shot](https://example.com/x.png)\n```\nafter"
    stripped = strip_fenced_regions(text)
    assert "npm run dev" not in stripped
    assert "![shot]" not in stripped
    assert "before" in stripped
    assert "after" in stripped


def test_strip_fenced_regions_preserves_real_content():
    # Real run-method + screenshot live outside the fence; quoted template
    # inside the fence should not be counted.
    text = (
        "Real reproduction with npm run dev.\n\n"
        "```\nExample only: agent-canvas\n"
        "![example](https://github.com/user-attachments/assets/example)\n```\n\n"
        "![real](https://github.com/user-attachments/assets/real123)\n"
    )
    stripped = strip_fenced_regions(text)
    assert "npm run dev" in stripped
    assert "real123" in stripped
    assert "agent-canvas" not in stripped
    assert "example" not in stripped or "real123" in stripped
