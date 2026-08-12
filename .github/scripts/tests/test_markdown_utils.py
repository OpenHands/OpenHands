"""Tests for markdown_utils.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_utils import find_fenced_regions, is_inside_fenced_region


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
    text = "#### Heading\n````\n### Not a heading\n```\n### After"
    regions = find_fenced_regions(text)
    assert len(regions) == 1
    assert regions[0][1] == len(text)


def test_is_inside_fenced_region():
    text = "before\n```\ninside\n```\nafter"
    regions = find_fenced_regions(text)
    assert is_inside_fenced_region(text.find("inside"), regions)
    assert not is_inside_fenced_region(text.find("before"), regions)
    assert not is_inside_fenced_region(text.find("after"), regions)
