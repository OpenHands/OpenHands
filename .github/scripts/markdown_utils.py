"""Helpers for parsing Markdown in GitHub automation scripts."""

from __future__ import annotations

import re

# A fenced code-block opener: up to 3 leading spaces, then 3+ backticks or tildes,
# then an optional info string. GitHub/CommonMark only recognises fences with up to
# 3 leading spaces, so stricter indent avoids false positives from indented text.
_FENCE_START_RE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


def find_fenced_regions(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) character ranges of fenced code blocks in `text`.

    Supports both triple-backtick and triple-tilde fences, and fenced blocks that
    run to the end of the body without a closing fence.
    """
    regions: list[tuple[int, int]] = []
    inside = False
    fence_char = ""
    fence_len = 0
    start = 0
    lines = text.splitlines(keepends=True)
    offset = 0

    for line in lines:
        if not inside:
            m = _FENCE_START_RE.match(line)
            if m:
                fence_char = m.group("fence")[0]
                fence_len = len(m.group("fence"))
                start = offset
                inside = True
        else:
            # The closing fence must use the same character and be at least as long.
            pattern = r"^\s{0,3}" + re.escape(fence_char) + "{" + str(fence_len) + r",}\s*$"
            if re.match(pattern, line):
                end = offset + len(line)
                regions.append((start, end))
                inside = False
        offset += len(line)

    # An unclosed fence extends to the end of the body.
    if inside:
        regions.append((start, offset))

    return regions


def is_inside_fenced_region(pos: int, regions: list[tuple[int, int]]) -> bool:
    """Return True if `pos` falls inside one of the fenced regions."""
    return any(start <= pos < end for start, end in regions)
