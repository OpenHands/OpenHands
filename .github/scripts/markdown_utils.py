"""Helpers for parsing Markdown in GitHub automation scripts."""

from __future__ import annotations

import re

# A fenced code-block opener follows CommonMark's grammar:
#   - 0-3 leading SPACES (not tabs — a tab indent is not a fence per CommonMark).
#   - 3+ backticks OR 3+ tildes (the fence character).
#   - An optional info string: for backtick fences the info string may not
#     contain any backticks; for tilde fences it may contain anything.
# The opener is matched line-by-line; `find_fenced_regions` below walks the
# input and tracks state, so the regexes are intentionally simple.
#
# Reference: https://spec.commonmark.org/0.31.2/#fenced-code-blocks
_FENCE_OPENER_BACKTICK_RE = re.compile(
    r"^(?P<indent>[ ]{0,3})(?P<fence>`{3,})(?P<info>[^`\n]*)$"
)
_FENCE_OPENER_TILDE_RE = re.compile(
    r"^(?P<indent>[ ]{0,3})(?P<fence>~{3,})(?P<info>[^\n]*)$"
)
# A closing fence: same indent constraint, same fence character, at least as
# long as the opener, with only spaces (and the newline) after the marker.
# Note: braces in the regex are doubled below so ``str.format`` does not
# interpret them as substitution fields.
_FENCE_CLOSER_RE_TEMPLATE = (
    r"^(?P<indent>[ ]{{0,3}}){fence}{{{minlen},}}[ \t]*(?:\n|$)"
)


def _try_open_fence(line: str) -> tuple[str, int] | None:
    """Return ``(fence_char, fence_len)`` if `line` opens a fence, else None.

    The opener grammar is type-specific: backtick fences may not have
    backticks in the info string, while tilde fences have no such restriction.
    """
    match = _FENCE_OPENER_TILDE_RE.match(line)
    if match is not None:
        return match.group("fence")[0], len(match.group("fence"))
    match = _FENCE_OPENER_BACKTICK_RE.match(line)
    if match is not None:
        return match.group("fence")[0], len(match.group("fence"))
    return None


def _try_close_fence(line: str, fence_char: str, fence_len: int) -> bool:
    """Return True if `line` closes the fence opened with `fence_char`/`fence_len`.

    A closing fence must use at most 3 leading spaces, the same fence
    character, a length at least that of the opener, and only spaces after
    the marker (no other text, per CommonMark).
    """
    pattern = _FENCE_CLOSER_RE_TEMPLATE.format(
        fence=re.escape(fence_char), minlen=fence_len
    )
    return re.match(pattern, line) is not None


def find_fenced_regions(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) character ranges of fenced code blocks in `text`.

    Supports both backtick and tilde fences, mixed-length openers/closers, and
    fences that run to the end of the body without a closer. The parser follows
    CommonMark's grammar closely enough for our use case (GitHub renders both
    flavors identically), so quoted templates and pasted logs do not produce
    phantom or truncated sections.
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
            opener = _try_open_fence(line)
            if opener is not None:
                fence_char, fence_len = opener
                start = offset
                inside = True
        else:
            if _try_close_fence(line, fence_char, fence_len):
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


def strip_fenced_regions(text: str) -> str:
    """Return `text` with fenced code blocks removed.

    Used by readiness helpers so quoted template snippets inside fences cannot
    satisfy run-method, screenshot, or checklist requirements.
    """
    regions = find_fenced_regions(text)
    if not regions:
        return text

    parts: list[str] = []
    last = 0
    for start, end in regions:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts)
