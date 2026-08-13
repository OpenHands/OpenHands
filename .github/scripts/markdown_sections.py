"""Helpers for parsing Markdown sections in GitHub issue and PR bodies."""

from __future__ import annotations

import re

FENCE_LINE_RE = re.compile(r"^[ ]{0,3}(?P<marker>`{3,}|~{3,})(?P<rest>[^\r\n]*)")


def _mask_line(line: str) -> str:
    """Replace line content while preserving offsets and line endings."""
    return "".join(char if char in "\r\n" else " " for char in line)


def without_fenced_code_blocks(body: str) -> str:
    """Mask fenced code blocks without changing character offsets."""
    masked_lines: list[str] = []
    fence_char: str | None = None
    fence_length = 0

    for line in body.splitlines(keepends=True):
        fence_match = FENCE_LINE_RE.match(line)

        if fence_char is None:
            if fence_match is None:
                masked_lines.append(line)
                continue

            marker = fence_match.group("marker")
            fence_char = marker[0]
            fence_length = len(marker)
            masked_lines.append(_mask_line(line))
            continue

        masked_lines.append(_mask_line(line))
        if fence_match is None:
            continue

        marker = fence_match.group("marker")
        if (
            marker[0] == fence_char
            and len(marker) >= fence_length
            and not fence_match.group("rest").strip()
        ):
            fence_char = None
            fence_length = 0

    return "".join(masked_lines)


def find_headings(body: str, heading_re: re.Pattern[str]) -> list[re.Match[str]]:
    """Return heading matches outside fenced code blocks."""
    return list(heading_re.finditer(without_fenced_code_blocks(body)))
