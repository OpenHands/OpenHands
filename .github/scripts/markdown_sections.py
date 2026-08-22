"""Helpers for parsing Markdown sections in GitHub issue and PR bodies."""

import re
from dataclasses import dataclass

# CommonMark allows an opening fence to carry leading indentation, and inside a
# list item that indentation is measured from the list's content column rather
# than the left margin. Rather than parse lists, accept any indentation on the
# opening fence and bound the closing fence relative to it (below).
FENCE_LINE_RE = re.compile(
    r"^(?P<indent>[ ]*)(?P<marker>`{3,}|~{3,})(?P<rest>[^\r\n]*)"
)

# CommonMark lets a closing fence sit up to three spaces further in than the
# fence it closes; beyond that it is content, not a terminator.
MAX_CLOSING_INDENT_OFFSET = 3


def _mask_line(line: str) -> str:
    """Replace line content while preserving offsets and line endings."""
    return "".join(char if char in "\r\n" else " " for char in line)


@dataclass(frozen=True)
class _Fence:
    """The marker used to open a fenced Markdown code block."""

    char: str
    length: int
    indent: int

    @classmethod
    def opened_by(cls, line: str) -> "_Fence | None":
        """Return the fence opened by ``line``, if any."""
        match = FENCE_LINE_RE.match(line)
        if match is None:
            return None
        marker = match.group("marker")
        return cls(
            char=marker[0],
            length=len(marker),
            indent=len(match.group("indent")),
        )

    def closed_by(self, line: str) -> bool:
        """Return whether ``line`` closes this fence."""
        match = FENCE_LINE_RE.match(line)
        if match is None:
            return False
        marker = match.group("marker")
        return (
            marker[0] == self.char
            and len(marker) >= self.length
            and len(match.group("indent")) <= self.indent + MAX_CLOSING_INDENT_OFFSET
            and not match.group("rest").strip()
        )


def without_fenced_code_blocks(body: str) -> str:
    """Mask fenced code blocks without changing character offsets.

    A fence that is never closed is left untouched. CommonMark would run it to
    the end of the document, but these parsers gate contributions: one stray
    marker in a pasted log would hide every heading after it and reject a report
    whose sections are plainly there. Masking only balanced fences keeps the
    failure on the side of accepting.
    """
    masked_lines: list[str] = []
    fenced_lines: list[str] = []
    fence: _Fence | None = None

    for line in body.splitlines(keepends=True):
        if fence is None:
            fence = _Fence.opened_by(line)
            if fence is None:
                masked_lines.append(line)
            else:
                fenced_lines.append(line)
        else:
            fenced_lines.append(line)
            if fence.closed_by(line):
                masked_lines.extend(_mask_line(text) for text in fenced_lines)
                fenced_lines.clear()
                fence = None

    # Whatever is left belongs to an unclosed fence, so it stays as written.
    masked_lines.extend(fenced_lines)

    return "".join(masked_lines)


def find_headings(body: str, heading_re: re.Pattern[str]) -> list[re.Match[str]]:
    """Return heading matches outside fenced code blocks."""
    return list(heading_re.finditer(without_fenced_code_blocks(body)))
