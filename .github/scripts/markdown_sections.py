"""Helpers for parsing Markdown sections in GitHub issue and PR bodies."""

from dataclasses import dataclass
import re

FENCE_LINE_RE = re.compile(r"^[ ]{0,3}(?P<marker>`{3,}|~{3,})(?P<rest>[^\r\n]*)")


def _mask_line(line: str) -> str:
    """Replace line content while preserving offsets and line endings."""
    return "".join(char if char in "\r\n" else " " for char in line)


@dataclass(frozen=True)
class _Fence:
    """The marker used to open a fenced Markdown code block."""

    char: str
    length: int

    @classmethod
    def opened_by(cls, line: str) -> "_Fence | None":
        """Return the fence opened by ``line``, if any."""
        match = FENCE_LINE_RE.match(line)
        if match is None:
            return None
        marker = match.group("marker")
        return cls(char=marker[0], length=len(marker))

    def closed_by(self, line: str) -> bool:
        """Return whether ``line`` closes this fence."""
        match = FENCE_LINE_RE.match(line)
        if match is None:
            return False
        marker = match.group("marker")
        return (
            marker[0] == self.char
            and len(marker) >= self.length
            and not match.group("rest").strip()
        )


def without_fenced_code_blocks(body: str) -> str:
    """Mask fenced code blocks without changing character offsets."""
    masked_lines: list[str] = []
    fence: _Fence | None = None

    for line in body.splitlines(keepends=True):
        if fence is None:
            fence = _Fence.opened_by(line)
            masked_lines.append(_mask_line(line) if fence else line)
        else:
            masked_lines.append(_mask_line(line))
            if fence.closed_by(line):
                fence = None

    return "".join(masked_lines)


def find_headings(body: str, heading_re: re.Pattern[str]) -> list[re.Match[str]]:
    """Return heading matches outside fenced code blocks."""
    return list(heading_re.finditer(without_fenced_code_blocks(body)))
