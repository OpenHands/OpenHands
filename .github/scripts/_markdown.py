"""Markdown helpers shared by the issue-readiness and PR-description gates.

Both gates decide whether a body satisfies a criterion by pattern-matching its
text (does it name a run method? does it embed a screenshot? does it have a
checklist item?). Those patterns must only ever see text the author is
*asserting*, never text they are *quoting* — otherwise a pasted template
example satisfies the gate while carrying no real information.

Fenced code blocks are the quoting construct that matters here: GitHub renders
them verbatim, so their contents are a display artifact rather than a claim.
Inline code spans are deliberately left alone — "I ran `npm run dev`" is a real
reproduction step, not a quoted example.
"""

from __future__ import annotations

import re

# A fenced code block, per CommonMark:
#   - the opening fence is 3+ backticks or tildes, indented at most 3 spaces,
#     optionally followed by an info string (```python);
#   - the closing fence must use the same character and be at least as long,
#     which `(?P=fence)` approximates by requiring an exact repeat — so a ```
#     inside a ~~~~ block does not close it;
#   - an unclosed fence runs to the end of the input.
CODE_FENCE_RE = re.compile(
    r"(?m)^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*"  # opening fence + info string
    r"(?:\n[\s\S]*?)?"  # body (lazy, may be empty)
    r"(?:\n[ \t]{0,3}(?P=fence)[ \t]*(?=\n|\Z)|\Z)"  # closing fence, or EOF
)


def strip_code_fences(text: str) -> str:
    """Return `text` with fenced code blocks removed.

    Substituting a newline rather than an empty string keeps surrounding lines
    separated, so line-anchored patterns (`(?m)^- [ ]`) cannot match text that
    only became line-initial because a fence was deleted between them.
    """
    return CODE_FENCE_RE.sub("\n", text)
