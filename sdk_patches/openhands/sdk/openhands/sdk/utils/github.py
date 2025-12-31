"""Utility functions for GitHub integrations."""

import re


# Zero-width joiner character (U+200D)
# We use ZWJ instead of ZWSP (U+200B) because:
# - ZWJ is semantically more appropriate (joins characters without adding space)
# - ZWJ has better support in modern renderers
# - ZWJ is invisible and doesn't affect text rendering or selection
ZWJ = "\u200d"


def sanitize_openhands_mentions(text: str) -> str:
    """Sanitize @OpenHands mentions in text to prevent self-mention loops.

    This function inserts a zero-width joiner (ZWJ) after the @ symbol in
    @OpenHands mentions, making them non-clickable in GitHub comments while
    preserving readability. The original case of the mention is preserved.

    Args:
        text: The text to sanitize

    Returns:
        Text with sanitized @OpenHands mentions (e.g., "@OpenHands" -> "@‍OpenHands")

    Examples:
        >>> sanitize_openhands_mentions("Thanks @OpenHands for the help!")
        'Thanks @\\u200dOpenHands for the help!'
        >>> sanitize_openhands_mentions("Check @openhands and @OPENHANDS")
        'Check @\\u200dopenhands and @\\u200dOPENHANDS'
        >>> sanitize_openhands_mentions("No mention here")
        'No mention here'
    """
    # Pattern to match @OpenHands mentions at word boundaries
    # Uses re.IGNORECASE so we don't need [Oo]pen[Hh]ands
    # Capture group preserves the original case
    pattern = r"@(OpenHands)\b"

    # Replace @ with @ + ZWJ while preserving the original case
    # The \1 backreference preserves the matched case
    sanitized = re.sub(pattern, f"@{ZWJ}\\1", text, flags=re.IGNORECASE)

    return sanitized
