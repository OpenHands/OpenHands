"""Utility functions for truncating text content."""

import hashlib
from pathlib import Path

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Default truncation limits
DEFAULT_TEXT_CONTENT_LIMIT = 50_000

# Default truncation notice
DEFAULT_TRUNCATE_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of the full "
    "response has been shown to you.</NOTE>"
)  # 113 chars

DEFAULT_TRUNCATE_NOTICE_WITH_PERSIST = (
    "<response clipped><NOTE>Due to the max output limit, only part of the full "
    "response has been shown to you. The complete output has been saved to "
    "{file_path} - you can use other tools to view the full content (truncated "
    "part starts around line {line_num}).</NOTE>"
)


def _save_full_content(content: str, save_dir: str, tool_prefix: str) -> str | None:
    """Save full content to the specified directory and return the file path."""

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate hash-based filename for deduplication
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    filename = f"{tool_prefix}_output_{content_hash}.txt"
    file_path = save_dir_path / filename

    # Only write if file doesn't exist (deduplication)
    if not file_path.exists():
        try:
            file_path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to save full content to {file_path}: {e}")
            return None

    return str(file_path)


def maybe_truncate(
    content: str,
    truncate_after: int | None = None,
    truncate_notice: str = DEFAULT_TRUNCATE_NOTICE,
    save_dir: str | None = None,
    tool_prefix: str = "output",
) -> str:
    """
    Truncate the middle of content if it exceeds the specified length.

    Keeps the head and tail of the content to preserve context at both ends.
    Optionally saves the full content to a file for later investigation.

    Args:
        content: The text content to potentially truncate
        truncate_after: Maximum length before truncation. If None, no truncation occurs
        truncate_notice: Notice to insert in the middle when content is truncated
        save_dir: Working directory to save full content file in
        tool_prefix: Prefix for the saved file (e.g., "bash", "browser", "editor")

    Returns:
        Original content if under limit, or truncated content with head and tail
        preserved and reference to saved file if applicable
    """
    # 1) Early exits: no truncation requested, or content already within limit
    if not truncate_after or len(content) <= truncate_after or truncate_after < 0:
        return content

    # 2) If even the base notice doesn't fit, return a slice of it
    if len(truncate_notice) >= truncate_after:
        return truncate_notice[:truncate_after]

    # 3) Calculate proposed head size based on base notice
    # (for consistent line number calc)
    available_chars = truncate_after - len(truncate_notice)
    # Prefer giving the "extra" char to head (ceil split)
    proposed_head = available_chars // 2 + (available_chars % 2)

    # 4) Optionally save full content, then construct the final notice
    final_notice = truncate_notice
    if save_dir:
        saved_file_path = _save_full_content(content, save_dir, tool_prefix)
        if saved_file_path:
            # Calculate line number where truncation happens (using head_chars)
            head_content_lines = len(content[:proposed_head].splitlines())

            final_notice = DEFAULT_TRUNCATE_NOTICE_WITH_PERSIST.format(
                file_path=saved_file_path,
                line_num=head_content_lines + 1,  # +1 to indicate next line
            )

    # 5) If the final notice (with persist info) alone fills the
    # budget, return a slice of it
    if len(final_notice) >= truncate_after:
        return final_notice[:truncate_after]

    # 6) Allocate remaining budget to head/tail
    remaining = truncate_after - len(final_notice)
    head_chars = min(
        proposed_head, remaining
    )  # Ensure head_chars doesn't exceed remaining
    tail_chars = remaining - head_chars  # non-negative due to previous checks

    return (
        content[:head_chars]
        + final_notice
        + (content[-tail_chars:] if tail_chars > 0 else "")
    )
