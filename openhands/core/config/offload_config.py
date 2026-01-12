"""Configuration for context offloading.

Context offloading saves large tool outputs to the filesystem and returns
references with previews, enabling lossless context management while reducing
token usage.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OffloadConfig(BaseModel):
    """Configuration for context offloading.

    Context offloading saves large tool outputs to filesystem and returns
    references with previews, preserving full content while reducing context
    window usage.

    Attributes:
        enabled: Whether context offloading is enabled.
        max_output_chars: Maximum output size (in chars) before offloading.
            Should be less than max_message_chars to ensure offload happens
            before truncation.
        offload_dir: Directory for offloaded files, relative to workspace.
        preview_head_lines: Number of lines from the start to include in preview.
        preview_tail_lines: Number of lines from the end to include in preview.
        preview_max_line_chars: Maximum characters per line in preview.
        cleanup_on_session_end: Whether to remove offloaded files when session ends.
        retention_hours: Hours to retain offloaded files if cleanup is disabled.
        max_total_size_mb: Maximum total size of offloaded files (0 = unlimited).
        offload_browser_dom: Whether to offload browser DOM objects.
        offload_browser_axtree: Whether to offload browser accessibility tree.
        browser_screenshot_thumbnail_width: Width of thumbnail for screenshots.
    """

    enabled: bool = Field(
        default=False,
        description='Whether context offloading is enabled (opt-in).',
    )

    max_output_chars: int = Field(
        default=25000,
        description='Offload outputs exceeding this size (in chars). '
        'Should be <= max_message_chars (30000) to ensure offload happens before truncation.',
        ge=1000,
    )

    offload_dir: str = Field(
        default='.openhands/context_offload',
        description='Directory for offloaded files, relative to workspace.',
    )

    preview_head_lines: int = Field(
        default=15,
        description='Number of lines from the start to include in preview.',
        ge=1,
    )

    preview_tail_lines: int = Field(
        default=5,
        description='Number of lines from the end to include in preview.',
        ge=0,
    )

    preview_max_line_chars: int = Field(
        default=200,
        description='Maximum characters per line in preview.',
        ge=50,
    )

    cleanup_on_session_end: bool = Field(
        default=True,
        description='Remove offloaded files when session ends.',
    )

    retention_hours: int = Field(
        default=24,
        description='Hours to retain offloaded files if cleanup is disabled.',
        ge=1,
    )

    max_total_size_mb: int = Field(
        default=500,
        description='Maximum total size of offloaded files in MB (0 = unlimited).',
        ge=0,
    )

    offload_browser_dom: bool = Field(
        default=True,
        description='Whether to offload browser DOM objects to files.',
    )

    offload_browser_axtree: bool = Field(
        default=True,
        description='Whether to offload browser accessibility tree to files.',
    )

    browser_screenshot_thumbnail_width: int = Field(
        default=400,
        description='Width of thumbnail for browser screenshots (height auto-scaled).',
        ge=100,
    )

    model_config = ConfigDict(extra='forbid')


def offload_config_from_toml_section(data: dict) -> dict[str, OffloadConfig]:
    """Create an OffloadConfig instance from a toml dictionary.

    Args:
        data: The TOML dictionary representing the [offload] section.

    Returns:
        dict[str, OffloadConfig]: A mapping where 'offload' is the key.
    """
    try:
        config = OffloadConfig(**data)
    except Exception as e:
        from openhands.core.logger import openhands_logger as logger

        logger.warning(f'Invalid offload configuration: {e}. Using defaults.')
        config = OffloadConfig()

    return {'offload': config}
