"""Context offloading for large tool outputs.

This module provides the ContextOffloader class that saves large tool outputs
to the filesystem and returns references with previews, enabling lossless
context management while reducing token usage.

Example:
    >>> from openhands.memory.offloader import ContextOffloader
    >>> from openhands.core.config.offload_config import OffloadConfig
    >>>
    >>> offloader = ContextOffloader(
    ...     config=OffloadConfig(enabled=True),
    ...     workspace_dir="/workspace",
    ...     session_id="abc123"
    ... )
    >>> if offloader.should_offload(large_output):
    ...     result = offloader.offload_text(large_output, "cmd")
    ...     print(result.preview_message)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger

# Allowed source types for offloading (prevents path traversal via source_type)
ALLOWED_SOURCE_TYPES = frozenset(
    {
        "cmd",
        "ipython",
        "browser_dom",
        "browser_axtree",
        "browser_html",
        "screenshot",
        "mcp",
        "file",
    }
)

# Regex for valid session_id (alphanumeric, hyphens, underscores only)
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

if TYPE_CHECKING:
    from openhands.core.config.offload_config import OffloadConfig


@dataclass
class OffloadResult:
    """Result of an offload operation.

    Attributes:
        file_path: Path to the offloaded file.
        original_size: Original size of the content in characters.
        preview_message: Message containing preview and retrieval instructions.
        offload_type: Type of offloaded content (e.g., "cmd", "browser_dom").
    """

    file_path: str
    original_size: int
    preview_message: str
    offload_type: str


class ContextOffloader:
    """Offloads large content to filesystem, returns references with previews.

    This class is responsible for:
    1. Detecting when content exceeds the configured threshold
    2. Saving full content to the filesystem
    3. Generating preview messages with head/tail content
    4. Tracking offloaded files for cleanup

    Key design principles:
    - Offload happens BEFORE truncation to preserve full content
    - Uses chars (not tokens) to align with existing max_message_chars
    - Provides actionable previews with head + tail
    - Supports cleanup on session end
    """

    OFFLOAD_MESSAGE_TEMPLATE = """[Output offloaded to file - size: {size:,} chars]

📁 Full content saved to: {file_path}

Preview (first {head_lines} lines):
{head_preview}

... [{omitted_lines:,} lines omitted] ...

(last {tail_lines} lines):
{tail_preview}

💡 To access full content:
  • View: cat {file_path}
  • Search: grep "pattern" {file_path}
  • Head: head -n 100 {file_path}
  • Tail: tail -n 100 {file_path}
"""

    OFFLOAD_MESSAGE_TEMPLATE_SHORT = """[Output offloaded - {size:,} chars]
📁 Saved to: {file_path}

Preview:
{preview}

💡 Use `cat {file_path}` to view full content.
"""

    OFFLOAD_JSON_TEMPLATE = """[{type_name} offloaded - {size:,} bytes]
📁 Saved to: {file_path}

Structure preview:
{structure_preview}

💡 View: cat {file_path} | jq '.'
💡 Query: cat {file_path} | jq '.key'
"""

    def __init__(
        self,
        config: OffloadConfig,
        workspace_dir: str,
        session_id: str,
    ) -> None:
        """Initialize the context offloader.

        Args:
            config: Offload configuration.
            workspace_dir: Base workspace directory path.
            session_id: Unique session identifier for organizing files.

        Raises:
            ValueError: If session_id contains invalid characters (path traversal prevention).
        """
        self.config = config
        self._total_size_bytes = 0

        # Sanitize session_id to prevent path traversal
        self.session_id = self._sanitize_session_id(session_id)

        # Sanitize offload_dir to prevent path traversal
        safe_offload_dir = self._sanitize_offload_dir(config.offload_dir)

        # Build full offload directory path
        workspace_path = Path(workspace_dir).resolve()
        self.offload_dir = workspace_path / safe_offload_dir / self.session_id

        # Verify the offload directory is within workspace (final path traversal check)
        if not self._is_path_within_workspace(self.offload_dir, workspace_path):
            raise ValueError(
                f"Offload directory {self.offload_dir} is outside workspace {workspace_path}"
            )

        # Track offloaded files for cleanup
        self._offloaded_files: list[Path] = []

        # Only create directory if offloading is enabled
        if config.enabled:
            # Run retention-based cleanup on startup BEFORE creating directory
            # (if cleanup_on_session_end is disabled)
            if not config.cleanup_on_session_end:
                self._run_retention_cleanup_on_init()

            # Now create the directory
            self.offload_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"ContextOffloader initialized: {self.offload_dir}")

    def _run_retention_cleanup_on_init(self) -> None:
        """Run retention cleanup on initialization.

        This is called during __init__ when cleanup_on_session_end is disabled
        to clean up old files from previous sessions.
        """
        try:
            removed = self.cleanup_expired_files()
            if removed > 0:
                logger.info(f"Startup cleanup removed {removed} expired offload files")
        except Exception as e:
            logger.warning(f"Retention cleanup on init failed: {e}")

    @staticmethod
    def _sanitize_session_id(session_id: str) -> str:
        """Sanitize session_id to prevent path traversal attacks.

        Args:
            session_id: The session identifier to sanitize.

        Returns:
            Sanitized session_id.

        Raises:
            ValueError: If session_id is invalid or empty after sanitization.
        """
        if not session_id:
            raise ValueError("session_id cannot be empty")

        # Remove any path separators and dangerous characters
        sanitized = session_id.replace("/", "").replace("\\", "").replace("..", "")

        # Validate format
        if not SESSION_ID_PATTERN.match(sanitized):
            # Fall back to a hash if the session_id contains invalid characters
            sanitized = hashlib.sha256(session_id.encode()).hexdigest()[:32]
            logger.warning(f"Invalid session_id format, using hash: {sanitized}")

        return sanitized

    @staticmethod
    def _sanitize_offload_dir(offload_dir: str) -> str:
        """Sanitize offload_dir to prevent path traversal attacks.

        Args:
            offload_dir: The offload directory path to sanitize.

        Returns:
            Sanitized offload directory path.
        """
        # Remove leading slashes (absolute path prevention)
        sanitized = offload_dir.lstrip("/")

        # Remove any .. path components
        parts = Path(sanitized).parts
        safe_parts = [p for p in parts if p != ".."]

        if not safe_parts:
            return ".openhands/context_offload"

        return str(Path(*safe_parts))

    @staticmethod
    def _is_path_within_workspace(target: Path, workspace: Path) -> bool:
        """Check if target path is within the workspace directory.

        Args:
            target: The target path to check.
            workspace: The workspace root path.

        Returns:
            True if target is within workspace, False otherwise.
        """
        try:
            target_resolved = target.resolve()
            workspace_resolved = workspace.resolve()
            # Use is_relative_to for cross-platform compatibility (Windows/Unix)
            # This handles both path separators correctly
            return target_resolved.is_relative_to(workspace_resolved)
        except (OSError, ValueError):
            return False

    def _validate_source_type(self, source_type: str) -> str:
        """Validate and sanitize source_type to prevent path traversal.

        Args:
            source_type: The source type to validate.

        Returns:
            Validated source type.

        Raises:
            ValueError: If source_type is not in allowed list.
        """
        if source_type not in ALLOWED_SOURCE_TYPES:
            # Log warning and fall back to generic type
            logger.warning(
                f'Unknown source_type "{source_type}", using "file". '
                f"Allowed types: {ALLOWED_SOURCE_TYPES}"
            )
            return "file"
        return source_type

    @property
    def enabled(self) -> bool:
        """Whether offloading is enabled."""
        return self.config.enabled

    def should_offload(self, content: str) -> bool:
        """Check if content exceeds the offload threshold.

        Args:
            content: The content to check.

        Returns:
            True if content should be offloaded, False otherwise.
        """
        if not self.config.enabled:
            return False
        return len(content) > self.config.max_output_chars

    def should_offload_bytes(self, size_bytes: int) -> bool:
        """Check if binary content exceeds the offload threshold.

        Args:
            size_bytes: Size of content in bytes.

        Returns:
            True if content should be offloaded, False otherwise.
        """
        if not self.config.enabled:
            return False
        # Approximate: 1 char ≈ 1 byte for ASCII, but base64 is ~1.33x
        return size_bytes > self.config.max_output_chars

    def _check_size_limit(self, new_size_bytes: int) -> bool:
        """Check if adding new content would exceed total size limit.

        Args:
            new_size_bytes: Size of new content to add.

        Returns:
            True if within limit, False if would exceed.
        """
        if self.config.max_total_size_mb == 0:
            return True  # No limit
        max_bytes = self.config.max_total_size_mb * 1024 * 1024
        return (self._total_size_bytes + new_size_bytes) <= max_bytes

    def _generate_filename(self, source_type: str, extension: str = "txt") -> Path:
        """Generate a unique filename for offloaded content.

        Args:
            source_type: Type of source (e.g., "cmd", "browser_dom").
            extension: File extension without dot.

        Returns:
            Path to the generated filename.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Add a short random hash for uniqueness
        random_hash = hashlib.md5(f"{timestamp}{source_type}".encode()).hexdigest()[:6]
        filename = f"{source_type}_{timestamp}_{random_hash}.{extension}"
        return self.offload_dir / filename

    def offload_text(
        self,
        content: str,
        source_type: str,
        source_id: str | None = None,
    ) -> OffloadResult:
        """Offload text content to filesystem.

        Args:
            content: The text content to offload.
            source_type: Type of source (e.g., "cmd", "ipython", "mcp").
            source_id: Optional identifier (e.g., command hash).

        Returns:
            OffloadResult with file path and preview message.

        Raises:
            IOError: If file write fails and fallback is needed.
        """
        # Validate source_type to prevent path traversal
        safe_source_type = self._validate_source_type(source_type)
        content_bytes = len(content.encode("utf-8"))

        # Check size limit
        if not self._check_size_limit(content_bytes):
            logger.warning(
                f"Offload size limit reached ({self.config.max_total_size_mb}MB). "
                "Skipping offload."
            )
            raise IOError("Offload size limit exceeded")

        file_path = self._generate_filename(safe_source_type, "txt")

        try:
            file_path.write_text(content, encoding="utf-8")
            self._offloaded_files.append(file_path)
            self._total_size_bytes += content_bytes
        except Exception as e:
            logger.error(f"Failed to write offload file {file_path}: {e}")
            raise IOError(f"Failed to offload: {e}") from e

        # Generate preview message
        preview_message = self._create_text_preview(
            content=content,
            file_path=str(file_path),
        )

        logger.info(
            f"Offloaded {safe_source_type} output: {len(content):,} chars -> {file_path}"
        )

        return OffloadResult(
            file_path=str(file_path),
            original_size=len(content),
            preview_message=preview_message,
            offload_type=safe_source_type,
        )

    def offload_json(
        self,
        data: dict,
        source_type: str,
        type_name: str | None = None,
    ) -> OffloadResult:
        """Offload JSON data (like DOM/AXTree) to filesystem.

        Args:
            data: Dictionary to serialize and offload.
            source_type: Type of source (e.g., "browser_dom", "browser_axtree").
            type_name: Human-readable type name for preview message.

        Returns:
            OffloadResult with file path and preview message.
        """
        # Validate source_type to prevent path traversal
        safe_source_type = self._validate_source_type(source_type)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        content_bytes = len(content.encode("utf-8"))

        if not self._check_size_limit(content_bytes):
            logger.warning("Offload size limit reached. Skipping JSON offload.")
            raise IOError("Offload size limit exceeded")

        file_path = self._generate_filename(safe_source_type, "json")

        try:
            file_path.write_text(content, encoding="utf-8")
            self._offloaded_files.append(file_path)
            self._total_size_bytes += content_bytes
        except Exception as e:
            logger.error(f"Failed to write JSON offload file {file_path}: {e}")
            raise IOError(f"Failed to offload JSON: {e}") from e

        # Generate structure preview
        structure_preview = self._create_json_structure_preview(data)
        preview_message = self.OFFLOAD_JSON_TEMPLATE.format(
            type_name=type_name or safe_source_type,
            size=content_bytes,
            file_path=str(file_path),
            structure_preview=structure_preview,
        )

        logger.info(
            f"Offloaded {safe_source_type} JSON: {content_bytes:,} bytes -> {file_path}"
        )

        return OffloadResult(
            file_path=str(file_path),
            original_size=content_bytes,
            preview_message=preview_message,
            offload_type=safe_source_type,
        )

    def offload_image(
        self,
        base64_data: str,
        source_type: str = "screenshot",
    ) -> tuple[str, str]:
        """Offload image and create thumbnail for LLM vision.

        Args:
            base64_data: Base64-encoded image data (may include data URL prefix).
            source_type: Type of image source.

        Returns:
            Tuple of (file_path, thumbnail_data_url).
            If offload fails, returns ("", original_base64_data).
        """
        # Validate source_type to prevent path traversal
        safe_source_type = self._validate_source_type(source_type)

        try:
            from PIL import Image
            import base64
            import io

            # Remove data URL prefix if present
            if base64_data.startswith("data:"):
                # Format: data:image/png;base64,xxxxx
                base64_data = base64_data.split(",", 1)[1]

            # Decode image
            image_bytes = base64.b64decode(base64_data)

            if not self._check_size_limit(len(image_bytes)):
                logger.warning("Offload size limit reached. Skipping image offload.")
                return "", f"data:image/png;base64,{base64_data}"

            image = Image.open(io.BytesIO(image_bytes))

            # Save full image
            file_path = self._generate_filename(safe_source_type, "png")
            image.save(file_path, "PNG")
            self._offloaded_files.append(file_path)
            self._total_size_bytes += len(image_bytes)

            # Create thumbnail for LLM vision
            thumbnail_width = self.config.browser_screenshot_thumbnail_width
            ratio = thumbnail_width / image.width
            thumbnail_height = int(image.height * ratio)
            thumbnail = image.resize(
                (thumbnail_width, thumbnail_height),
                Image.Resampling.LANCZOS,
            )

            # Convert thumbnail to base64 data URL
            buffer = io.BytesIO()
            thumbnail.save(buffer, format="PNG", optimize=True)
            thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            thumbnail_data_url = f"data:image/png;base64,{thumbnail_base64}"

            logger.info(
                f"Offloaded {safe_source_type}: {len(image_bytes):,} bytes -> {file_path}, "
                f"thumbnail: {len(thumbnail_base64):,} chars"
            )

            return str(file_path), thumbnail_data_url

        except ImportError:
            logger.warning("PIL not available. Skipping image offload.")
            return "", f"data:image/png;base64,{base64_data}"
        except Exception as e:
            logger.error(f"Failed to offload image: {e}")
            return "", f"data:image/png;base64,{base64_data}"

    def _create_text_preview(self, content: str, file_path: str) -> str:
        """Create a preview message with head and tail content.

        Args:
            content: Full text content.
            file_path: Path to the offloaded file.

        Returns:
            Formatted preview message.
        """
        lines = content.splitlines()
        total_lines = len(lines)
        max_chars = self.config.preview_max_line_chars

        head_count = self.config.preview_head_lines
        tail_count = self.config.preview_tail_lines

        # If content is short enough, show all
        if total_lines <= head_count + tail_count:
            preview = "\n".join(
                f"  {i + 1}: {line[:max_chars]}"
                + ("..." if len(line) > max_chars else "")
                for i, line in enumerate(lines)
            )
            return self.OFFLOAD_MESSAGE_TEMPLATE_SHORT.format(
                size=len(content),
                file_path=file_path,
                preview=preview,
            )

        # Head preview
        head_preview = "\n".join(
            f"  {i + 1}: {line[:max_chars]}" + ("..." if len(line) > max_chars else "")
            for i, line in enumerate(lines[:head_count])
        )

        # Tail preview
        tail_start = total_lines - tail_count
        tail_preview = "\n".join(
            f"  {tail_start + i + 1}: {line[:max_chars]}"
            + ("..." if len(line) > max_chars else "")
            for i, line in enumerate(lines[tail_start:])
        )

        omitted = total_lines - head_count - tail_count

        return self.OFFLOAD_MESSAGE_TEMPLATE.format(
            size=len(content),
            file_path=file_path,
            head_lines=head_count,
            head_preview=head_preview,
            omitted_lines=omitted,
            tail_lines=tail_count,
            tail_preview=tail_preview,
        )

    def _create_json_structure_preview(
        self,
        data: dict,
        max_depth: int = 2,
        max_keys: int = 5,
    ) -> str:
        """Create a structural preview of JSON data.

        Args:
            data: Dictionary to preview.
            max_depth: Maximum nesting depth to show.
            max_keys: Maximum keys to show per level.

        Returns:
            Formatted structure preview.
        """

        def summarize(obj: object, depth: int = 0) -> str:
            indent = "  " * depth
            if depth > max_depth:
                return f"{indent}..."

            if isinstance(obj, dict):
                if not obj:
                    return f"{indent}{{}}"
                items = []
                for i, (k, v) in enumerate(obj.items()):
                    if i >= max_keys:
                        items.append(f"{indent}  ... ({len(obj) - max_keys} more keys)")
                        break
                    val_summary = summarize(v, depth + 1).lstrip()
                    items.append(f"{indent}  {k}: {val_summary}")
                return "{\n" + "\n".join(items) + f"\n{indent}}}"

            elif isinstance(obj, list):
                if not obj:
                    return "[]"
                return f"[... {len(obj)} items]"

            else:
                s = str(obj)
                if len(s) > 50:
                    return s[:50] + "..."
                return s

        return summarize(data)

    def cleanup(self) -> int:
        """Clean up all offloaded files for this session.

        Returns:
            Number of files removed.
        """
        if not self.config.cleanup_on_session_end:
            logger.debug("Cleanup disabled. Retaining offloaded files.")
            return 0

        removed = 0
        for file_path in self._offloaded_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove offloaded file {file_path}: {e}")

        # Try to remove empty directory
        try:
            if self.offload_dir.exists() and not any(self.offload_dir.iterdir()):
                self.offload_dir.rmdir()
                # Also try to remove parent session directory if empty
                parent = self.offload_dir.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
        except Exception:
            pass  # Directory not empty or other issue

        logger.info(f"Cleaned up {removed} offloaded files")
        self._offloaded_files.clear()
        self._total_size_bytes = 0
        return removed

    def cleanup_expired_files(self) -> int:
        """Clean up offloaded files older than retention_hours.

        This method cleans up files from ALL sessions in the offload directory
        that exceed the retention period. It's typically called on startup or
        periodically to clean up stale data.

        Returns:
            Number of files removed.
        """
        if self.config.cleanup_on_session_end:
            # If cleanup_on_session_end is enabled, retention is not used
            return 0

        if not self.offload_dir.parent.exists():
            return 0

        removed = 0
        cutoff_time = datetime.now().timestamp() - (self.config.retention_hours * 3600)

        # Scan all session directories under the offload parent
        offload_parent = self.offload_dir.parent
        try:
            for session_dir in offload_parent.iterdir():
                if not session_dir.is_dir():
                    continue

                for file_path in session_dir.iterdir():
                    if not file_path.is_file():
                        continue

                    try:
                        file_mtime = file_path.stat().st_mtime
                        if file_mtime < cutoff_time:
                            file_path.unlink()
                            removed += 1
                            logger.debug(f"Removed expired offload file: {file_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove expired file {file_path}: {e}"
                        )

                # Remove empty session directories
                try:
                    if session_dir.exists() and not any(session_dir.iterdir()):
                        session_dir.rmdir()
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Error during retention cleanup: {e}")

        if removed > 0:
            logger.info(
                f"Retention cleanup: removed {removed} files older than "
                f"{self.config.retention_hours} hours"
            )

        return removed

    def get_stats(self) -> dict:
        """Get statistics about offloaded content.

        Returns:
            Dictionary with offload statistics.
        """
        return {
            "enabled": self.config.enabled,
            "files_count": len(self._offloaded_files),
            "total_size_bytes": self._total_size_bytes,
            "total_size_mb": round(self._total_size_bytes / (1024 * 1024), 2),
            "offload_dir": str(self.offload_dir),
        }
