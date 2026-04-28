"""I/O utility functions for the app server."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from openhands.utils.async_utils import call_sync_from_async


def _atomic_write(file_path: Path, contents: str) -> None:
    """Write contents to a file atomically.

    Uses a write-to-temp-then-rename strategy to prevent race conditions
    where concurrent writes could corrupt the file.

    Args:
        file_path: The path to the file to write.
        contents: The string contents to write.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = f'{file_path}.tmp.{os.getpid()}.{threading.get_ident()}'
    try:
        with open(temp_path, 'w') as f:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


async def write_file_atomic(file_path: Path, contents: str) -> None:
    """Asynchronously write contents to a file atomically.

    This is an async wrapper around _atomic_write that runs the blocking
    I/O operation in a thread pool.

    Args:
        file_path: The path to the file to write.
        contents: The string contents to write.
    """
    await call_sync_from_async(_atomic_write, file_path, contents)
