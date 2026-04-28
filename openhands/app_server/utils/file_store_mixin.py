"""I/O utility functions for the app server."""

from __future__ import annotations

import tempfile
from pathlib import Path

from openhands.utils.async_utils import call_sync_from_async


class FileStoreMixin:
    """Mixin providing common file I/O operations for file-based stores.

    Classes using this mixin must define:
        - root_dir: Path - the base directory for file storage
        - filename: str - the name of the file to read/write
    """

    root_dir: Path
    filename: str

    @property
    def file_path(self) -> Path:
        return self.root_dir / self.filename

    def _read_file(self) -> str:
        with open(self.file_path, encoding='utf-8') as f:
            return f.read()

    def _write_file_atomic(self, contents: str) -> None:
        """Write contents to file atomically using write-to-temp-then-rename."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=self.root_dir,
            delete=False,
        ) as f:
            f.write(contents)
            temp_path = Path(f.name)
        temp_path.replace(self.file_path)

    async def _read_file_async(self) -> str:
        return await call_sync_from_async(self._read_file)

    async def _write_file_async(self, contents: str) -> None:
        await call_sync_from_async(self._write_file_atomic, contents)

    @staticmethod
    def _resolve_root_dir(path: str) -> Path:
        """Resolve a path string to a Path, expanding ~ if present."""
        return Path(path).expanduser()
