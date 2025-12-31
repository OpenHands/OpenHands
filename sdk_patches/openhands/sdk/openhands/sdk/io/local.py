import os
import shutil

from openhands.sdk.io.cache import MemoryLRUCache
from openhands.sdk.logger import get_logger
from openhands.sdk.observability.laminar import observe

from .base import FileStore


logger = get_logger(__name__)


class LocalFileStore(FileStore):
    root: str
    cache: MemoryLRUCache

    def __init__(
        self,
        root: str,
        cache_limit_size: int = 500,
        cache_memory_size: int = 20 * 1024 * 1024,
    ) -> None:
        """Initialize a LocalFileStore with caching.

        Args:
            root: Root directory for file storage.
            cache_limit_size: Maximum number of cached entries (default: 500).
            cache_memory_size: Maximum cache memory in bytes (default: 20MB).

        Note:
            The cache assumes exclusive access to files. External modifications
            to files will not be detected and may result in stale cache reads.
        """
        if root.startswith("~"):
            root = os.path.expanduser(root)
        root = os.path.abspath(os.path.normpath(root))
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.cache = MemoryLRUCache(cache_memory_size, cache_limit_size)

    def get_full_path(self, path: str) -> str:
        # strip leading slash to keep relative under root
        if path.startswith("/"):
            path = path[1:]
        # normalize path separators to handle both Unix (/) and Windows (\) styles
        normalized_path = path.replace("\\", "/")
        full = os.path.abspath(
            os.path.normpath(os.path.join(self.root, normalized_path))
        )
        # ensure sandboxing
        if os.path.commonpath([self.root, full]) != self.root:
            raise ValueError(f"path escapes filestore root: {path}")

        return full

    @observe(name="LocalFileStore.write", span_type="TOOL")
    def write(self, path: str, contents: str | bytes) -> None:
        full_path = self.get_full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if isinstance(contents, str):
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(contents)
            self.cache[full_path] = contents
        else:
            with open(full_path, "wb") as f:
                f.write(contents)
            # Don't cache binary content - LocalFileStore is meant for JSON data
            # If binary data is written and then read, it will error on read

    def read(self, path: str) -> str:
        full_path = self.get_full_path(path)

        if full_path in self.cache:
            return self.cache[full_path]

        if not os.path.exists(full_path):
            raise FileNotFoundError(path)

        with open(full_path, encoding="utf-8") as f:
            result = f.read()

        self.cache[full_path] = result
        return result

    @observe(name="LocalFileStore.list", span_type="TOOL")
    def list(self, path: str) -> list[str]:
        full_path = self.get_full_path(path)
        if not os.path.exists(full_path):
            return []

        # If path is a file, return the file itself (S3-consistent behavior)
        if os.path.isfile(full_path):
            return [path]

        # Otherwise it's a directory, return its contents
        files = [os.path.join(path, f) for f in os.listdir(full_path)]
        files = [f + "/" if os.path.isdir(self.get_full_path(f)) else f for f in files]
        return files

    @observe(name="LocalFileStore.delete", span_type="TOOL")
    def delete(self, path: str) -> None:
        try:
            full_path = self.get_full_path(path)
            if not os.path.exists(full_path):
                logger.debug(f"Local path does not exist: {full_path}")
                return

            if os.path.isfile(full_path):
                os.remove(full_path)
                del self.cache[full_path]
                logger.debug(f"Removed local file: {full_path}")
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                self.cache.clear()
                logger.debug(f"Removed local directory: {full_path}")

        except Exception as e:
            logger.error(f"Error clearing local file store: {str(e)}")
