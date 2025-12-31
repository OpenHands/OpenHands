from typing import Any

from cachetools import LRUCache

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class MemoryLRUCache(LRUCache):
    """LRU cache with both entry count and memory size limits.

    This cache enforces two limits:
    1. Maximum number of entries (maxsize)
    2. Maximum memory usage in bytes (max_memory)

    When either limit is exceeded, the least recently used items are evicted.

    Note: Memory tracking is based on string length for simplicity and accuracy.
    For non-string values, sys.getsizeof is used as a rough approximation.
    """

    def __init__(self, max_memory: int, max_size: int, *args, **kwargs):
        # Ensure minimum maxsize of 1 to avoid LRUCache issues
        maxsize = max(1, max_size)
        super().__init__(maxsize=maxsize, *args, **kwargs)
        self.max_memory = max_memory
        self.current_memory = 0

    def _get_size(self, value: Any) -> int:
        """Calculate size of value for memory tracking.

        For strings (the common case in FileStore), we use len() which gives
        accurate character count. For other types, we use sys.getsizeof() as
        a rough approximation.
        """
        if isinstance(value, str):
            # For strings, len() gives character count which is what we care about
            # This is much more accurate than sys.getsizeof for our use case
            return len(value)
        elif isinstance(value, bytes):
            return len(value)
        else:
            # For other types, fall back to sys.getsizeof
            # This is mainly for edge cases and won't be accurate for nested
            # structures, but it's better than nothing
            try:
                import sys

                return sys.getsizeof(value)
            except Exception:
                return 0

    def __setitem__(self, key: Any, value: Any) -> None:
        new_size = self._get_size(value)

        # Don't cache items that are larger than max_memory
        # This prevents cache thrashing where one huge item evicts everything
        if new_size > self.max_memory:
            logger.debug(
                f"Item too large for cache ({new_size} bytes > "
                f"{self.max_memory} bytes), skipping cache"
            )
            return

        # Update memory accounting if key exists
        if key in self:
            old_value = self[key]
            self.current_memory -= self._get_size(old_value)

        self.current_memory += new_size

        # Evict items until we're under memory limit
        while self.current_memory > self.max_memory and len(self) > 0:
            self.popitem()

        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        if key in self:
            old_value = self[key]
            self.current_memory -= self._get_size(old_value)

        super().__delitem__(key)
