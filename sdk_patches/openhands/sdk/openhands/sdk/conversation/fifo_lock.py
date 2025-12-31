"""
FIFO Lock implementation that guarantees first-in-first-out access ordering.

This provides fair lock access where threads acquire the lock in the exact order
they requested it, preventing starvation that can occur with standard RLock.
"""

import threading
import time
from collections import deque
from typing import Any, Self


class FIFOLock:
    """
    A reentrant lock that guarantees FIFO (first-in-first-out) access ordering.

    Unlike Python's standard RLock, this lock ensures that threads acquire
    the lock in the exact order they requested it, providing fairness and
    preventing lock starvation.

    Features:
    - Reentrant: Same thread can acquire multiple times
    - FIFO ordering: Threads get lock in request order
    - Context manager support: Use with 'with' statement
    - Thread-safe: Safe for concurrent access
    """

    _mutex: threading.Lock
    _count: int

    def __init__(self) -> None:
        self._mutex = threading.Lock()  # Protects internal state
        self._waiters: deque[threading.Condition] = (
            deque()
        )  # FIFO queue of waiting threads
        self._owner: int | None = None  # Current lock owner thread ID
        self._count = 0  # Reentrancy counter

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """
        Acquire the lock.

        Args:
            blocking: If True, block until lock is acquired. If False, return
                     immediately.
            timeout: Maximum time to wait for lock (ignored if blocking=False).
                    -1 means wait indefinitely.

        Returns:
            True if lock was acquired, False otherwise.
        """
        ident = threading.get_ident()
        start = time.monotonic()

        with self._mutex:
            # Reentrant case
            if self._owner == ident:
                self._count += 1
                return True

            if self._owner is None and not self._waiters:
                self._owner = ident
                self._count = 1
                return True

            if not blocking:
                # Give up immediately
                return False

            # Add to wait queue
            me = threading.Condition(self._mutex)
            self._waiters.append(me)

            while True:
                # If I'm at the front of the queue and nobody owns it → acquire
                if self._waiters[0] is me and self._owner is None:
                    self._waiters.popleft()
                    self._owner = ident
                    self._count = 1
                    return True

                if timeout >= 0:
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        self._waiters.remove(me)
                        return False
                    me.wait(remaining)
                else:
                    me.wait()

    def release(self) -> None:
        """
        Release the lock.

        Raises:
            RuntimeError: If the current thread doesn't own the lock.
        """
        ident = threading.get_ident()
        with self._mutex:
            if self._owner != ident:
                raise RuntimeError("Cannot release lock not owned by current thread")
            assert self._count >= 1, (
                "When releasing the resource, the count must be >= 1"
            )
            self._count -= 1
            if self._count == 0:
                self._owner = None
                if self._waiters:
                    self._waiters[0].notify()

    def __enter__(self: Self) -> Self:
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.release()

    def locked(self) -> bool:
        """
        Return True if the lock is currently held by any thread.
        """
        with self._mutex:
            return self._owner is not None

    def owned(self) -> bool:
        """
        Return True if the lock is currently held by the calling thread.
        """
        with self._mutex:
            return self._owner == threading.get_ident()
