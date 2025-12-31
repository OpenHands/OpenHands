import atexit
import inspect
import threading
import weakref
from collections.abc import Callable
from typing import Any

import anyio
from anyio.from_thread import start_blocking_portal

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class AsyncExecutor:
    """
    Thin wrapper around AnyIO's BlockingPortal to execute async code
    from synchronous contexts with proper resource and timeout handling.
    """

    def __init__(self):
        self._portal = None
        self._portal_cm = None
        self._lock = threading.Lock()
        self._atexit_registered = False

    def _ensure_portal(self):
        with self._lock:
            if self._portal is None:
                self._portal_cm = start_blocking_portal()
                self._portal = self._portal_cm.__enter__()
                # Register atexit handler to ensure cleanup on interpreter shutdown
                if not self._atexit_registered:
                    # Use weakref to avoid keeping the executor alive
                    weak_self = weakref.ref(self)

                    def cleanup():
                        executor = weak_self()
                        if executor is not None:
                            try:
                                executor.close()
                            except Exception:
                                pass

                    atexit.register(cleanup)
                    self._atexit_registered = True
            return self._portal

    def run_async(
        self,
        awaitable_or_fn: Callable[..., Any] | Any,
        *args,
        timeout: float | None = None,
        **kwargs,
    ) -> Any:
        """
        Run a coroutine or async function from sync code.

        Args:
            awaitable_or_fn: coroutine or async function
            *args: positional arguments (only used if awaitable_or_fn is a function)
            timeout: optional timeout in seconds
            **kwargs: keyword arguments (only used if awaitable_or_fn is a function)
        """
        portal = self._ensure_portal()

        # Construct coroutine
        if inspect.iscoroutine(awaitable_or_fn):
            coro = awaitable_or_fn
        elif inspect.iscoroutinefunction(awaitable_or_fn):
            coro = awaitable_or_fn(*args, **kwargs)
        else:
            raise TypeError("run_async expects a coroutine or async function")

        # Apply timeout by wrapping in an async function with fail_after
        if timeout is not None:

            async def _with_timeout():
                with anyio.fail_after(timeout):
                    return await coro

            return portal.call(_with_timeout)
        else:

            async def _execute():
                return await coro

            return portal.call(_execute)

    def close(self):
        with self._lock:
            portal_cm = self._portal_cm
            self._portal_cm = None
            self._portal = None

        if portal_cm is not None:
            try:
                portal_cm.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing BlockingPortal: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
