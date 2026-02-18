"""Idle timeout manager for sandbox containers.

Tracks sandbox activity via proxy traffic and pauses sandboxes
that have been idle beyond the configured timeout.
"""

import logging
import time

_logger = logging.getLogger(__name__)


class IdleTimeoutManager:
    """Tracks sandbox activity and detects idle sandboxes.

    Activity is recorded by proxy routers whenever HTTP or WebSocket
    traffic flows to/from a sandbox.  A background task periodically
    calls ``get_sandboxes_to_warn`` and ``get_sandboxes_to_pause``
    to drive the lifecycle.
    """

    def __init__(self, timeout_seconds: int = 1800, warning_seconds: int = 300) -> None:
        self.timeout_seconds = timeout_seconds
        self.warning_seconds = warning_seconds
        # sandbox_id -> monotonic timestamp of last activity
        self._last_activity: dict[str, float] = {}
        # sandbox_id -> True if a warning has already been sent
        self._warning_sent: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Activity tracking
    # ------------------------------------------------------------------

    def touch(self, sandbox_id: str) -> None:
        """Record activity for a sandbox, resetting its idle timer."""
        now = time.monotonic()
        self._last_activity[sandbox_id] = now
        # Reset warning flag on new activity
        self._warning_sent[sandbox_id] = False

    def remove(self, sandbox_id: str) -> None:
        """Stop tracking a sandbox (e.g. after it has been paused)."""
        self._last_activity.pop(sandbox_id, None)
        self._warning_sent.pop(sandbox_id, None)

    # ------------------------------------------------------------------
    # Idle detection
    # ------------------------------------------------------------------

    def _idle_seconds(self, sandbox_id: str) -> float | None:
        last = self._last_activity.get(sandbox_id)
        if last is None:
            return None
        return time.monotonic() - last

    def get_sandboxes_to_warn(self) -> list[str]:
        """Return sandbox IDs that have crossed the warning threshold.

        Only returns IDs that have not yet been warned.
        """
        threshold = self.timeout_seconds - self.warning_seconds
        result: list[str] = []
        for sandbox_id in list(self._last_activity):
            idle = self._idle_seconds(sandbox_id)
            if (
                idle is not None
                and idle >= threshold
                and not self._warning_sent.get(sandbox_id, False)
            ):
                self._warning_sent[sandbox_id] = True
                result.append(sandbox_id)
        return result

    def get_sandboxes_to_pause(self) -> list[str]:
        """Return sandbox IDs that have exceeded the idle timeout."""
        result: list[str] = []
        for sandbox_id in list(self._last_activity):
            idle = self._idle_seconds(sandbox_id)
            if idle is not None and idle >= self.timeout_seconds:
                result.append(sandbox_id)
        return result

    # ------------------------------------------------------------------
    # Status query (used by the REST endpoint)
    # ------------------------------------------------------------------

    def get_idle_status(self, sandbox_id: str) -> dict | None:
        """Return the idle status dict for a sandbox, or ``None`` if untracked."""
        idle = self._idle_seconds(sandbox_id)
        if idle is None:
            return None
        warning_threshold = self.timeout_seconds - self.warning_seconds
        return {
            'idle_seconds': int(idle),
            'timeout_seconds': self.timeout_seconds,
            'warning_seconds': self.warning_seconds,
            'is_warning': idle >= warning_threshold,
            'remaining_seconds': max(0, int(self.timeout_seconds - idle)),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: IdleTimeoutManager | None = None


def init_idle_timeout_manager(
    timeout_seconds: int = 1800,
    warning_seconds: int = 300,
) -> IdleTimeoutManager:
    """Create (or replace) the global idle-timeout manager.

    Called once during application startup from the lifespan service.
    """
    global _manager
    _manager = IdleTimeoutManager(
        timeout_seconds=timeout_seconds,
        warning_seconds=warning_seconds,
    )
    _logger.info(
        'Idle timeout manager initialised: '
        f'timeout={timeout_seconds}s, warning={warning_seconds}s'
    )
    return _manager


def get_idle_timeout_manager() -> IdleTimeoutManager | None:
    """Return the global manager, or ``None`` if not yet initialised."""
    return _manager
