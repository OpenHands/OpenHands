"""Server-side store for Databricks U2M OAuth session material.

Holds **secret** material — OAuth access/refresh tokens and confidential-app
client secrets — OUT of the signed-but-unencrypted Starlette session cookie.
The cookie carries only an opaque session id; everything sensitive lives here,
keyed by that id.

Scaling note
------------
This is an **in-memory, single-process** store. It is *not* shared across
workers or replicas, so the U2M browser-login flow must be deployed with a
single worker (``uvicorn``/``gunicorn -w 1``) — otherwise a sign-in handled by
one worker is invisible to the worker serving the follow-up request. For a
multi-worker / multi-pod deployment, swap ``u2m_session_store`` for a shared
backend (e.g. Redis) that implements the same ``put``/``get``/``delete`` API.

Memory is bounded two ways so a long-running process does not leak:
  * a sliding **TTL** evicts idle sessions, and
  * a hard **max-entries** cap drops the least-recently-used record.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

# A U2M session lives as long as its refresh token is useful. Seven days with a
# sliding window comfortably covers an interactive working period without
# retaining abandoned sessions forever.
_DEFAULT_TTL_S = 7 * 24 * 3600
_DEFAULT_MAX_ENTRIES = 10_000


class U2MSessionStore:
    """Thread-safe TTL + LRU store mapping an opaque session id to a record dict.

    Records are arbitrary dicts (e.g. ``{"tokens": {...}, "oauth_client_secret":
    "..."}``). ``put`` merges into any existing record so callers can populate
    different keys at different points in the OAuth flow.
    """

    def __init__(
        self,
        ttl_s: float = _DEFAULT_TTL_S,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._lock = threading.Lock()
        # session_id -> (expiry_epoch, record)
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def put(self, session_id: str, mapping: dict[str, Any]) -> None:
        """Merge *mapping* into the record for *session_id* and refresh its TTL."""
        if not session_id:
            return
        with self._lock:
            self._evict_expired_locked()
            _expiry, record = self._data.get(session_id, (0.0, {}))
            merged = {**record, **mapping}
            self._data[session_id] = (time.time() + self._ttl_s, merged)
            self._data.move_to_end(session_id)
            # Bound memory: drop least-recently-used records past the cap.
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def get(self, session_id: str | None) -> dict[str, Any] | None:
        """Return a copy of the record (sliding-refreshing its TTL), or None."""
        if not session_id:
            return None
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return None
            expiry, record = entry
            if time.time() >= expiry:
                del self._data[session_id]
                return None
            self._data[session_id] = (time.time() + self._ttl_s, record)
            self._data.move_to_end(session_id)
            return dict(record)

    def delete(self, session_id: str | None) -> None:
        """Remove the record for *session_id* if present."""
        if not session_id:
            return
        with self._lock:
            self._data.pop(session_id, None)

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [sid for sid, (exp, _) in self._data.items() if now >= exp]
        for sid in expired:
            del self._data[sid]


# Process-wide singleton. See the module docstring for the single-worker caveat.
u2m_session_store = U2MSessionStore()
