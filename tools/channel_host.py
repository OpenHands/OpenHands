"""Inbound channel host: one plugin interface, one envelope, one pipeline.

Adapters implement ``ChannelPlugin``. Downstream consumers (kanban, loops,
Meetily) only see the normalized envelope. Persistence is SQLite so a Slack
thread stays one session across restarts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from project_config import CHANNEL_TYPES

CHANNELS_DB_FILENAME = "channels.sqlite"
ENVELOPE_FIELDS = (
    "source",
    "channel_ref",
    "author",
    "text",
    "thread_ref",
    "timestamp",
    "raw",
)
DIRECTION_INBOUND = "inbound"
DIRECTION_OUTBOUND = "outbound"
STATE_STOPPED = "stopped"
STATE_RUNNING = "running"
STATE_UNCONFIGURED = "unconfigured"
STATE_ERROR = "error"
DEFAULT_DEDUP_WINDOW_S = 5.0
HOLD_FOR_HUMAN = "hold_for_human"

InboundHandler = Callable[[dict[str, Any]], None]


class ChannelError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(ChannelError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


class ChannelPlugin(Protocol):
    channel_type: str

    def start(self) -> None: ...

    def idempotent_stop(self) -> None: ...

    def send(
        self,
        channel_ref: str,
        message: str,
        *,
        thread_ref: str | None = None,
    ) -> str: ...

    def on_message(self, handler: Callable[[dict[str, Any]], None]) -> None: ...

    def ack(self, correlation_id: str) -> None: ...

    def status(self) -> dict[str, Any]: ...


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, CHANNELS_DB_FILENAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def normalize_payload(source: str, raw: dict[str, Any]) -> dict[str, Any]:
    thread_ref = str(
        raw.get("thread_ref") or raw.get("thread_ts") or raw.get("ts") or ""
    )
    return {
        "source": source,
        "channel_ref": str(raw.get("channel_ref") or raw.get("channel") or ""),
        "author": str(raw.get("author") or raw.get("user") or ""),
        "text": str(raw.get("text") or ""),
        "thread_ref": thread_ref,
        "timestamp": str(raw.get("timestamp") or utc_now()),
        "raw": raw,
    }


class MemoryChannelAdapter:
    """In-memory ChannelPlugin used by tests (and as a development stub)."""

    def __init__(self, channel_type: str = "slack") -> None:
        if channel_type not in CHANNEL_TYPES:
            raise ChannelError(f"Unknown channel type: {channel_type}")
        self.channel_type = channel_type
        self._state = STATE_STOPPED
        self._handler: Callable[[dict[str, Any]], None] | None = None
        self._acked: set[str] = set()
        self.sent: list[dict[str, Any]] = []

    def start(self) -> None:
        self._state = STATE_RUNNING

    def idempotent_stop(self) -> None:
        self._state = STATE_STOPPED

    def send(
        self,
        channel_ref: str,
        message: str,
        *,
        thread_ref: str | None = None,
    ) -> str:
        correlation_id = new_id()
        self.sent.append(
            {
                "channel_ref": channel_ref,
                "message": message,
                "thread_ref": thread_ref,
                "correlation_id": correlation_id,
            }
        )
        return correlation_id

    def on_message(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handler = handler

    def ack(self, correlation_id: str) -> None:
        self._acked.add(correlation_id)

    def status(self) -> dict[str, Any]:
        return {"state": self._state, "metrics": {"sent": len(self.sent)}}

    def deliver(self, raw: dict[str, Any]) -> None:
        if self._handler is not None:
            self._handler(raw)


class ChannelHost:
    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        dedup_window_s: float = DEFAULT_DEDUP_WINDOW_S,
        inbound_handler: InboundHandler | None = None,
    ) -> None:
        self.dedup_window_s = dedup_window_s
        self.inbound_handler = inbound_handler or self._default_handler
        self._lock = threading.Lock()
        self._adapters: dict[str, ChannelPlugin] = {}
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                session_id TEXT,
                envelope_json TEXT NOT NULL,
                acked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dedup (
                fingerprint TEXT PRIMARY KEY,
                seen_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def register(
        self,
        adapter: ChannelPlugin,
        *,
        channel_id: str | None = None,
    ) -> str:
        cid = channel_id or adapter.channel_type
        with self._lock:
            self._adapters[cid] = adapter
            adapter.on_message(lambda raw, _cid=cid: self.ingest(_cid, raw))
        return cid

    def discover_adapters(
        self,
        factories: dict[str, Callable[[], ChannelPlugin]],
    ) -> list[str]:
        """Register factories for ``CHANNEL_TYPES``. Missing types are skipped."""
        registered: list[str] = []
        for channel_type in CHANNEL_TYPES:
            factory = factories.get(channel_type)
            if factory is None:
                continue
            registered.append(self.register(factory(), channel_id=channel_type))
        return registered

    def _adapter(self, channel_id: str) -> ChannelPlugin:
        adapter = self._adapters.get(channel_id)
        if adapter is None:
            raise NotFoundError(f"Channel {channel_id} not found")
        return adapter

    def list_channels(self) -> list[dict[str, Any]]:
        return [self.get_channel(channel_id) for channel_id in self._adapters]

    def get_channel(self, channel_id: str) -> dict[str, Any]:
        adapter = self._adapter(channel_id)
        status = adapter.status()
        return {
            "id": channel_id,
            "type": adapter.channel_type,
            "status": status,
            "config": getattr(adapter, "config_summary", lambda: {})(),
        }

    def start(self, channel_id: str) -> dict[str, Any]:
        adapter = self._adapter(channel_id)
        if adapter.status().get("state") == STATE_UNCONFIGURED:
            return self.get_channel(channel_id)
        adapter.start()
        return self.get_channel(channel_id)

    def stop(self, channel_id: str) -> dict[str, Any]:
        adapter = self._adapter(channel_id)
        adapter.idempotent_stop()
        return self.get_channel(channel_id)

    def ingest(self, channel_id: str, raw: dict[str, Any]) -> dict[str, Any] | None:
        adapter = self._adapter(channel_id)
        source = adapter.channel_type
        normalizer = getattr(adapter, "normalize", None)
        envelope = (
            normalizer(raw)
            if callable(normalizer)
            else normalize_payload(source, raw)
        )
        envelope["source"] = source
        if envelope.get(HOLD_FOR_HUMAN):
            envelope[HOLD_FOR_HUMAN] = True
        if self._is_duplicate(source, envelope):
            return None
        session_id = self._session_for(source, str(envelope.get("thread_ref") or ""))
        envelope["session_id"] = session_id
        correlation_id = str(envelope.get("correlation_id") or new_id())
        envelope["correlation_id"] = correlation_id
        self._log(
            channel_id,
            DIRECTION_INBOUND,
            correlation_id,
            session_id,
            envelope,
        )
        self.inbound_handler(envelope)
        if not envelope.get(HOLD_FOR_HUMAN):
            self.ack(correlation_id)
        return envelope

    def send(
        self,
        channel_id: str,
        channel_ref: str,
        message: str,
        *,
        thread_ref: str | None = None,
    ) -> str:
        adapter = self._adapter(channel_id)
        correlation_id = adapter.send(
            channel_ref, message, thread_ref=thread_ref
        )
        session_id = (
            self._session_for(adapter.channel_type, thread_ref)
            if thread_ref
            else None
        )
        envelope = {
            "source": adapter.channel_type,
            "channel_ref": channel_ref,
            "author": "agent",
            "text": message,
            "thread_ref": thread_ref or "",
            "timestamp": utc_now(),
            "raw": {"outbound": True},
            "session_id": session_id,
            "correlation_id": correlation_id,
        }
        self._log(
            channel_id,
            DIRECTION_OUTBOUND,
            correlation_id,
            session_id,
            envelope,
        )
        return correlation_id

    def auto_reply(
        self,
        channel_id: str,
        envelope: dict[str, Any],
        message: str,
    ) -> str | None:
        if envelope.get(HOLD_FOR_HUMAN):
            return None
        return self.send(
            channel_id,
            str(envelope.get("channel_ref") or ""),
            message,
            thread_ref=str(envelope.get("thread_ref") or "") or None,
        )

    def ack(self, correlation_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT channel_id FROM messages WHERE correlation_id = ? LIMIT 1",
                (correlation_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Message {correlation_id} not found")
            self._conn.execute(
                "UPDATE messages SET acked = 1 WHERE correlation_id = ?",
                (correlation_id,),
            )
            self._conn.commit()
        try:
            self._adapter(row["channel_id"]).ack(correlation_id)
        except ChannelError:
            pass
        messages = self.list_messages(correlation_id=correlation_id, limit=1)
        return messages[0]

    def update_config(self, channel_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        adapter = self._adapter(channel_id)
        updater = getattr(adapter, "update_config", None)
        if callable(updater):
            updater(payload)
        return self.get_channel(channel_id)

    def list_messages(
        self,
        *,
        channel_id: str | None = None,
        direction: str | None = None,
        correlation_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if channel_id:
            clauses.append("channel_id = ?")
            params.append(channel_id)
        if direction:
            clauses.append("direction = ?")
            params.append(direction)
        if correlation_id:
            clauses.append("correlation_id = ?")
            params.append(correlation_id)
        params.extend([max(1, int(limit)), max(0, int(offset))])
        sql = (
            "SELECT * FROM messages WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            envelope = json.loads(row["envelope_json"])
            items.append(
                {
                    "id": row["id"],
                    "channel_id": row["channel_id"],
                    "direction": row["direction"],
                    "correlation_id": row["correlation_id"],
                    "session_id": row["session_id"],
                    "acked": bool(row["acked"]),
                    "created_at": row["created_at"],
                    **envelope,
                }
            )
        return items

    def _default_handler(self, envelope: dict[str, Any]) -> None:
        return

    def _session_for(self, source: str, thread_ref: str) -> str:
        key = f"{source}|{thread_ref or new_id()}"
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id FROM sessions WHERE thread_key = ?",
                (key,),
            ).fetchone()
            if row:
                return row["session_id"]
            session_id = new_id()
            self._conn.execute(
                "INSERT INTO sessions (thread_key, session_id, created_at) VALUES (?, ?, ?)",
                (key, session_id, utc_now()),
            )
            self._conn.commit()
            return session_id

    def _is_duplicate(self, source: str, envelope: dict[str, Any]) -> bool:
        fingerprint = "|".join(
            (
                source,
                str(envelope.get("thread_ref") or ""),
                str(envelope.get("text") or ""),
            )
        )
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT seen_at FROM dedup WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if row and now - float(row["seen_at"]) <= self.dedup_window_s:
                return True
            self._conn.execute(
                "INSERT OR REPLACE INTO dedup (fingerprint, seen_at) VALUES (?, ?)",
                (fingerprint, now),
            )
            self._conn.commit()
        return False

    def _log(
        self,
        channel_id: str,
        direction: str,
        correlation_id: str,
        session_id: str | None,
        envelope: dict[str, Any],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO messages (
                    id, channel_id, direction, correlation_id, session_id,
                    envelope_json, acked, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    new_id(),
                    channel_id,
                    direction,
                    correlation_id,
                    session_id,
                    json.dumps(envelope),
                    utc_now(),
                ),
            )
            self._conn.commit()
