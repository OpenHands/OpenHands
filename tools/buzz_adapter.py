"""Buzz inbound adapter: sockets by default, webhook when a URL is configured."""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlparse

from channel_host import (
    STATE_RUNNING,
    STATE_STOPPED,
    STATE_UNCONFIGURED,
    new_id,
    normalize_payload,
)
from project_config import CHANNEL_TYPES

CHANNEL_TYPE_BUZZ = "buzz"
BUZZ_TOKEN = "BUZZ_TOKEN"
BUZZ_WEBHOOK_URL = "BUZZ_WEBHOOK_URL"

assert CHANNEL_TYPE_BUZZ in CHANNEL_TYPES


class FakeBuzzTransport:
    def __init__(self) -> None:
        self.connected = False
        self.posted: list[dict[str, Any]] = []
        self._on_event: Callable[[dict[str, Any]], None] | None = None

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def on_event(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._on_event = handler

    def emit(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event)

    def post_message(
        self,
        channel_ref: str,
        text: str,
        thread_ref: str | None = None,
    ) -> str:
        correlation_id = new_id()
        self.posted.append(
            {
                "channel_ref": channel_ref,
                "text": text,
                "thread_ref": thread_ref,
                "correlation_id": correlation_id,
            }
        )
        return correlation_id


class BuzzAdapter:
    channel_type = CHANNEL_TYPE_BUZZ

    def __init__(
        self,
        *,
        secrets_get: Callable[[], dict[str, str]] | None = None,
        transport: FakeBuzzTransport | None = None,
    ) -> None:
        self._secrets_get = secrets_get or (lambda: {})
        self._transport = transport if transport is not None else FakeBuzzTransport()
        self._state = STATE_STOPPED
        self._handler: Callable[[dict[str, Any]], None] | None = None
        self._transport.on_event(self.handle_event)

    def _secrets(self) -> dict[str, str]:
        return self._secrets_get() or {}

    def _configured(self) -> bool:
        secrets = self._secrets()
        return bool(secrets.get(BUZZ_TOKEN))

    def _mode(self) -> str:
        if self._secrets().get(BUZZ_WEBHOOK_URL):
            return "webhook"
        return "sockets"

    def config_summary(self) -> dict[str, Any]:
        webhook = self._secrets().get(BUZZ_WEBHOOK_URL) or ""
        return {
            "mode": self._mode() if self._configured() else STATE_UNCONFIGURED,
            "webhook_host": urlparse(webhook).netloc if webhook else "",
        }

    def status(self) -> dict[str, Any]:
        if not self._configured():
            return {"state": STATE_UNCONFIGURED, "mode": None, "metrics": {}}
        return {"state": self._state, "mode": self._mode(), "metrics": {}}

    def start(self) -> None:
        if not self._configured():
            self._state = STATE_UNCONFIGURED
            return
        self._transport.connect()
        self._state = STATE_RUNNING

    def idempotent_stop(self) -> None:
        self._transport.close()
        self._state = STATE_STOPPED

    def send(
        self,
        channel_ref: str,
        message: str,
        *,
        thread_ref: str | None = None,
    ) -> str:
        return self._transport.post_message(channel_ref, message, thread_ref)

    def on_message(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handler = handler

    def ack(self, correlation_id: str) -> None:
        return

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        return normalize_payload(self.channel_type, raw)

    def handle_event(self, raw: dict[str, Any]) -> None:
        if self._handler is not None:
            self._handler(raw)

    def handle_webhook(self, payload: dict[str, Any]) -> None:
        event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
        self.handle_event(event)
