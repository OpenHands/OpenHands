"""Slack inbound adapter implementing ChannelPlugin.

Sockets mode is the default (no public URL). A webhook URL in secrets
switches inbound to REST. The websocket/HTTP client is injected so tests
never open a real Slack connection.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Callable
from urllib.parse import urlparse

from channel_host import (
    STATE_RUNNING,
    STATE_STOPPED,
    STATE_UNCONFIGURED,
    ChannelPlugin,
    new_id,
    normalize_payload,
)
from project_config import CHANNEL_TYPES

SLACK_BOT_TOKEN = "SLACK_BOT_TOKEN"
SLACK_APP_TOKEN = "SLACK_APP_TOKEN"
SLACK_WEBHOOK_URL = "SLACK_WEBHOOK_URL"
CHANNEL_TYPE_SLACK = "slack"

assert CHANNEL_TYPE_SLACK in CHANNEL_TYPES


class FakeSlackTransport:
    """Test double for the Slack socket client / chat.postMessage."""

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
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> str:
        correlation_id = new_id()
        self.posted.append(
            {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
                "correlation_id": correlation_id,
            }
        )
        return correlation_id


def match_routing_rule(
    channel_ref: str,
    rules: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for rule in rules:
        pattern = str(rule.get("pattern") or "")
        if pattern and fnmatch.fnmatch(channel_ref, pattern):
            return rule
    return None


class SlackAdapter:
    channel_type = CHANNEL_TYPE_SLACK

    def __init__(
        self,
        *,
        secrets_get: Callable[[], dict[str, str]] | None = None,
        transport: FakeSlackTransport | None = None,
        routing_rules: list[dict[str, Any]] | None = None,
        bot_user_id: str = "USLACKBOT",
    ) -> None:
        self._secrets_get = secrets_get or (lambda: {})
        self._transport = transport if transport is not None else FakeSlackTransport()
        self.routing_rules = list(routing_rules or [])
        self.bot_user_id = bot_user_id
        self._state = STATE_STOPPED
        self._handler: Callable[[dict[str, Any]], None] | None = None
        self._transport.on_event(self.handle_event)

    def _secrets(self) -> dict[str, str]:
        return self._secrets_get() or {}

    def _configured(self) -> bool:
        secrets = self._secrets()
        bot = secrets.get(SLACK_BOT_TOKEN)
        app = secrets.get(SLACK_APP_TOKEN)
        webhook = secrets.get(SLACK_WEBHOOK_URL)
        return bool(bot) and bool(app or webhook)

    def _mode(self) -> str:
        if self._secrets().get(SLACK_WEBHOOK_URL):
            return "webhook"
        return "sockets"

    def config_summary(self) -> dict[str, Any]:
        secrets = self._secrets()
        webhook = secrets.get(SLACK_WEBHOOK_URL) or ""
        host = urlparse(webhook).netloc if webhook else ""
        return {
            "mode": self._mode() if self._configured() else STATE_UNCONFIGURED,
            "has_bot_token": bool(secrets.get(SLACK_BOT_TOKEN)),
            "has_app_token": bool(secrets.get(SLACK_APP_TOKEN)),
            "webhook_host": host,
            "routing_rules": self.routing_rules,
        }

    def status(self) -> dict[str, Any]:
        if not self._configured():
            return {
                "state": STATE_UNCONFIGURED,
                "mode": None,
                "metrics": {"posted": len(self._transport.posted)},
            }
        return {
            "state": self._state,
            "mode": self._mode(),
            "metrics": {"posted": len(self._transport.posted)},
        }

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
        return self._transport.post_message(channel_ref, message, thread_ts=thread_ref)

    def on_message(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handler = handler

    def ack(self, correlation_id: str) -> None:
        return

    def update_config(self, payload: dict[str, Any]) -> None:
        if "routing_rules" in payload:
            self.routing_rules = list(payload.get("routing_rules") or [])
        if "bot_user_id" in payload and payload["bot_user_id"]:
            self.bot_user_id = str(payload["bot_user_id"])

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        envelope = normalize_payload(self.channel_type, raw)
        rule = match_routing_rule(envelope["channel_ref"], self.routing_rules)
        if rule:
            envelope["routing"] = {
                "project_id": rule.get("project_id"),
                "loop": rule.get("loop"),
                "card_type": rule.get("card_type"),
            }
        return envelope

    def filter_inbound(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        user = str(raw.get("user") or raw.get("bot_id") or "")
        if user and user == self.bot_user_id:
            return None
        if raw.get("subtype") == "bot_message":
            return None
        return self.normalize(raw)

    def handle_event(self, raw: dict[str, Any]) -> None:
        if self.filter_inbound(raw) is None:
            return
        if self._handler is not None:
            self._handler(raw)

    def handle_webhook(self, payload: dict[str, Any]) -> None:
        event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
        self.handle_event(event)


_PLUGIN_CHECK: ChannelPlugin = SlackAdapter(secrets_get=lambda: {})
