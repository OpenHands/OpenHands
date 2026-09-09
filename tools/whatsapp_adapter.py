"""WhatsApp inbound adapter with per-message cost controls."""

from __future__ import annotations

from typing import Any, Callable

from channel_host import (
    HOLD_FOR_HUMAN,
    STATE_RUNNING,
    STATE_STOPPED,
    STATE_UNCONFIGURED,
    new_id,
    normalize_payload,
)
from project_config import CHANNEL_TYPES

CHANNEL_TYPE_WHATSAPP = "whatsapp"
WHATSAPP_TOKEN = "WHATSAPP_TOKEN"
WHATSAPP_PHONE_ID = "WHATSAPP_PHONE_ID"
DEFAULT_PROVIDER_COST = 0.005
DEFAULT_CONTEXT_RATE = 0.00002

assert CHANNEL_TYPE_WHATSAPP in CHANNEL_TYPES


class FakeWhatsAppTransport:
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


class WhatsAppAdapter:
    channel_type = CHANNEL_TYPE_WHATSAPP

    def __init__(
        self,
        *,
        secrets_get: Callable[[], dict[str, str]] | None = None,
        transport: FakeWhatsAppTransport | None = None,
        cost_cap: float | None = None,
        human_only: bool = False,
        provider_cost: float = DEFAULT_PROVIDER_COST,
        context_rate: float = DEFAULT_CONTEXT_RATE,
    ) -> None:
        self._secrets_get = secrets_get or (lambda: {})
        self._transport = transport if transport is not None else FakeWhatsAppTransport()
        self.cost_cap = cost_cap
        self.human_only = human_only
        self.provider_cost = provider_cost
        self.context_rate = context_rate
        self._state = STATE_STOPPED
        self._handler: Callable[[dict[str, Any]], None] | None = None
        self._estimated_spend = 0.0
        self._transport.on_event(self.handle_event)

    def estimate_cost(self, text: str) -> float:
        tokens = max(1, len(text) / 4)
        return round(self.provider_cost + tokens * self.context_rate, 6)

    def _configured(self) -> bool:
        secrets = self._secrets_get() or {}
        return bool(secrets.get(WHATSAPP_TOKEN) and secrets.get(WHATSAPP_PHONE_ID))

    def config_summary(self) -> dict[str, Any]:
        return {
            "cost_cap": self.cost_cap,
            "human_only": self.human_only,
            "has_token": bool((self._secrets_get() or {}).get(WHATSAPP_TOKEN)),
        }

    def status(self) -> dict[str, Any]:
        if not self._configured():
            return {
                "state": STATE_UNCONFIGURED,
                "metrics": {"estimated_spend": self._estimated_spend},
            }
        return {
            "state": self._state,
            "metrics": {"estimated_spend": self._estimated_spend},
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
        return self._transport.post_message(channel_ref, message, thread_ref)

    def on_message(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handler = handler

    def ack(self, correlation_id: str) -> None:
        return

    def update_config(self, payload: dict[str, Any]) -> None:
        if "cost_cap" in payload:
            self.cost_cap = payload["cost_cap"]
        if "human_only" in payload:
            self.human_only = bool(payload["human_only"])

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        envelope = normalize_payload(self.channel_type, raw)
        cost = self.estimate_cost(envelope["text"])
        envelope["estimated_cost"] = cost
        self._estimated_spend += cost
        hold = self.human_only or (
            self.cost_cap is not None and cost > float(self.cost_cap)
        )
        if hold:
            envelope[HOLD_FOR_HUMAN] = True
        return envelope

    def handle_event(self, raw: dict[str, Any]) -> None:
        if self._handler is not None:
            self._handler(raw)
