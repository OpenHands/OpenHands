"""Tests for the Buzz inbound adapter."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from channel_host import ChannelHost  # noqa: E402
from buzz_adapter import BuzzAdapter, FakeBuzzTransport  # noqa: E402


class BuzzAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeBuzzTransport()
        self.adapter = BuzzAdapter(
            secrets_get=lambda: {"BUZZ_TOKEN": "tok"},
            transport=self.transport,
        )
        self.host = ChannelHost(":memory:")
        self.addCleanup(self.host.close)
        self.host.register(self.adapter, channel_id="buzz")
        self.host.start("buzz")

    def test_normalize_thread_and_send(self) -> None:
        first = self.host.ingest(
            "buzz",
            {"channel": "bz-1", "user": "u", "text": "ping", "thread_ts": "tb"},
        )
        second = self.host.ingest(
            "buzz",
            {"channel": "bz-1", "user": "u", "text": "pong-in", "thread_ts": "tb"},
        )
        assert first is not None and second is not None
        self.assertEqual(first["source"], "buzz")
        self.assertEqual(first["session_id"], second["session_id"])
        cid = self.host.send("buzz", "bz-1", "ack", thread_ref="tb")
        self.assertTrue(cid)

    def test_sockets_and_webhook_normalize_identically(self) -> None:
        event = {
            "channel": "bz-1",
            "user": "u",
            "text": "same",
            "ts": "1",
            "thread_ts": "1",
        }
        sockets = self.adapter.normalize(event)
        webhook = BuzzAdapter(
            secrets_get=lambda: {
                "BUZZ_TOKEN": "tok",
                "BUZZ_WEBHOOK_URL": "https://buzz.example/hook",
            },
            transport=FakeBuzzTransport(),
        )
        via_webhook = webhook.normalize(event)
        self.assertEqual(sockets["text"], via_webhook["text"])
        self.assertEqual(sockets["thread_ref"], via_webhook["thread_ref"])
        self.assertEqual(sockets["channel_ref"], via_webhook["channel_ref"])
        self.assertEqual(webhook.status().get("mode"), "webhook")
        self.assertEqual(self.adapter.status().get("mode"), "sockets")
        webhook.handle_webhook({"event": event})
