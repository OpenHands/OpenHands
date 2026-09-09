"""Tests for the WhatsApp inbound adapter."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from channel_host import HOLD_FOR_HUMAN, ChannelHost  # noqa: E402
from whatsapp_adapter import FakeWhatsAppTransport, WhatsAppAdapter  # noqa: E402


class WhatsAppAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeWhatsAppTransport()
        self.adapter = WhatsAppAdapter(
            secrets_get=lambda: {
                "WHATSAPP_TOKEN": "tok",
                "WHATSAPP_PHONE_ID": "123",
            },
            transport=self.transport,
            cost_cap=1.0,
        )
        self.host = ChannelHost(":memory:")
        self.addCleanup(self.host.close)
        self.host.register(self.adapter, channel_id="whatsapp")
        self.host.start("whatsapp")

    def test_normalize_thread_and_send(self) -> None:
        first = self.host.ingest(
            "whatsapp",
            {
                "channel": "wa-1",
                "user": "alice",
                "text": "hi",
                "thread_ts": "t-wa",
            },
        )
        second = self.host.ingest(
            "whatsapp",
            {
                "channel": "wa-1",
                "user": "alice",
                "text": "again",
                "thread_ts": "t-wa",
            },
        )
        assert first is not None and second is not None
        self.assertEqual(first["source"], "whatsapp")
        self.assertEqual(first["session_id"], second["session_id"])
        cid = self.host.send("whatsapp", "wa-1", "pong", thread_ref="t-wa")
        self.assertTrue(cid)
        self.assertEqual(self.transport.posted[-1]["text"], "pong")

    def test_cost_estimate_recorded(self) -> None:
        envelope = self.host.ingest(
            "whatsapp",
            {"channel": "wa-1", "user": "alice", "text": "hello there", "thread_ts": "t"},
        )
        assert envelope is not None
        self.assertGreater(envelope["estimated_cost"], 0)
        log = self.host.list_messages(channel_id="whatsapp", direction="inbound")
        self.assertEqual(log[0]["estimated_cost"], envelope["estimated_cost"])
        self.assertIn("estimated_spend", self.adapter.status()["metrics"])

    def test_over_cap_is_held(self) -> None:
        self.adapter.cost_cap = 0.0000001
        envelope = self.host.ingest(
            "whatsapp",
            {"channel": "wa-1", "user": "alice", "text": "expensive", "thread_ts": "t"},
        )
        assert envelope is not None
        self.assertTrue(envelope[HOLD_FOR_HUMAN])
        self.assertIsNone(self.host.auto_reply("whatsapp", envelope, "nope"))
        self.assertEqual(self.transport.posted, [])

    def test_human_only_disables_auto_reply(self) -> None:
        self.adapter.human_only = True
        envelope = self.host.ingest(
            "whatsapp",
            {"channel": "wa-1", "user": "bob", "text": "help", "thread_ts": "t2"},
        )
        assert envelope is not None
        self.assertTrue(envelope[HOLD_FOR_HUMAN])
        self.assertIsNone(self.host.auto_reply("whatsapp", envelope, "bot"))
