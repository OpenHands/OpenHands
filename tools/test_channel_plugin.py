"""Unit and API tests for the channel host.

Run from the repo root:

    python3 -m unittest tools.test_channel_plugin
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from channel_host import (  # noqa: E402
    DIRECTION_INBOUND,
    DIRECTION_OUTBOUND,
    ENVELOPE_FIELDS,
    STATE_RUNNING,
    STATE_STOPPED,
    ChannelHost,
    MemoryChannelAdapter,
)
from channel_host_api import handle_request  # noqa: E402
from project_config import CHANNEL_TYPES  # noqa: E402


def _request(
    host: ChannelHost,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return handle_request(host, method, path, body)


class MemoryAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = MemoryChannelAdapter(channel_type="slack")
        self.host = ChannelHost(":memory:")
        self.addCleanup(self.host.close)
        self.channel_id = self.host.register(self.adapter)

    def test_lifecycle_is_idempotent(self) -> None:
        self.host.start(self.channel_id)
        self.host.start(self.channel_id)
        status = self.host.get_channel(self.channel_id)
        self.assertEqual(status["status"]["state"], STATE_RUNNING)
        self.host.stop(self.channel_id)
        self.host.stop(self.channel_id)
        status = self.host.get_channel(self.channel_id)
        self.assertEqual(status["status"]["state"], STATE_STOPPED)

    def test_normalize_raw_payload(self) -> None:
        self.host.start(self.channel_id)
        envelope = self.host.ingest(
            self.channel_id,
            {
                "channel": "C123",
                "user": "U9",
                "text": "ship it",
                "thread_ts": "1.1",
                "ts": "1.2",
            },
        )
        self.assertIsNotNone(envelope)
        assert envelope is not None
        for field in ENVELOPE_FIELDS:
            self.assertIn(field, envelope)
        self.assertEqual(envelope["source"], "slack")
        self.assertEqual(envelope["channel_ref"], "C123")
        self.assertEqual(envelope["author"], "U9")
        self.assertEqual(envelope["text"], "ship it")
        self.assertEqual(envelope["thread_ref"], "1.1")

    def test_dedup_drops_duplicate_within_window(self) -> None:
        self.host.start(self.channel_id)
        raw = {
            "channel": "C1",
            "user": "U1",
            "text": "hello",
            "thread_ts": "9.0",
        }
        first = self.host.ingest(self.channel_id, raw)
        second = self.host.ingest(self.channel_id, raw)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        messages = self.host.list_messages(channel_id=self.channel_id)
        inbound = [m for m in messages if m["direction"] == DIRECTION_INBOUND]
        self.assertEqual(len(inbound), 1)

    def test_thread_maps_to_stable_session(self) -> None:
        self.host.start(self.channel_id)
        first = self.host.ingest(
            self.channel_id,
            {"channel": "C1", "user": "A", "text": "one", "thread_ts": "t-1"},
        )
        second = self.host.ingest(
            self.channel_id,
            {"channel": "C1", "user": "B", "text": "two", "thread_ts": "t-1"},
        )
        other = self.host.ingest(
            self.channel_id,
            {"channel": "C1", "user": "A", "text": "other", "thread_ts": "t-2"},
        )
        assert first is not None and second is not None and other is not None
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertNotEqual(first["session_id"], other["session_id"])

    def test_send_returns_correlation_id_and_ack_marks_it(self) -> None:
        self.host.start(self.channel_id)
        correlation_id = self.host.send(self.channel_id, "C1", "pong")
        self.assertTrue(correlation_id)
        outbound = self.host.list_messages(
            channel_id=self.channel_id,
            direction=DIRECTION_OUTBOUND,
            correlation_id=correlation_id,
        )
        self.assertEqual(len(outbound), 1)
        self.assertFalse(outbound[0]["acked"])
        marked = self.host.ack(correlation_id)
        self.assertTrue(marked["acked"])
        outbound = self.host.list_messages(correlation_id=correlation_id)
        self.assertTrue(outbound[0]["acked"])

    def test_message_log_round_trips_both_directions(self) -> None:
        self.host.start(self.channel_id)
        inbound = self.host.ingest(
            self.channel_id,
            {"channel": "C1", "user": "U1", "text": "hi", "thread_ts": "t"},
        )
        correlation_id = self.host.send(self.channel_id, "C1", "yo", thread_ref="t")
        log = self.host.list_messages(channel_id=self.channel_id)
        directions = {row["direction"] for row in log}
        self.assertEqual(directions, {DIRECTION_INBOUND, DIRECTION_OUTBOUND})
        assert inbound is not None
        self.assertEqual(inbound["text"], "hi")
        self.assertEqual(log[0]["correlation_id"] or correlation_id, log[0]["correlation_id"])
        self.assertTrue(any(row["correlation_id"] == correlation_id for row in log))

    def test_absent_channel_types_are_not_started(self) -> None:
        host = ChannelHost(":memory:")
        self.addCleanup(host.close)
        host.discover_adapters({})
        self.assertEqual(host.list_channels(), [])
        host.discover_adapters({"slack": lambda: MemoryChannelAdapter(channel_type="slack")})
        listed = host.list_channels()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], "slack")
        self.assertEqual(listed[0]["status"]["state"], STATE_STOPPED)
        self.assertTrue(set(CHANNEL_TYPES) >= {"slack", "whatsapp", "buzz"})


class ChannelApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.host = ChannelHost(":memory:")
        self.addCleanup(self.host.close)
        self.host.register(MemoryChannelAdapter(channel_type="slack"), channel_id="slack")

    def test_list_and_get_and_lifecycle(self) -> None:
        status, listed = _request(self.host, "GET", "/api/channels")
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)
        status, detail = _request(self.host, "GET", "/api/channels/slack")
        self.assertEqual(status, 200)
        self.assertEqual(detail["id"], "slack")
        status, started = _request(self.host, "POST", "/api/channels/slack/start")
        self.assertEqual(status, 200)
        self.assertEqual(started["status"]["state"], STATE_RUNNING)
        status, started_again = _request(self.host, "POST", "/api/channels/slack/start")
        self.assertEqual(status, 200)
        status, stopped = _request(self.host, "POST", "/api/channels/slack/stop")
        self.assertEqual(status, 200)
        self.assertEqual(stopped["status"]["state"], STATE_STOPPED)
        _request(self.host, "POST", "/api/channels/slack/stop")

    def test_messages_filter_and_pagination(self) -> None:
        self.host.start("slack")
        self.host.ingest(
            "slack",
            {"channel": "C1", "user": "U1", "text": "a", "thread_ts": "t"},
        )
        cid = self.host.send("slack", "C1", "b")
        status, page = _request(
            self.host,
            "GET",
            "/api/channels/messages?channel=slack&direction=outbound&limit=10",
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(page["items"]), 1)
        self.assertEqual(page["items"][0]["correlation_id"], cid)
        status, filtered = _request(
            self.host,
            "GET",
            f"/api/channels/messages?correlation_id={cid}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(filtered["items"]), 1)


if __name__ == "__main__":
    unittest.main()
