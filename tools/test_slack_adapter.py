"""Unit tests for the Slack inbound adapter.

Run from the repo root:

    python3 -m unittest tools.test_slack_adapter
"""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from channel_host import ChannelHost, STATE_STOPPED, STATE_UNCONFIGURED  # noqa: E402
from slack_adapter import (  # noqa: E402
    FakeSlackTransport,
    SlackAdapter,
    match_routing_rule,
)


def _secrets(**kwargs: str) -> dict[str, str]:
    return kwargs


class SlackAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeSlackTransport()
        self.adapter = SlackAdapter(
            secrets_get=lambda: _secrets(
                SLACK_BOT_TOKEN="xoxb-test",
                SLACK_APP_TOKEN="xapp-test",
            ),
            transport=self.transport,
            bot_user_id="BME",
            routing_rules=[
                {
                    "pattern": "C-eng-*",
                    "project_id": "proj-eng",
                    "loop": "commit-loop",
                    "card_type": "task",
                },
                {
                    "pattern": "C-bugs",
                    "project_id": "proj-bugs",
                    "card_type": "bug",
                },
            ],
        )
        self.host = ChannelHost(":memory:")
        self.addCleanup(self.host.close)
        self.host.register(self.adapter, channel_id="slack")
        self.host.start("slack")

    def test_normalize_inbound_event(self) -> None:
        envelope = self.host.ingest(
            "slack",
            {
                "type": "message",
                "channel": "C-eng-1",
                "user": "U1",
                "text": "please ship",
                "ts": "10.1",
                "thread_ts": "10.0",
            },
        )
        assert envelope is not None
        self.assertEqual(envelope["source"], "slack")
        self.assertEqual(envelope["author"], "U1")
        self.assertEqual(envelope["text"], "please ship")
        self.assertEqual(envelope["thread_ref"], "10.0")
        self.assertEqual(envelope["channel_ref"], "C-eng-1")

    def test_thread_session_mapping(self) -> None:
        a = self.host.ingest(
            "slack",
            {
                "channel": "C1",
                "user": "U1",
                "text": "start",
                "thread_ts": "t-9",
                "ts": "1",
            },
        )
        b = self.host.ingest(
            "slack",
            {
                "channel": "C1",
                "user": "U2",
                "text": "followup",
                "thread_ts": "t-9",
                "ts": "2",
            },
        )
        assert a is not None and b is not None
        self.assertEqual(a["session_id"], b["session_id"])

    def test_send_round_trip(self) -> None:
        cid = self.host.send("slack", "C1", "ack", thread_ref="t-9")
        self.assertTrue(cid)
        self.assertEqual(self.transport.posted[-1]["text"], "ack")
        self.assertEqual(self.transport.posted[-1]["thread_ts"], "t-9")
        self.assertEqual(self.transport.posted[-1]["channel"], "C1")

    def test_self_message_loop_guard(self) -> None:
        envelope = self.adapter.filter_inbound(
            {"channel": "C1", "user": "BME", "text": "echo", "ts": "1"}
        )
        self.assertIsNone(envelope)
        delivered = []
        self.adapter.on_message(delivered.append)
        self.adapter.handle_event(
            {"channel": "C1", "user": "BME", "text": "echo", "ts": "1"}
        )
        self.assertEqual(delivered, [])

    def test_missing_tokens_unconfigured_and_no_start(self) -> None:
        transport = FakeSlackTransport()
        adapter = SlackAdapter(secrets_get=lambda: {}, transport=transport)
        host = ChannelHost(":memory:")
        self.addCleanup(host.close)
        host.register(adapter, channel_id="slack")
        self.assertEqual(adapter.status()["state"], STATE_UNCONFIGURED)
        host.start("slack")
        self.assertFalse(transport.connected)
        self.assertEqual(host.get_channel("slack")["status"]["state"], STATE_UNCONFIGURED)
        self.assertEqual(host.get_channel("slack")["status"]["state"], STATE_UNCONFIGURED)

    def test_routing_rule_selects_project(self) -> None:
        rule = match_routing_rule("C-eng-frontend", self.adapter.routing_rules)
        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule["project_id"], "proj-eng")
        self.assertEqual(rule["card_type"], "task")
        bugs = match_routing_rule("C-bugs", self.adapter.routing_rules)
        assert bugs is not None
        self.assertEqual(bugs["project_id"], "proj-bugs")
        self.assertIsNone(match_routing_rule("C-random", self.adapter.routing_rules))

    def test_webhook_and_sockets_normalize_the_same(self) -> None:
        event = {
            "channel": "C1",
            "user": "U1",
            "text": "hi",
            "ts": "3.0",
            "thread_ts": "3.0",
        }
        sockets = self.adapter.normalize(event)
        webhook_adapter = SlackAdapter(
            secrets_get=lambda: _secrets(
                SLACK_BOT_TOKEN="xoxb-test",
                SLACK_WEBHOOK_URL="https://example.test/slack",
            ),
            transport=FakeSlackTransport(),
        )
        webhook = webhook_adapter.normalize(event)
        self.assertEqual(sockets["text"], webhook["text"])
        self.assertEqual(sockets["thread_ref"], webhook["thread_ref"])
        self.assertEqual(webhook_adapter.status().get("mode"), "webhook")

    def test_start_does_not_crash_when_already_stopped(self) -> None:
        self.host.stop("slack")
        self.host.stop("slack")
        self.assertEqual(self.host.get_channel("slack")["status"]["state"], STATE_STOPPED)


if __name__ == "__main__":
    unittest.main()
