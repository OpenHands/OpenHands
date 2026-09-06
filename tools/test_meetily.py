"""Tests for Meetily transcript ingest."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kanban import KanbanStore  # noqa: E402
from meetily import (  # noqa: E402
    CARD_TYPE_BUG,
    CARD_TYPE_DISCUSSION,
    CARD_TYPE_FEATURE,
    CARD_TYPE_REFINEMENT,
    CARD_TYPE_TASK,
    MeetilyService,
    RouterExtractor,
    extract_actions,
    parse_transcript,
    type_text,
)
from meetily_api import handle_request  # noqa: E402


class FakeExtractor:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self.payload = payload
        self.called = False

    def summarize_and_extract(self, utterances: list[dict[str, str]]) -> dict[str, Any] | None:
        self.called = True
        self.utterances = utterances
        return self.payload


JSON_TRANSCRIPT = """
{
  "utterances": [
    {"speaker": "Ada", "time": "00:01", "text": "We found a crash bug in login."},
    {"speaker": "Lin", "time": "00:02", "text": "Feature request: add SSO."}
  ]
}
"""

MD_TRANSCRIPT = """
# Notes
**Ada** (00:01): Please file a bug for the crash.
**Lin** (00:02): Feature request: export CSV
"""

CSV_TRANSCRIPT = """speaker,time,text
Ada,00:01,Discuss whether we should rewrite the cache
Lin,00:02,Chore: polish the settings copy
"""


class MeetilyParseTests(unittest.TestCase):
    def test_parses_json_markdown_csv(self) -> None:
        json_rows = parse_transcript(JSON_TRANSCRIPT)
        self.assertEqual(json_rows[0]["speaker"], "Ada")
        self.assertIn("crash", json_rows[0]["text"])
        md_rows = parse_transcript(MD_TRANSCRIPT)
        self.assertTrue(any("crash" in row["text"] for row in md_rows))
        csv_rows = parse_transcript(CSV_TRANSCRIPT)
        self.assertEqual(csv_rows[0]["speaker"], "Ada")

    def test_typing_rules(self) -> None:
        self.assertEqual(type_text("crash bug in login"), CARD_TYPE_BUG)
        self.assertEqual(type_text("Feature request: add SSO"), CARD_TYPE_FEATURE)
        self.assertEqual(type_text("Discuss whether we should"), CARD_TYPE_DISCUSSION)
        self.assertEqual(type_text("Chore: polish the copy"), CARD_TYPE_REFINEMENT)
        self.assertEqual(type_text("Todo: write the runbook"), CARD_TYPE_TASK)

    def test_fuzzy_dedup_skips_near_duplicate(self) -> None:
        store = KanbanStore(":memory:")
        board = store.create_board("Meetings")
        service = MeetilyService(store)
        first = service.ingest(JSON_TRANSCRIPT, board_id=board["id"])
        self.assertGreaterEqual(len(first["created"]), 1)
        second = service.ingest(JSON_TRANSCRIPT, board_id=board["id"])
        self.assertEqual(second["created"], [])
        self.assertGreaterEqual(len(second["duplicates"]), 1)
        self.assertEqual(
            second["duplicates"][0]["existing_card_id"],
            first["created"][0]["id"],
        )

    def test_llm_extractor_and_fallback(self) -> None:
        utterances = parse_transcript(JSON_TRANSCRIPT)
        fake = FakeExtractor(
            {
                "summary": "Auth work",
                "items": [{"title": "Fix login crash", "card_type": "bug"}],
            }
        )
        llm = extract_actions(utterances, fake)
        self.assertTrue(fake.called)
        self.assertEqual(llm["source"], "llm")
        self.assertEqual(llm["items"][0]["card_type"], CARD_TYPE_BUG)
        fallback = extract_actions(utterances, FakeExtractor(None))
        self.assertEqual(fallback["source"], "deterministic")
        self.assertGreaterEqual(len(fallback["items"]), 1)
        missing_provider = extract_actions(utterances, RouterExtractor())
        self.assertEqual(missing_provider["source"], "deterministic")

    def test_webhook_accepts_utterances(self) -> None:
        store = KanbanStore(":memory:")
        board = store.create_board("Meetings")
        service = MeetilyService(store)
        status, payload = handle_request(
            service,
            "POST",
            "/api/meetings/webhook",
            {
                "board_id": board["id"],
                "utterances": [
                    {"speaker": "Ada", "time": "00:01", "text": "Todo: write the runbook"}
                ],
            },
        )
        self.assertEqual(status, 201)
        self.assertGreaterEqual(len(payload["created"]), 1)

    def test_action_api_does_not_create_cards(self) -> None:
        store = KanbanStore(":memory:")
        service = MeetilyService(store)
        status, payload = handle_request(
            service,
            "POST",
            "/api/meetings/action",
            {"text": JSON_TRANSCRIPT},
        )
        self.assertEqual(status, 200)
        self.assertIn("items", payload)
        self.assertEqual(store.list_boards(), [])


class MeetilyCliTests(unittest.TestCase):
    def test_cli_imports_file(self) -> None:
        import cli_meetily

        store = KanbanStore(":memory:")
        board = store.create_board("Meetings")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(JSON_TRANSCRIPT)
            path = handle.name
        result = cli_meetily.import_transcript(path, store=store, board_id=board["id"])
        self.assertGreaterEqual(len(result["created"]), 1)


if __name__ == "__main__":
    unittest.main()
