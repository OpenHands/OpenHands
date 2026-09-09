"""Tests for pre-assignment cost estimation and budget alerts."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kanban import KanbanStore  # noqa: E402
from kanban_api import handle_request  # noqa: E402
from cost_estimator import (  # noqa: E402
    BUDGET_OK,
    BUDGET_OVER,
    BUDGET_WARNING,
    apply_estimate,
    budget_status,
    estimate_cost_usd,
    estimate_task,
    record_actuals,
)


class EstimateLogicTests(unittest.TestCase):
    def test_higher_priority_costs_more(self) -> None:
        low = estimate_task(title="Fix typo", priority="P3")
        high = estimate_task(title="Fix typo", priority="P0")
        self.assertGreater(high["estimate_tokens"], low["estimate_tokens"])
        self.assertGreater(high["estimate_cost"], low["estimate_cost"])
        self.assertEqual(high["complexity"], "high")
        self.assertEqual(low["complexity"], "low")

    def test_longer_description_adds_tokens(self) -> None:
        short = estimate_task(title="Auth", description="login")
        long = estimate_task(title="Auth", description="x" * 900)
        self.assertGreater(long["estimate_tokens"], short["estimate_tokens"])
        self.assertEqual(long["complexity"], "high")

    def test_opus_is_more_expensive_than_sonnet(self) -> None:
        tokens = 10_000
        self.assertGreater(
            estimate_cost_usd(tokens, "claude-opus"),
            estimate_cost_usd(tokens, "claude-sonnet"),
        )

    def test_budget_warning_and_over_cap(self) -> None:
        ok = budget_status(1.0, 1.0, 10.0)
        warn = budget_status(7.0, 1.5, 10.0)
        over = budget_status(9.0, 2.0, 10.0)
        self.assertEqual(ok["status"], BUDGET_OK)
        self.assertEqual(warn["status"], BUDGET_WARNING)
        self.assertEqual(over["status"], BUDGET_OVER)

    def test_no_cap_is_ok(self) -> None:
        self.assertEqual(budget_status(50.0, 50.0, None)["status"], BUDGET_OK)


class StoreIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KanbanStore(":memory:")
        board = self.store.create_board("Work")
        self.column_id = board["columns"][0]["id"]
        self.board_id = board["id"]

    def tearDown(self) -> None:
        self.store.close()

    def test_apply_estimate_writes_card_fields(self) -> None:
        card = self.store.create_card(
            self.column_id,
            title="Ship login",
            description="Users can sign in",
            priority="P1",
        )
        result = apply_estimate(self.store, card["id"], model="claude-sonnet", cost_cap=25.0)
        updated = result["card"]
        self.assertGreater(updated["estimate_tokens"], 0)
        self.assertGreater(updated["estimate_cost"], 0)
        self.assertEqual(updated["model_used"], "claude-sonnet")
        self.assertEqual(result["budget"]["status"], BUDGET_OK)

    def test_record_actuals_compares_to_estimate(self) -> None:
        card = self.store.create_card(self.column_id, title="Ship login")
        apply_estimate(self.store, card["id"])
        estimated = self.store.get_card(card["id"])
        result = record_actuals(
            self.store,
            card["id"],
            actual_tokens=int(estimated["estimate_tokens"]) + 100,
            actual_cost=float(estimated["estimate_cost"]) + 0.5,
        )
        self.assertEqual(result["comparison"]["token_delta"], 100)
        self.assertTrue(result["comparison"]["over_estimate"])

    def test_estimate_route(self) -> None:
        card = self.store.create_card(self.column_id, title="Ship login")
        status, payload = handle_request(
            self.store,
            "POST",
            f"/api/cards/{card['id']}/estimate",
            {"model": "glm-5.2", "cost_cap": 5},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["estimate"]["model"], "glm-5.2")
        self.assertIn("budget", payload)


if __name__ == "__main__":
    unittest.main()
