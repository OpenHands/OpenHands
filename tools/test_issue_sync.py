"""Unit and API tests for optional JIRA/Linear/Plane kanban sync.

Run from the repo root:

    python3 tools/test_issue_sync.py
"""

from __future__ import annotations

import sys
import os
import unittest
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from issue_sync import (  # noqa: E402
    PROVIDER_KINDS,
    IssueSyncError,
    IssueSyncStore,
)
from issue_sync_api import handle_request  # noqa: E402
from kanban import KanbanStore  # noqa: E402


def _request(
    store: IssueSyncStore,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    return handle_request(store, method, path, body)


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[Any] = []

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append(
            {"method": method, "url": url, "headers": headers or {}, "body": body}
        )
        if self.responses:
            return self.responses.pop(0)
        return {
            "id": "ext-1",
            "key": "OH-1",
            "url": "https://example.test/OH-1",
            "title": (body or {}).get("title") or "Remote",
            "status": "Backlog",
        }


class IssueSyncStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kanban = KanbanStore(":memory:")
        self.transport = FakeTransport()
        self.store = IssueSyncStore(
            ":memory:",
            kanban_store=self.kanban,
            transport=self.transport,
        )
        self.board = self.kanban.create_board("Work")
        self.backlog_id = self.board["columns"][0]["id"]
        self.progress_id = self.board["columns"][1]["id"]
        self.card = self.kanban.create_card(self.backlog_id, title="Local card")

    def tearDown(self) -> None:
        self.store.close()
        self.kanban.close()

    def test_sync_is_noop_without_providers(self) -> None:
        result = self.store.sync()
        self.assertEqual(result["pushed"], 0)
        self.assertEqual(result["pulled"], 0)
        self.assertEqual(self.transport.calls, [])
        self.assertEqual(len(self.kanban.get_board(self.board["id"])["columns"][0]["cards"]), 1)

    def test_add_providers_and_mappings(self) -> None:
        jira = self.store.add_provider(
            "jira",
            name="Acme Jira",
            api_key="jira-key",
            base_url="https://acme.atlassian.net",
            project_key="OH",
        )
        linear = self.store.add_provider(
            "linear", name="Acme Linear", api_key="lin-key"
        )
        plane = self.store.add_provider(
            "plane",
            name="Acme Plane",
            api_key="plane-key",
            base_url="https://plane.example",
            workspace="acme",
            project_id="proj",
        )
        self.assertEqual(
            {item["kind"] for item in self.store.list_providers()},
            set(PROVIDER_KINDS),
        )
        mapping = self.store.set_mapping(
            jira["id"], local_column="In Progress", remote_status="In Progress"
        )
        self.assertEqual(mapping["local_column"], "In Progress")
        listed = self.store.list_mappings(jira["id"])
        self.assertEqual(len(listed), 1)
        self.assertEqual(linear["kind"], "linear")
        self.assertEqual(plane["kind"], "plane")

    def test_push_creates_remote_issue(self) -> None:
        provider = self.store.add_provider(
            "jira",
            name="Jira",
            api_key="k",
            base_url="https://jira.example",
            project_key="OH",
        )
        pushed = self.store.push_card(self.card["id"], provider["id"])
        self.assertEqual(pushed["remote_id"], "ext-1")
        self.assertEqual(pushed["remote_key"], "OH-1")
        self.assertEqual(self.transport.calls[0]["method"], "POST")
        self.assertIn("jira.example", self.transport.calls[0]["url"])

    def test_pull_creates_and_moves_cards(self) -> None:
        provider = self.store.add_provider(
            "linear", name="Linear", api_key="k"
        )
        self.store.set_mapping(
            provider["id"], local_column="In Progress", remote_status="In Progress"
        )
        self.transport.responses.append(
            [
                {
                    "id": "lin-9",
                    "key": "ENG-9",
                    "url": "https://linear.app/ENG-9",
                    "title": "Pulled bug",
                    "status": "In Progress",
                }
            ]
        )
        result = self.store.pull(provider["id"])
        self.assertEqual(result["pulled"], 1)
        board = self.kanban.get_board(self.board["id"])
        titles = [
            card["title"]
            for column in board["columns"]
            for card in column["cards"]
        ]
        self.assertIn("Pulled bug", titles)
        progress_titles = [card["title"] for card in board["columns"][1]["cards"]]
        self.assertIn("Pulled bug", progress_titles)

    def test_two_way_sync_and_create_from_finding(self) -> None:
        provider = self.store.add_provider(
            "jira",
            name="Jira",
            api_key="k",
            base_url="https://jira.example",
            project_key="OH",
        )
        self.store.set_mapping(
            provider["id"], local_column="Backlog", remote_status="Backlog"
        )
        self.transport.responses.append({"id": "ext-1", "key": "OH-1", "url": "u"})
        self.transport.responses.append([])
        result = self.store.sync(provider["id"], direction="both")
        self.assertGreaterEqual(result["pushed"], 1)

        finding = self.store.create_from_finding(
            provider["id"],
            title="TODO: flaky test",
            description="found in CI",
            kind="todo",
            board_id=self.board["id"],
        )
        self.assertEqual(finding["card"]["title"], "TODO: flaky test")
        self.assertEqual(finding["link"]["remote_id"], "ext-1")

    def test_unknown_provider_kind_rejected(self) -> None:
        with self.assertRaises(IssueSyncError):
            self.store.add_provider("github", name="Nope", api_key="k")


class IssueSyncApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kanban = KanbanStore(":memory:")
        self.transport = FakeTransport()
        self.store = IssueSyncStore(
            ":memory:",
            kanban_store=self.kanban,
            transport=self.transport,
        )
        self.board = self.kanban.create_board("Work")
        self.card = self.kanban.create_card(
            self.board["columns"][0]["id"], title="API card"
        )

    def tearDown(self) -> None:
        self.store.close()
        self.kanban.close()

    def test_provider_mapping_sync_and_push_endpoints(self) -> None:
        status, provider = _request(
            self.store,
            "POST",
            "/api/issue-sync/providers",
            {
                "kind": "jira",
                "name": "Jira",
                "api_key": "secret",
                "base_url": "https://jira.example",
                "project_key": "OH",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(provider["kind"], "jira")

        status, listed = _request(self.store, "GET", "/api/issue-sync/providers")
        self.assertEqual(status, 200)
        self.assertEqual(len(listed), 1)

        status, mapping = _request(
            self.store,
            "POST",
            "/api/issue-sync/mappings",
            {
                "provider_id": provider["id"],
                "local_column": "Review",
                "remote_status": "In Review",
            },
        )
        self.assertEqual(status, 201)
        status, mappings = _request(self.store, "GET", "/api/issue-sync/mappings")
        self.assertEqual(status, 200)
        self.assertEqual(mappings[0]["id"], mapping["id"])

        status, pushed = _request(
            self.store,
            "POST",
            f"/api/issue-sync/cards/{self.card['id']}/push",
            {"provider_id": provider["id"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(pushed["remote_key"], "OH-1")

        self.transport.responses.append([])
        status, synced = _request(
            self.store,
            "POST",
            "/api/issue-sync/sync",
            {"provider_id": provider["id"], "direction": "pull"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(synced["pulled"], 0)

    def test_missing_and_validation(self) -> None:
        status, payload = _request(
            self.store, "POST", "/api/issue-sync/providers", {"kind": "jira"}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        status, payload = _request(self.store, "GET", "/api/unknown")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
