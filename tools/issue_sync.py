"""Optional two-way kanban sync with JIRA, Linear, and Plane.

Local kanban does not depend on this module. Configure a provider per
project when external tickets should track cards. HTTP calls go through
an injectable transport so tests never hit the network.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from kanban import KanbanStore, new_id, utc_now

ISSUE_SYNC_DB_FILENAME = "issue_sync.sqlite"
PROVIDER_KINDS = ("jira", "linear", "plane")
FINDING_KINDS = ("bug", "todo")
SYNC_DIRECTIONS = ("push", "pull", "both")
DEFAULT_LINEAR_URL = "https://api.linear.app/graphql"
DEFAULT_JIRA_ISSUE_TYPE = "Task"


class IssueSyncError(Exception):
    """Raised for invalid issue-sync operations."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NotFoundError(IssueSyncError):
    """Raised when a provider, mapping, or card link does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=404)


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any: ...


class UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        payload = json.dumps(body).encode() if body is not None else None
        request = Request(url, data=payload, headers=headers or {}, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise IssueSyncError(f"Remote {exc.code}: {detail}") from exc
        except URLError as exc:
            raise IssueSyncError(f"Remote request failed: {exc.reason}") from exc
        return json.loads(raw) if raw else {}


def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".openhands", "agent-canvas")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, ISSUE_SYNC_DB_FILENAME)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _parse_config(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


class IssueSyncStore:
    """Provider config, column mappings, and push/pull against kanban cards."""

    def __init__(
        self,
        db_path: str = ":memory:",
        kanban_store: KanbanStore | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.db_path = db_path
        self.kanban_store = kanban_store
        self.transport = transport or UrllibTransport()
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS providers (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                api_key TEXT NOT NULL,
                base_url TEXT,
                config TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mappings (
                id TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                local_column TEXT NOT NULL,
                remote_status TEXT NOT NULL,
                FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE,
                UNIQUE (provider_id, local_column)
            );
            CREATE TABLE IF NOT EXISTS card_links (
                id TEXT PRIMARY KEY,
                card_id TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                remote_id TEXT NOT NULL,
                remote_key TEXT,
                remote_url TEXT,
                UNIQUE (card_id, provider_id)
            );
            """
        )
        self.conn.commit()

    def add_provider(
        self,
        kind: str,
        name: str,
        api_key: str,
        base_url: str | None = None,
        **config: Any,
    ) -> dict[str, Any]:
        kind = (kind or "").strip()
        if kind not in PROVIDER_KINDS:
            raise IssueSyncError(
                f"kind must be one of {', '.join(PROVIDER_KINDS)}"
            )
        name = (name or "").strip()
        api_key = (api_key or "").strip()
        if not name:
            raise IssueSyncError("name is required")
        if not api_key:
            raise IssueSyncError("api_key is required")
        if kind == "jira" and not (base_url or "").strip():
            raise IssueSyncError("JIRA base_url is required")
        if kind == "linear":
            base_url = (base_url or "").strip() or DEFAULT_LINEAR_URL
        provider_id = new_id()
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO providers (
                    id, kind, name, api_key, base_url, config, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_id,
                    kind,
                    name,
                    api_key,
                    (base_url or "").strip() or None,
                    json.dumps(config),
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self.get_provider(provider_id)

    def list_providers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM providers ORDER BY created_at ASC"
            ).fetchall()
        return [self.get_provider(row["id"]) for row in rows]

    def get_provider(self, provider_id: str) -> dict[str, Any]:
        with self._lock:
            provider = _row_to_dict(
                self.conn.execute(
                    "SELECT * FROM providers WHERE id = ?", (provider_id,)
                ).fetchone()
            )
        if provider is None:
            raise NotFoundError(f"Provider {provider_id} not found")
        provider["config"] = _parse_config(provider.get("config"))
        return provider

    def set_mapping(
        self,
        provider_id: str,
        local_column: str,
        remote_status: str,
    ) -> dict[str, Any]:
        self.get_provider(provider_id)
        local_column = (local_column or "").strip()
        remote_status = (remote_status or "").strip()
        if not local_column or not remote_status:
            raise IssueSyncError("local_column and remote_status are required")
        mapping_id = new_id()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO mappings (id, provider_id, local_column, remote_status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider_id, local_column) DO UPDATE SET
                    remote_status = excluded.remote_status
                """,
                (mapping_id, provider_id, local_column, remote_status),
            )
            self.conn.commit()
            row = self.conn.execute(
                """
                SELECT * FROM mappings
                WHERE provider_id = ? AND local_column = ?
                """,
                (provider_id, local_column),
            ).fetchone()
        return _row_to_dict(row)  # type: ignore[return-value]

    def list_mappings(self, provider_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if provider_id:
                rows = self.conn.execute(
                    "SELECT * FROM mappings WHERE provider_id = ? ORDER BY local_column",
                    (provider_id,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM mappings ORDER BY local_column"
                ).fetchall()
        return [_row_to_dict(row) for row in rows]  # type: ignore[misc]

    def push_card(self, card_id: str, provider_id: str) -> dict[str, Any]:
        if self.kanban_store is None:
            raise IssueSyncError("kanban store is required")
        provider = self.get_provider(provider_id)
        card = self.kanban_store.get_card(card_id)
        column = self.kanban_store.get_column(card["column_id"])
        remote_status = self._remote_status(provider_id, column["name"])
        existing = self._link_for(card_id, provider_id)
        payload = self._create_or_update_remote(
            provider, card, remote_status, existing
        )
        return self._upsert_link(card_id, provider_id, payload)

    def pull(self, provider_id: str) -> dict[str, Any]:
        if self.kanban_store is None:
            raise IssueSyncError("kanban store is required")
        provider = self.get_provider(provider_id)
        issues = self._list_remote_issues(provider)
        pulled = 0
        for issue in issues:
            self._apply_remote_issue(provider, issue)
            pulled += 1
        return {"pulled": pulled}

    def sync(
        self,
        provider_id: str | None = None,
        direction: str = "both",
    ) -> dict[str, Any]:
        direction = (direction or "both").strip()
        if direction not in SYNC_DIRECTIONS:
            raise IssueSyncError(
                f"direction must be one of {', '.join(SYNC_DIRECTIONS)}"
            )
        providers = (
            [self.get_provider(provider_id)]
            if provider_id
            else self.list_providers()
        )
        pushed = 0
        pulled = 0
        for provider in providers:
            if direction in {"push", "both"}:
                pushed += self._push_board(provider["id"])
            if direction in {"pull", "both"}:
                pulled += self.pull(provider["id"])["pulled"]
        return {"pushed": pushed, "pulled": pulled}

    def create_from_finding(
        self,
        provider_id: str,
        title: str,
        description: str = "",
        kind: str = "bug",
        board_id: str | None = None,
    ) -> dict[str, Any]:
        if self.kanban_store is None:
            raise IssueSyncError("kanban store is required")
        kind = (kind or "bug").strip()
        if kind not in FINDING_KINDS:
            raise IssueSyncError(f"kind must be one of {', '.join(FINDING_KINDS)}")
        title = (title or "").strip()
        if not title:
            raise IssueSyncError("title is required")
        board = self._target_board(self.get_provider(provider_id), board_id)
        column_id = board["columns"][0]["id"]
        card = self.kanban_store.create_card(
            column_id,
            title,
            description=description,
            priority="P1" if kind == "bug" else "P2",
        )
        link = self.push_card(card["id"], provider_id)
        return {"card": self.kanban_store.get_card(card["id"]), "link": link}

    def _push_board(self, provider_id: str) -> int:
        provider = self.get_provider(provider_id)
        board = self._target_board(provider)
        count = 0
        for column in board["columns"]:
            for card in column["cards"]:
                self.push_card(card["id"], provider_id)
                count += 1
        return count

    def _target_board(
        self, provider: dict[str, Any], board_id: str | None = None
    ) -> dict[str, Any]:
        assert self.kanban_store is not None
        resolved = board_id or provider["config"].get("board_id")
        if resolved:
            return self.kanban_store.get_board(str(resolved))
        boards = self.kanban_store.list_boards()
        if not boards:
            raise IssueSyncError("No kanban board to sync")
        return self.kanban_store.get_board(boards[0]["id"])

    def _remote_status(self, provider_id: str, local_column: str) -> str:
        for mapping in self.list_mappings(provider_id):
            if mapping["local_column"] == local_column:
                return mapping["remote_status"]
        return local_column

    def _local_column(self, provider_id: str, remote_status: str) -> str:
        for mapping in self.list_mappings(provider_id):
            if mapping["remote_status"] == remote_status:
                return mapping["local_column"]
        return remote_status

    def _link_for(self, card_id: str, provider_id: str) -> dict[str, Any] | None:
        with self._lock:
            return _row_to_dict(
                self.conn.execute(
                    """
                    SELECT * FROM card_links
                    WHERE card_id = ? AND provider_id = ?
                    """,
                    (card_id, provider_id),
                ).fetchone()
            )

    def _upsert_link(
        self, card_id: str, provider_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        existing = self._link_for(card_id, provider_id)
        link_id = existing["id"] if existing else new_id()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO card_links (
                    id, card_id, provider_id, remote_id, remote_key, remote_url
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id, provider_id) DO UPDATE SET
                    remote_id = excluded.remote_id,
                    remote_key = excluded.remote_key,
                    remote_url = excluded.remote_url
                """,
                (
                    link_id,
                    card_id,
                    provider_id,
                    payload["remote_id"],
                    payload.get("remote_key"),
                    payload.get("remote_url"),
                ),
            )
            self.conn.commit()
        return self._link_for(card_id, provider_id)  # type: ignore[return-value]

    def _apply_remote_issue(
        self, provider: dict[str, Any], issue: dict[str, Any]
    ) -> None:
        assert self.kanban_store is not None
        board = self._target_board(provider)
        column_name = self._local_column(provider["id"], issue["status"])
        column = next(
            (item for item in board["columns"] if item["name"] == column_name),
            board["columns"][0],
        )
        with self._lock:
            existing = _row_to_dict(
                self.conn.execute(
                    """
                    SELECT * FROM card_links
                    WHERE provider_id = ? AND remote_id = ?
                    """,
                    (provider["id"], issue["id"]),
                ).fetchone()
            )
        if existing:
            self.kanban_store.update_card(existing["card_id"], title=issue["title"])
            card = self.kanban_store.get_card(existing["card_id"])
            if card["column_id"] != column["id"]:
                self.kanban_store.move_card(existing["card_id"], column["id"], 0)
            return
        card = self.kanban_store.create_card(column["id"], issue["title"])
        self._upsert_link(
            card["id"],
            provider["id"],
            {
                "remote_id": issue["id"],
                "remote_key": issue.get("key"),
                "remote_url": issue.get("url"),
            },
        )

    def _create_or_update_remote(
        self,
        provider: dict[str, Any],
        card: dict[str, Any],
        remote_status: str,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        url, body, headers = self._write_request(
            provider, card, remote_status, existing
        )
        payload = self.transport.request("POST", url, headers=headers, body=body)
        return self._normalize_created(provider["kind"], payload, card)

    def _list_remote_issues(self, provider: dict[str, Any]) -> list[dict[str, Any]]:
        url, body, headers, method = self._list_request(provider)
        payload = self.transport.request(method, url, headers=headers, body=body)
        return self._normalize_issues(provider["kind"], payload)

    def _auth_headers(self, provider: dict[str, Any]) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if provider["kind"] == "jira":
            headers["Authorization"] = f"Bearer {provider['api_key']}"
        elif provider["kind"] == "linear":
            headers["Authorization"] = provider["api_key"]
        else:
            headers["X-API-Key"] = provider["api_key"]
        return headers

    def _write_request(
        self,
        provider: dict[str, Any],
        card: dict[str, Any],
        remote_status: str,
        existing: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        headers = self._auth_headers(provider)
        kind = provider["kind"]
        title = card["title"]
        description = card.get("description") or ""
        if kind == "jira":
            base = str(provider["base_url"]).rstrip("/")
            return (
                f"{base}/rest/api/3/issue",
                {
                    "fields": {
                        "project": {"key": provider["config"].get("project_key")},
                        "summary": title,
                        "description": description,
                        "issuetype": {"name": DEFAULT_JIRA_ISSUE_TYPE},
                    },
                    "title": title,
                    "status": remote_status,
                    "existing_id": existing["remote_id"] if existing else None,
                },
                headers,
            )
        if kind == "linear":
            return (
                str(provider["base_url"]),
                {
                    "query": "mutation($title: String!) { issueCreate(input: {title: $title}) { issue { id identifier url } } }",
                    "variables": {"title": title},
                    "title": title,
                    "status": remote_status,
                },
                headers,
            )
        base = str(provider["base_url"] or "").rstrip("/")
        workspace = provider["config"].get("workspace")
        project_id = provider["config"].get("project_id")
        return (
            f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/",
            {"name": title, "description": description, "title": title, "status": remote_status},
            headers,
        )

    def _list_request(
        self, provider: dict[str, Any]
    ) -> tuple[str, dict[str, Any] | None, dict[str, str], str]:
        headers = self._auth_headers(provider)
        kind = provider["kind"]
        if kind == "jira":
            base = str(provider["base_url"]).rstrip("/")
            query = urlencode(
                {"jql": f"project={provider['config'].get('project_key', '')}"}
            )
            return f"{base}/rest/api/3/search?{query}", None, headers, "GET"
        if kind == "linear":
            return (
                str(provider["base_url"]),
                {"query": "{ issues { nodes { id identifier title url state { name } } } }"},
                headers,
                "POST",
            )
        base = str(provider["base_url"] or "").rstrip("/")
        workspace = provider["config"].get("workspace")
        project_id = provider["config"].get("project_id")
        return (
            f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/",
            None,
            headers,
            "GET",
        )

    def _normalize_created(
        self, kind: str, payload: Any, card: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        if kind == "linear":
            issue = (
                payload.get("data", {})
                .get("issueCreate", {})
                .get("issue", payload)
            )
            return {
                "remote_id": str(issue.get("id") or payload.get("id") or new_id()),
                "remote_key": issue.get("identifier") or payload.get("key"),
                "remote_url": issue.get("url") or payload.get("url"),
            }
        return {
            "remote_id": str(payload.get("id") or new_id()),
            "remote_key": payload.get("key") or payload.get("identifier"),
            "remote_url": payload.get("url") or payload.get("html_url") or payload.get("self"),
        }

    def _normalize_issues(self, kind: str, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("issues"), list):
            items = payload["issues"]
        elif isinstance(payload, dict):
            nodes = (
                payload.get("data", {})
                .get("issues", {})
                .get("nodes")
            )
            items = nodes if isinstance(nodes, list) else payload.get("results") or []
        else:
            items = []
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
            status = item.get("status")
            if isinstance(status, dict):
                status = status.get("name")
            state = item.get("state")
            if isinstance(state, dict):
                status = status or state.get("name")
            normalized.append(
                {
                    "id": str(item.get("id") or item.get("key") or new_id()),
                    "key": item.get("key") or item.get("identifier"),
                    "url": item.get("url") or item.get("html_url") or item.get("self"),
                    "title": item.get("title")
                    or item.get("name")
                    or fields.get("summary")
                    or "Untitled",
                    "status": status or fields.get("status", {}).get("name")
                    if isinstance(fields.get("status"), dict)
                    else status or "Backlog",
                }
            )
        return normalized
