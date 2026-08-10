"""Shared fixtures for mcp-recon tests."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
RECON = Path(__file__).resolve().parents[1]
for path in (ROOT, RECON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PENTEST_SCOPE_ALLOWLIST", "example.com,*.lab.local,10.0.0.0/8")
    monkeypatch.setenv("SESSION_API_KEY", "test-session-key")
    monkeypatch.setenv("FINDINGS_SERVICE_URL", "http://findings.test")
    from shared.confirmation import clear_confirmation_state

    clear_confirmation_state()
    yield
    clear_confirmation_state()


ENGAGEMENT_ID = str(uuid.uuid4())


class FakeFindingsTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.posts: list[dict] = []
        self.status_code = 201
        self.force_auth_fail = False
        self._created: dict[str, str] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.force_auth_fail:
            return httpx.Response(401, json={"detail": "Unauthorized"})
        auth = request.headers.get("X-Session-API-Key")
        if not auth:
            return httpx.Response(401, json={"detail": "Unauthorized"})
        body = json.loads(request.content.decode())
        self.posts.append(body)
        key = f"{body.get('title')}|{body.get('asset')}|{body.get('endpoint')}"
        if key in self._created and self.status_code == 201:
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "detail": "Duplicate finding",
                        "existing_finding_id": self._created[key],
                    }
                },
            )
        fid = str(uuid.uuid4())
        self._created[key] = fid
        return httpx.Response(
            self.status_code,
            json={
                "id": fid,
                "engagement_id": body["engagement_id"],
                "source_tool": body["source_tool"],
                "title": body["title"],
                "severity": body["severity"],
                "status": "new",
            },
        )
