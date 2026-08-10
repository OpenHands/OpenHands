"""Shared fixtures for mcp-webscan tests."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
WEBSCAN = Path(__file__).resolve().parents[1]
for path in (ROOT, WEBSCAN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PENTEST_SCOPE_ALLOWLIST", "example.com,*.lab.local")
    monkeypatch.setenv("PENTEST_AUTONOMY_MODE", "semi_autonomous")
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

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("X-Session-API-Key")
        if not auth:
            return httpx.Response(401, json={"detail": "Unauthorized"})
        body = json.loads(request.content.decode())
        self.posts.append(body)
        return httpx.Response(
            201,
            json={
                "id": str(uuid.uuid4()),
                "engagement_id": body["engagement_id"],
                "source_tool": body["source_tool"],
                "title": body["title"],
                "severity": body["severity"],
                "status": "new",
            },
        )
