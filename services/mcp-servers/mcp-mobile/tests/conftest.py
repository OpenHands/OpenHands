"""Shared fixtures for mcp-mobile tests."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
MOBILE = Path(__file__).resolve().parents[1]
for path in (ROOT, MOBILE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("PENTEST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PENTEST_AUTONOMY_MODE", "semi_autonomous")
    monkeypatch.setenv("SESSION_API_KEY", "test-session-key")
    monkeypatch.setenv("FINDINGS_SERVICE_URL", "http://findings.test")
    monkeypatch.setenv("MOBSF_URL", "http://mobsf.test:8000")
    monkeypatch.setenv("MOBSF_API_KEY", "test-mobsf-key")
    monkeypatch.setenv("ADB_HOST", "android-emulator")
    monkeypatch.setenv("ADB_PORT", "5555")
    (tmp_path / "sample.apk").write_bytes(b"PK\x03\x04fake-apk")
    from shared.confirmation import clear_confirmation_state

    clear_confirmation_state()
    yield
    clear_confirmation_state()


ENGAGEMENT_ID = "00000000-0000-0000-0000-000000000190"


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


class FakeMobsfTransport(httpx.AsyncBaseTransport):
    """Minimal MobSF REST mock for upload/scan/report_json."""

    def __init__(self, report: dict | None = None):
        self.calls: list[tuple[str, str]] = []
        self.report = report or {
            "package_name": "com.example.app",
            "appsec": {
                "high": [
                    {
                        "title": "Mock high issue",
                        "description": "from FakeMobsfTransport",
                    }
                ]
            },
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        auth = request.headers.get("Authorization")
        if not auth:
            return httpx.Response(401, json={"error": "Unauthorized"})
        if path.endswith("/upload"):
            return httpx.Response(
                200,
                json={
                    "hash": "abc123hash",
                    "file_name": "sample.apk",
                    "scan_type": "apk",
                },
            )
        if path.endswith("/scan"):
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/report_json"):
            return httpx.Response(200, json=self.report)
        if path.endswith("/scorecard"):
            return httpx.Response(200, json={"security_score": 50})
        return httpx.Response(404, json={"error": "not found"})
