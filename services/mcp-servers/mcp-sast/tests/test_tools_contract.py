"""AC-189-A1 / A2 / A3 — mcp-sast tool contract."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from shared.findings_client import FindingsClient  # noqa: E402
from tools.semgrep_scan import run_semgrep_scan  # noqa: E402
from tools.trivy_scan import run_trivy_scan  # noqa: E402
import server as sast_server  # noqa: E402


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = {
            "id": "00000000-0000-0000-0000-000000000001",
            "engagement_id": "00000000-0000-0000-0000-000000000099",
            "source_tool": "semgrep",
            "title": "ok",
            "severity": "high",
            "status": "new",
        }
        return httpx.Response(201, json=body)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PENTEST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("SESSION_API_KEY", "test-session-key")
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


# @spec PROJETOSIN-189 — AC-189-A1
def test_list_tools_exposes_semgrep_and_trivy():
    names = sast_server.list_tool_names()
    assert "sast_semgrep_scan" in names
    assert "sast_trivy_scan" in names
    assert sast_server.REQUIRED_CAPABILITY == "pentest.sast.run"


# @spec PROJETOSIN-189 — AC-189-A2
@pytest.mark.asyncio
async def test_semgrep_fixture_posts_finding(workspace: Path):
    fixture = json.loads(
        (HERE / "fixtures" / "semgrep_sample.json").read_text(encoding="utf-8")
    )

    async def runner(_path: Path, _config: str | None) -> dict[str, Any]:
        return fixture

    transport = _RecordingTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    raw = await run_semgrep_scan(
        engagement_id="00000000-0000-0000-0000-000000000099",
        path=".",
        findings=client,
        runner=runner,
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["findings_count"] >= 1
    assert len(transport.requests) >= 1
    posted = json.loads(transport.requests[0].content.decode("utf-8"))
    assert posted["source_tool"] == "semgrep"
    assert posted["severity"] == "high"  # ERROR → high


# @spec PROJETOSIN-189 — AC-189-A2
@pytest.mark.asyncio
async def test_trivy_stub_posts_finding(workspace: Path):
    transport = _RecordingTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    raw = await run_trivy_scan(
        engagement_id="00000000-0000-0000-0000-000000000099",
        target=".",
        findings=client,
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["findings_count"] >= 1
    assert len(transport.requests) >= 1
    posted = json.loads(transport.requests[0].content.decode("utf-8"))
    assert posted["source_tool"] == "trivy"
    assert posted["severity"] == "high"


# @spec PROJETOSIN-189 — AC-189-A3
@pytest.mark.asyncio
async def test_path_outside_workspace_errors_without_post(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    transport = _RecordingTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    outside = str(Path(os.environ["PENTEST_WORKSPACE_DIR"]).parent / "other")
    raw = await run_semgrep_scan(
        engagement_id="00000000-0000-0000-0000-000000000099",
        path=outside,
        findings=client,
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert body["error"] == "path_traversal"
    assert transport.requests == []


# @spec PROJETOSIN-189 — AC-189-A3
@pytest.mark.asyncio
async def test_trivy_path_traversal_no_post(workspace: Path):
    transport = _RecordingTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    raw = await run_trivy_scan(
        engagement_id="00000000-0000-0000-0000-000000000099",
        target="../../etc/passwd",
        findings=client,
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert body["error"] == "path_traversal"
    assert transport.requests == []
