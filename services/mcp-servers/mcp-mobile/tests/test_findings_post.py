"""AC-190-2 / AC-190-4 — findings post + workspace path guard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from shared.findings_client import FindingsClient
from tests.conftest import ENGAGEMENT_ID, FakeFindingsTransport
from tools.mobsf_static import run_mobsf_static


# @spec PROJETOSIN-190 — AC-190-2
@pytest.mark.asyncio
async def test_ac_190_2_mobsf_static_fixture_posts_finding(tmp_path: Path):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "mobsf_report_sample.json"
        ).read_text(encoding="utf-8")
    )

    async def runner(_apk: Path) -> dict[str, Any]:
        return {
            "hash": "fixturehash",
            "file_name": "sample.apk",
            "report": fixture,
        }

    transport = FakeFindingsTransport()
    client = FindingsClient(base_url="http://findings.test", transport=transport)
    raw = await run_mobsf_static(
        engagement_id=ENGAGEMENT_ID,
        apk_path="sample.apk",
        findings=client,
        runner=runner,
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["findings_count"] >= 1
    assert len(transport.posts) >= 1
    assert transport.posts[0]["source_tool"] == "mobsf"
    assert transport.posts[0]["severity"] in {
        "critical",
        "high",
        "medium",
        "low",
        "info",
    }


# @spec PROJETOSIN-190 — AC-190-4
@pytest.mark.asyncio
async def test_ac_190_4_path_outside_workspace_rejected(tmp_path: Path):
    transport = FakeFindingsTransport()
    client = FindingsClient(base_url="http://findings.test", transport=transport)
    outside = str(Path(os.environ["PENTEST_WORKSPACE_DIR"]).parent / "other.apk")
    raw = await run_mobsf_static(
        engagement_id=ENGAGEMENT_ID,
        apk_path=outside,
        findings=client,
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert body["error"] == "path_traversal"
    assert transport.posts == []


@pytest.mark.asyncio
async def test_missing_mobsf_key_returns_structured_error(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MOBSF_API_KEY", raising=False)
    transport = FakeFindingsTransport()
    client = FindingsClient(base_url="http://findings.test", transport=transport)

    async def boom(_apk: Path) -> dict[str, Any]:
        from mobsf_client import MobsfConfigError

        raise MobsfConfigError("MOBSF_API_KEY is unset (fail-closed)")

    raw = await run_mobsf_static(
        engagement_id=ENGAGEMENT_ID,
        apk_path="sample.apk",
        findings=client,
        runner=boom,
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert body["error"] == "mobsf_config"
    assert transport.posts == []
