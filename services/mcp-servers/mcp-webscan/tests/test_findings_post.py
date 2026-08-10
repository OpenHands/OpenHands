"""AC-187-3/4 — webscan findings post + scope."""

from __future__ import annotations

import json

import pytest

from shared.findings_client import FindingsClient
from tests.conftest import ENGAGEMENT_ID, FakeFindingsTransport


@pytest.mark.asyncio
async def test_ac_187_3_passive_zap_posts_finding():
    from tools.zap_passive import run_zap_passive

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    raw = await run_zap_passive(
        target="https://example.com",
        engagement_id=ENGAGEMENT_ID,
        findings=client,
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert len(transport.posts) == 1
    assert transport.posts[0]["source_tool"] == "zap"


@pytest.mark.asyncio
async def test_ac_187_4_nikto_out_of_scope_zero_posts():
    from tools.nikto import run_nikto

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    raw = await run_nikto(
        target="https://evil.not-allowed.test",
        engagement_id=ENGAGEMENT_ID,
        findings=client,
    )
    body = json.loads(raw)
    assert body["error"] == "scope_violation"
    assert transport.posts == []
