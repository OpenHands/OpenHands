"""AC-187-5/6 — confirmation gate for active tools."""

from __future__ import annotations

import json

import pytest

from shared.confirmation import ACTIVE_TOOLS, approve_confirmation, require_confirmation
from shared.findings_client import FindingsClient
from tests.conftest import ENGAGEMENT_ID, FakeFindingsTransport


@pytest.mark.asyncio
async def test_ac_187_5_active_without_token_confirmation_required():
    from tools.sqlmap import run_sqlmap
    from tools.zap_active import run_zap_active

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )

    for runner in (run_zap_active, run_sqlmap):
        transport.posts.clear()
        raw = await runner(
            target="https://example.com",
            engagement_id=ENGAGEMENT_ID,
            autonomy_mode="semi_autonomous",
            findings=client,
        )
        body = json.loads(raw)
        assert body["ok"] is False
        assert body["error"] == "confirmation_required"
        assert body["request_id"]
        assert transport.posts == []


@pytest.mark.asyncio
async def test_ac_187_6_with_valid_token_executes_and_posts():
    from tools.zap_active import run_zap_active

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )

    first = json.loads(
        await run_zap_active(
            target="https://example.com",
            engagement_id=ENGAGEMENT_ID,
            autonomy_mode="semi_autonomous",
            findings=client,
        )
    )
    assert first["error"] == "confirmation_required"
    token = approve_confirmation(first["request_id"])

    second = json.loads(
        await run_zap_active(
            target="https://example.com",
            engagement_id=ENGAGEMENT_ID,
            autonomy_mode="semi_autonomous",
            confirmation_token=token,
            findings=client,
        )
    )
    assert second["ok"] is True
    assert len(transport.posts) >= 1
    assert transport.posts[0]["source_tool"] == "zap"


@pytest.mark.asyncio
async def test_ac_187_8_gate_unit():
    assert "zap_active_scan" in ACTIVE_TOOLS
    with pytest.raises(Exception) as exc:
        await require_confirmation(
            "zap_active_scan",
            "semi_autonomous",
            {"target": "x"},
        )
    assert exc.value.code == "confirmation_required"  # type: ignore[attr-defined]
