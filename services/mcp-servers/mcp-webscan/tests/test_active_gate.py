"""AC-187-5/6 — confirmation gate for active tools."""

from __future__ import annotations

import inspect
import json

import pytest

from shared.confirmation import (
    ACTIVE_TOOLS,
    approve_confirmation,
    get_autonomy_mode,
    require_confirmation,
)
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
            findings=client,
        )
    )
    assert first["error"] == "confirmation_required"
    token = approve_confirmation(first["request_id"])

    second = json.loads(
        await run_zap_active(
            target="https://example.com",
            engagement_id=ENGAGEMENT_ID,
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
            {"target": "x"},
        )
    assert exc.value.code == "confirmation_required"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_high2_agent_cannot_bypass_gate_via_autonomy_arg(monkeypatch):
    """HIGH-2: autonomy is server-side only; LLM tool args must not skip gate."""
    monkeypatch.setenv("PENTEST_AUTONOMY_MODE", "semi_autonomous")
    assert get_autonomy_mode() == "semi_autonomous"

    from tools.zap_active import run_zap_active

    # Runners must not accept autonomy_mode from the agent.
    assert "autonomy_mode" not in inspect.signature(run_zap_active).parameters

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    body = json.loads(
        await run_zap_active(
            target="https://example.com",
            engagement_id=ENGAGEMENT_ID,
            findings=client,
        )
    )
    assert body["ok"] is False
    assert body["error"] == "confirmation_required"
    assert transport.posts == []


@pytest.mark.asyncio
async def test_high2_mcp_schema_omits_autonomy_mode():
    """HIGH-2: active MCP tools must not expose autonomy_mode in the schema."""
    from server import mcp

    tools = await mcp.list_tools()
    gated = {
        "web_zap_active_scan",
        "web_sqlmap_run",
        "web_nuclei_scan",
    }
    for tool in tools:
        if tool.name not in gated:
            continue
        props = (tool.inputSchema or {}).get("properties") or {}
        assert "autonomy_mode" not in props, tool.name


@pytest.mark.asyncio
async def test_high2_server_autonomous_env_skips_active_gate(monkeypatch):
    """Server-set autonomous mode may skip ACTIVE_TOOLS (MAX_RISK empty in MVP)."""
    monkeypatch.setenv("PENTEST_AUTONOMY_MODE", "autonomous")
    from tools.zap_active import run_zap_active

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    body = json.loads(
        await run_zap_active(
            target="https://example.com",
            engagement_id=ENGAGEMENT_ID,
            findings=client,
        )
    )
    assert body["ok"] is True
    assert len(transport.posts) >= 1
