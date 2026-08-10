"""AC-187-2 — mcp-webscan tool contract."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_ac_187_2_webscan_exposes_dast_tools():
    from server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    expected = {
        "web_zap_spider",
        "web_zap_passive_scan",
        "web_zap_active_scan",
        "web_nuclei_scan",
        "web_wapiti_scan",
        "web_nikto_scan",
        "web_sqlmap_run",
    }
    assert expected.issubset(names)
    for tool in tools:
        assert tool.inputSchema is not None
        assert tool.inputSchema.get("type") == "object"
        props = tool.inputSchema.get("properties") or {}
        assert "engagement_id" in props
