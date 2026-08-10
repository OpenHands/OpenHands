"""AC-187-1 — mcp-recon tool contract / list_tools."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_ac_187_1_recon_exposes_at_least_three_tools():
    from server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert "recon_subfinder" in names
    assert "recon_httpx" in names
    assert "recon_reconftw" in names
    assert len(names) >= 3
    for tool in tools:
        assert tool.inputSchema is not None
        assert tool.inputSchema.get("type") == "object"
