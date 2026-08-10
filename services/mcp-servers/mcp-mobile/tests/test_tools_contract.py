"""AC-190-1 / AC-190-8 — mcp-mobile tool contract."""

from __future__ import annotations

import pytest

import server as mobile_server


# @spec PROJETOSIN-190 — AC-190-1
@pytest.mark.asyncio
async def test_ac_190_1_lists_at_least_eight_tools_with_schemas():
    tools = await mobile_server.mcp.list_tools()
    names = {t.name for t in tools}
    expected = set(mobile_server.list_tool_names())
    assert len(expected) >= 8
    assert expected.issubset(names)
    for tool in tools:
        if tool.name not in expected:
            continue
        assert tool.inputSchema is not None
        assert tool.inputSchema.get("type") == "object"


# @spec PROJETOSIN-190 — AC-190-8
def test_ac_190_8_required_capability():
    assert mobile_server.REQUIRED_CAPABILITY == "pentest.mobile.dynamic"
