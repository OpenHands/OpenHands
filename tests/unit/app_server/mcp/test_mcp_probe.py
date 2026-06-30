"""Unit tests for MCP remote probe error categorization."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from openhands.app_server.mcp.mcp_probe import (
    _categorize_remote_error,
    headers_from_server_config,
    probe_remote_mcp_server,
)
from openhands.app_server.mcp.mcp_test_models import (
    MCPServerFailureCategory,
    McpServerTransport,
)


def test_headers_from_server_config():
    headers = headers_from_server_config(
        {
            'url': 'http://127.0.0.1:8012/mcp',
            'headers': {'Authorization': 'Bearer demo-secret-key'},
        }
    )
    assert headers == {'Authorization': 'Bearer demo-secret-key'}


def test_categorize_authentication_error():
    request = httpx.Request('GET', 'https://example.com/mcp')
    response = httpx.Response(401, request=request)
    category, message = _categorize_remote_error(
        httpx.HTTPStatusError('auth', request=request, response=response)
    )
    assert category == MCPServerFailureCategory.AUTHENTICATION
    assert '401' in message


def test_categorize_connection_error():
    category, _ = _categorize_remote_error(
        httpx.ConnectError(
            'connection refused', request=httpx.Request('GET', 'http://x')
        )
    )
    assert category == MCPServerFailureCategory.CONNECTION


@pytest.mark.asyncio
async def test_probe_remote_mcp_server_success(monkeypatch):
    class FakeClient:
        def __init__(self, transport):
            self.transport = transport

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_tools(self):
            return [{'name': 'tool_a'}, {'name': 'tool_b'}]

    monkeypatch.setattr('openhands.app_server.mcp.mcp_probe.Client', FakeClient)

    result = await probe_remote_mcp_server(
        url='https://example.com/mcp',
        headers={'Authorization': 'Bearer secret'},
        transport=McpServerTransport.SHTTP,
    )
    assert result.success is True
    assert result.tool_count == 2
    assert result.latency_ms is not None


@pytest.mark.asyncio
async def test_probe_remote_mcp_server_empty_tools(monkeypatch):
    class FakeClient:
        def __init__(self, transport):
            del transport

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_tools(self):
            return []

    monkeypatch.setattr('openhands.app_server.mcp.mcp_probe.Client', FakeClient)

    result = await probe_remote_mcp_server(
        url='https://example.com/mcp',
        headers=None,
        transport=McpServerTransport.SSE,
    )
    assert result.success is False
    assert result.category == MCPServerFailureCategory.TOOL_DISCOVERY


@pytest.mark.asyncio
async def test_probe_remote_mcp_server_timeout(monkeypatch):
    class FakeClient:
        def __init__(self, transport):
            del transport

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_tools(self):
            await asyncio.sleep(60)

    monkeypatch.setattr('openhands.app_server.mcp.mcp_probe.Client', FakeClient)

    result = await probe_remote_mcp_server(
        url='https://example.com/mcp',
        headers=None,
        transport=McpServerTransport.SHTTP,
    )
    assert result.success is False
    assert result.category == MCPServerFailureCategory.TIMEOUT
