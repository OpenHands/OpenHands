"""Remote MCP server probing (SSE / SHTTP) from the app-server."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastmcp import Client
from fastmcp.client.transports import SSETransport, StreamableHttpTransport

from openhands.app_server.mcp.mcp_test_models import (
    McpProbeResult,
    MCPServerFailureCategory,
    McpServerTransport,
)
from openhands.app_server.utils.logger import openhands_logger as logger

REMOTE_PROBE_TIMEOUT_S = 30.0


def _sanitize_message(message: str) -> str:
    lowered = message.lower()
    for token in ('bearer ', 'api_key', 'apikey', 'authorization', 'token'):
        if token in lowered:
            return 'Connection failed (details redacted)'
    if len(message) > 500:
        return message[:500] + '...'
    return message


def _categorize_remote_error(exc: Exception) -> tuple[MCPServerFailureCategory, str]:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return MCPServerFailureCategory.TIMEOUT, 'Connection timed out'

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return (
                MCPServerFailureCategory.AUTHENTICATION,
                f'Authentication failed (HTTP {status})',
            )
        if status >= 500:
            return (
                MCPServerFailureCategory.CONNECTION,
                f'Remote server error (HTTP {status})',
            )
        return (
            MCPServerFailureCategory.PROTOCOL,
            f'Unexpected HTTP response (HTTP {status})',
        )

    if isinstance(exc, httpx.RequestError):
        return MCPServerFailureCategory.CONNECTION, _sanitize_message(str(exc))

    message = _sanitize_message(str(exc) or exc.__class__.__name__)
    lowered = message.lower()
    if 'auth' in lowered or '401' in lowered or '403' in lowered:
        return MCPServerFailureCategory.AUTHENTICATION, message
    if 'connect' in lowered or 'refused' in lowered or 'resolve' in lowered:
        return MCPServerFailureCategory.CONNECTION, message
    if 'tool' in lowered:
        return MCPServerFailureCategory.TOOL_DISCOVERY, message
    return MCPServerFailureCategory.PROTOCOL, message


async def probe_remote_mcp_server(
    *,
    url: str,
    headers: dict[str, str] | None,
    transport: McpServerTransport,
) -> McpProbeResult:
    """Connect to a remote MCP server and list tools."""
    started = time.monotonic()
    try:
        async with asyncio.timeout(REMOTE_PROBE_TIMEOUT_S):
            if transport == McpServerTransport.SSE:
                http_transport: SSETransport | StreamableHttpTransport = SSETransport(
                    url=url, headers=headers
                )
            else:
                http_transport = StreamableHttpTransport(url=url, headers=headers)
            async with Client(transport=http_transport) as client:
                tools = await client.list_tools()
        tool_count = len(tools or [])
        latency_ms = int((time.monotonic() - started) * 1000)
        if tool_count == 0:
            return McpProbeResult(
                success=False,
                tool_count=0,
                latency_ms=latency_ms,
                category=MCPServerFailureCategory.TOOL_DISCOVERY,
                message='Connected but no tools were returned',
            )
        return McpProbeResult(
            success=True,
            tool_count=tool_count,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.debug('Remote MCP probe failed for %s: %s', url, exc, exc_info=True)
        category, message = _categorize_remote_error(exc)
        latency_ms = int((time.monotonic() - started) * 1000)
        return McpProbeResult(
            success=False,
            latency_ms=latency_ms,
            category=category,
            message=message,
        )


def headers_from_server_config(server_config: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    raw_headers = server_config.get('headers')
    if isinstance(raw_headers, dict):
        for key, value in raw_headers.items():
            if isinstance(key, str) and isinstance(value, str):
                headers[key] = value
    return headers
