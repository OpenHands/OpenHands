import asyncio
import logging
from typing import Any

import httpx
from fastmcp import Client
from fastmcp.mcp_config import MCPConfig
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class MCPServerValidationResult(BaseModel):
    ok: bool
    error_kind: str | None = None
    message: str
    tools: list[str] = Field(default_factory=list)


def _safe_error_response(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return 'timeout', 'Timed out while connecting to the MCP server.'

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code in (401, 403):
            return 'auth', 'MCP server authentication failed.'
        return 'http_status', 'MCP server returned an error response.'

    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RequestError)):
        return 'connection', 'Could not connect to the MCP server.'

    if isinstance(exc, OSError):
        return 'connection', 'Could not connect to the MCP server.'

    return 'unknown', 'Could not validate the MCP server.'


async def validate_mcp_server_config(
    server_name: str,
    server_config: dict[str, Any],
    timeout: float = 5,
) -> MCPServerValidationResult:
    try:
        mcp_config = MCPConfig.model_validate(
            {'mcpServers': {server_name: server_config}}
        )
    except ValidationError:
        logger.info('Invalid MCP server config for %s', server_name)
        return MCPServerValidationResult(
            ok=False,
            error_kind='invalid_config',
            message='MCP server config is invalid.',
        )

    try:
        async with Client(
            mcp_config,
            timeout=timeout,
            init_timeout=timeout,
        ) as client:
            tools = await client.list_tools()
    except Exception as exc:
        error_kind, message = _safe_error_response(exc)
        logger.info('MCP validation failed for %s: %s', server_name, error_kind)
        return MCPServerValidationResult(
            ok=False,
            error_kind=error_kind,
            message=message,
        )

    return MCPServerValidationResult(
        ok=True,
        message='MCP server is reachable.',
        tools=[tool.name for tool in tools],
    )
