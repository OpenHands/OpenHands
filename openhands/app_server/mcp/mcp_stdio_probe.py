"""Stdio MCP server probing via an ephemeral agent-server sandbox."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastmcp.mcp_config import MCPConfig, StdioMCPServer
from pydantic import SecretStr

from openhands.app_server.mcp.mcp_test_models import (
    McpProbeResult,
    MCPServerFailureCategory,
)
from openhands.app_server.sandbox.sandbox_models import AGENT_SERVER, SandboxInfo
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.conversation.request import StartConversationRequest
from openhands.sdk.llm import LLM
from openhands.sdk.settings import ConversationSettings, OpenHandsAgentSettings
from openhands.sdk.workspace import LocalWorkspace

STDIO_PROBE_TIMEOUT_S = 90.0
SANDBOX_READY_TIMEOUT_S = 60.0
EVENT_POLL_TIMEOUT_S = 75.0
DEFAULT_PROBE_WORKSPACE_DIR = '/workspace/project'
# MCP stdio probes only verify tool registration; they never call the LLM.
# Use a model with a known >=16k context window so user LLM settings cannot
# block the probe (e.g. local models configured with 8192 context).
PROBE_LLM_MODEL = 'gpt-4o'
PROBE_LLM = LLM(model=PROBE_LLM_MODEL, api_key=SecretStr('mcp-probe-placeholder'))


def _agent_server_url(sandbox: SandboxInfo) -> str | None:
    if not sandbox.exposed_urls:
        return None
    for exposed in sandbox.exposed_urls:
        if exposed.name == AGENT_SERVER:
            return exposed.url.rstrip('/')
    return sandbox.exposed_urls[0].url.rstrip('/')


def _tool_name_from_event_tool(tool: dict[str, Any]) -> str | None:
    """Resolve a tool name from a serialized SystemPromptEvent tool entry."""
    title = tool.get('title')
    if isinstance(title, str) and title:
        return title
    mcp_tool = tool.get('mcp_tool') or {}
    name = (
        mcp_tool.get('name')
        or tool.get('name')
        or (tool.get('function') or {}).get('name')
    )
    if isinstance(name, str) and name:
        return name
    return None


def _latest_system_prompt_tools(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    system_prompt_events = [
        event for event in events if event.get('kind') == 'SystemPromptEvent'
    ]
    if not system_prompt_events:
        return []
    tools = system_prompt_events[-1].get('tools') or []
    return [tool for tool in tools if isinstance(tool, dict)]


def _tool_names_from_events(events: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from the latest SystemPromptEvent in a page of events."""
    names: list[str] = []
    for tool in _latest_system_prompt_tools(events):
        name = _tool_name_from_event_tool(tool)
        if name:
            names.append(name)
    return names


def _mcp_tool_names_from_events(events: list[dict[str, Any]]) -> list[str]:
    """Extract MCP tool names from the latest SystemPromptEvent."""
    names: list[str] = []
    for tool in _latest_system_prompt_tools(events):
        is_mcp = tool.get('kind') == 'MCPToolDefinition' or tool.get('mcp_tool')
        if not is_mcp:
            continue
        name = _tool_name_from_event_tool(tool)
        if name:
            names.append(name)
    return names


def _build_conversation_body(
    agent_settings: OpenHandsAgentSettings,
    server_id: str,
    server_dump: dict[str, Any],
    conversation_id: UUID,
    workspace_working_dir: str,
) -> dict[str, Any]:
    mcp_config = MCPConfig(
        mcpServers={server_id: StdioMCPServer.model_validate(server_dump)}
    )
    configured = agent_settings.model_copy(
        update={
            'mcp_config': mcp_config,
            'llm': PROBE_LLM,
        }
    )
    agent = configured.create_agent()
    conv_settings = ConversationSettings(
        conversation_id=conversation_id,
        workspace=LocalWorkspace(working_dir=workspace_working_dir),
    )
    request = conv_settings.create_request(
        StartConversationRequest,
        agent=agent,
        conversation_id=conversation_id,
    )
    return request.model_dump(mode='json', context={'expose_secrets': True})


async def _initialize_probe_conversation(
    *,
    httpx_client: httpx.AsyncClient,
    agent_server_url: str,
    conversation_id: UUID,
    headers: dict[str, str],
) -> None:
    """Trigger agent init so SystemPromptEvent (with MCP tools) is emitted.

    Conversation creation alone does not call ``init_state``; a no-run message
    forces MCP tool discovery without invoking the LLM.
    """
    response = await httpx_client.post(
        f'{agent_server_url}/api/conversations/{conversation_id.hex}/events',
        json={
            'role': 'user',
            'content': [{'type': 'text', 'text': 'ping'}],
            'run': False,
        },
        headers=headers,
        timeout=SANDBOX_READY_TIMEOUT_S,
    )
    response.raise_for_status()


async def probe_stdio_mcp_server(
    *,
    sandbox_service: SandboxService,
    sandbox_spec_service: SandboxSpecService,
    httpx_client: httpx.AsyncClient,
    agent_settings: OpenHandsAgentSettings,
    server_id: str,
    server_dump: dict[str, Any],
) -> McpProbeResult:
    """Start an ephemeral sandbox and verify stdio MCP tools appear in the system prompt."""
    started = time.monotonic()
    sandbox: SandboxInfo | None = None
    conversation_id = uuid4()
    try:
        async with asyncio.timeout(STDIO_PROBE_TIMEOUT_S):
            sandbox = await sandbox_service.start_sandbox()
            sandbox_id = sandbox.id
            ready = await sandbox_service.wait_for_sandbox_running(
                sandbox_id,
                timeout=int(SANDBOX_READY_TIMEOUT_S),
                httpx_client=httpx_client,
            )
            if not ready:
                return McpProbeResult(
                    success=False,
                    sandbox_id=sandbox_id,
                    category=MCPServerFailureCategory.SANDBOX,
                    message='Sandbox failed to become ready',
                    latency_ms=int((time.monotonic() - started) * 1000),
                )

            sandbox = await sandbox_service.get_sandbox(sandbox_id)
            if sandbox is None:
                return McpProbeResult(
                    success=False,
                    sandbox_id=sandbox_id,
                    category=MCPServerFailureCategory.SANDBOX,
                    message='Sandbox not found after startup',
                    latency_ms=int((time.monotonic() - started) * 1000),
                )

            agent_server_url = _agent_server_url(sandbox)
            session_api_key = sandbox.session_api_key
            if not agent_server_url or not session_api_key:
                return McpProbeResult(
                    success=False,
                    sandbox_id=sandbox_id,
                    category=MCPServerFailureCategory.SANDBOX,
                    message='Sandbox is missing agent-server URL or session key',
                    latency_ms=int((time.monotonic() - started) * 1000),
                )

            sandbox_spec = await sandbox_spec_service.get_sandbox_spec(
                sandbox.sandbox_spec_id
            )
            workspace_working_dir = (
                sandbox_spec.working_dir
                if sandbox_spec and sandbox_spec.working_dir
                else DEFAULT_PROBE_WORKSPACE_DIR
            )

            body = _build_conversation_body(
                agent_settings,
                server_id,
                server_dump,
                conversation_id,
                workspace_working_dir,
            )
            headers = {'X-Session-API-Key': session_api_key}
            response = await httpx_client.post(
                f'{agent_server_url}/api/conversations',
                json=body,
                headers=headers,
                timeout=SANDBOX_READY_TIMEOUT_S,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                return McpProbeResult(
                    success=False,
                    sandbox_id=sandbox_id,
                    category=MCPServerFailureCategory.PROTOCOL,
                    message=f'Failed to start test conversation (HTTP {exc.response.status_code})',
                    latency_ms=int((time.monotonic() - started) * 1000),
                )

            try:
                await _initialize_probe_conversation(
                    httpx_client=httpx_client,
                    agent_server_url=agent_server_url,
                    conversation_id=conversation_id,
                    headers=headers,
                )
            except httpx.HTTPStatusError as exc:
                return McpProbeResult(
                    success=False,
                    sandbox_id=sandbox_id,
                    category=MCPServerFailureCategory.PROTOCOL,
                    message=(
                        'Failed to initialize test conversation for MCP discovery '
                        f'(HTTP {exc.response.status_code})'
                    ),
                    latency_ms=int((time.monotonic() - started) * 1000),
                )

            deadline = time.monotonic() + EVENT_POLL_TIMEOUT_S
            tool_names: list[str] = []
            mcp_tool_names: list[str] = []
            while time.monotonic() < deadline:
                events_response = await httpx_client.get(
                    f'{agent_server_url}/api/conversations/{conversation_id.hex}/events/search',
                    params={'limit': 100},
                    headers=headers,
                    timeout=15.0,
                )
                if events_response.status_code == 404:
                    await asyncio.sleep(1)
                    continue
                events_response.raise_for_status()
                payload = events_response.json()
                events = payload.get('items') or payload.get('events') or []
                tool_names = _tool_names_from_events(events)
                mcp_tool_names = _mcp_tool_names_from_events(events)
                if mcp_tool_names:
                    latency_ms = int((time.monotonic() - started) * 1000)
                    return McpProbeResult(
                        success=True,
                        tool_count=len(mcp_tool_names),
                        latency_ms=latency_ms,
                        sandbox_id=sandbox_id,
                    )
                await asyncio.sleep(1)

            events_response = await httpx_client.get(
                f'{agent_server_url}/api/conversations/{conversation_id.hex}/events/search',
                params={'limit': 100},
                headers=headers,
                timeout=15.0,
            )
            if events_response.status_code == 200:
                payload = events_response.json()
                events = payload.get('items') or payload.get('events') or []
                tool_names = _tool_names_from_events(events)
                mcp_tool_names = _mcp_tool_names_from_events(events)

            latency_ms = int((time.monotonic() - started) * 1000)
            if tool_names and not mcp_tool_names:
                return McpProbeResult(
                    success=False,
                    tool_count=len(tool_names),
                    latency_ms=latency_ms,
                    sandbox_id=sandbox_id,
                    category=MCPServerFailureCategory.TOOL_DISCOVERY,
                    message=(
                        'Conversation started but MCP tools were not registered '
                        f'(found built-in tools: {", ".join(tool_names[:5])})'
                    ),
                )
            return McpProbeResult(
                success=False,
                latency_ms=latency_ms,
                sandbox_id=sandbox_id,
                category=MCPServerFailureCategory.TOOL_DISCOVERY,
                message=(
                    'Conversation started but MCP tools were not registered. '
                    'For stdio servers, ensure the command and script path exist '
                    f'inside the sandbox workspace ({workspace_working_dir!r}).'
                ),
            )
    except TimeoutError:
        return McpProbeResult(
            success=False,
            sandbox_id=sandbox.id if sandbox else None,
            category=MCPServerFailureCategory.TIMEOUT,
            message='Stdio MCP test timed out',
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        logger.exception('Stdio MCP probe failed for server %s', server_id)
        return McpProbeResult(
            success=False,
            sandbox_id=sandbox.id if sandbox else None,
            category=MCPServerFailureCategory.INTERNAL,
            message=str(exc)[:500],
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox_service.delete_sandbox(sandbox.id)
            except Exception:
                logger.warning(
                    'Failed to delete MCP test sandbox %s', sandbox.id, exc_info=True
                )
