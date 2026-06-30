"""Orchestrates MCP server connection tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from fastmcp.mcp_config import MCPConfig

from openhands.agent_server.models import utc_now
from openhands.app_server.mcp.mcp_probe import (
    REMOTE_PROBE_TIMEOUT_S,
    headers_from_server_config,
    probe_remote_mcp_server,
)
from openhands.app_server.mcp.mcp_stdio_probe import (
    STDIO_PROBE_TIMEOUT_S,
    probe_stdio_mcp_server,
)
from openhands.app_server.mcp.mcp_test_models import (
    McpProbeResult,
    MCPServerFailureCategory,
    McpServerTestRun,
    McpServerTestRunStatus,
    McpServerTransport,
)
from openhands.app_server.mcp.mcp_test_run_service import McpServerTestRunService
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.settings import ACPAgentSettings, OpenHandsAgentSettings

_background_tasks: set[asyncio.Task] = set()


@dataclass
class ResolvedMcpServer:
    server_id: str
    transport: McpServerTransport
    server_dump: dict[str, Any]
    url: str | None = None
    headers: dict[str, str] | None = None


class McpServerNotFoundError(LookupError):
    pass


class McpServerConfigurationError(ValueError):
    pass


def resolve_mcp_server(settings: Settings, server_id: str) -> ResolvedMcpServer:
    agent_settings = settings.agent_settings
    if isinstance(agent_settings, ACPAgentSettings):
        raise McpServerConfigurationError(
            'Custom MCP servers are not supported for ACP agent settings'
        )

    mcp_config: MCPConfig | None = agent_settings.mcp_config
    if not mcp_config or not mcp_config.mcpServers:
        raise McpServerNotFoundError(f'MCP server {server_id!r} not found')

    server = mcp_config.mcpServers.get(server_id)
    if server is None:
        raise McpServerNotFoundError(f'MCP server {server_id!r} not found')

    server_dump = server.model_dump(exclude_none=True)
    if server_dump.get('command'):
        return ResolvedMcpServer(
            server_id=server_id,
            transport=McpServerTransport.STDIO,
            server_dump=server_dump,
        )

    url = server_dump.get('url')
    if not url:
        raise McpServerConfigurationError(
            f'MCP server {server_id!r} is missing both command and url'
        )

    transport_value = server_dump.get('transport')
    if transport_value == 'sse':
        transport = McpServerTransport.SSE
    else:
        transport = McpServerTransport.SHTTP

    return ResolvedMcpServer(
        server_id=server_id,
        transport=transport,
        server_dump=server_dump,
        url=url,
        headers=headers_from_server_config(server_dump) or None,
    )


@dataclass
class McpTestService:
    test_run_service: McpServerTestRunService
    sandbox_service: SandboxService
    sandbox_spec_service: SandboxSpecService
    httpx_client: httpx.AsyncClient
    user_context: UserContext

    async def start_test(self, server_id: str, settings: Settings) -> McpServerTestRun:
        resolved = resolve_mcp_server(settings, server_id)
        existing = await self.test_run_service.get_running_test_run(server_id)
        if existing is not None:
            return existing

        run = McpServerTestRun(
            server_id=server_id,
            transport=resolved.transport,
            status=McpServerTestRunStatus.RUNNING,
            created_by_user_id=await self.user_context.get_user_id(),
        )
        run = await self.test_run_service.create_test_run(run)
        user_id = run.created_by_user_id
        task = asyncio.create_task(
            _run_test_in_background(
                test_id=run.id,
                server_id=server_id,
                user_id=user_id,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return run

    async def run_test(
        self,
        *,
        settings: Settings,
        resolved: ResolvedMcpServer,
    ) -> McpProbeResult:
        if resolved.transport == McpServerTransport.STDIO:
            agent_settings = settings.agent_settings
            if not isinstance(agent_settings, OpenHandsAgentSettings):
                return McpProbeResult(
                    success=False,
                    category=MCPServerFailureCategory.CONFIGURATION,
                    message='Invalid agent settings for stdio MCP test',
                )
            return await probe_stdio_mcp_server(
                sandbox_service=self.sandbox_service,
                sandbox_spec_service=self.sandbox_spec_service,
                httpx_client=self.httpx_client,
                agent_settings=agent_settings,
                server_id=resolved.server_id,
                server_dump=resolved.server_dump,
            )

        assert resolved.url is not None
        return await probe_remote_mcp_server(
            url=resolved.url,
            headers=resolved.headers,
            transport=resolved.transport,
        )

    async def finish_test_run(
        self, test_id: UUID, result: McpProbeResult
    ) -> McpServerTestRun | None:
        run = await self.test_run_service.get_test_run(test_id)
        if run is None:
            return None
        if run.status != McpServerTestRunStatus.RUNNING:
            return run

        run.finished_at = utc_now()
        run.tool_count = result.tool_count
        run.latency_ms = result.latency_ms
        run.sandbox_id = result.sandbox_id
        if result.success:
            run.status = McpServerTestRunStatus.SUCCEEDED
            run.category = None
            run.message = None
        else:
            run.status = McpServerTestRunStatus.FAILED
            run.category = result.category or MCPServerFailureCategory.INTERNAL
            run.message = result.message
        return await self.test_run_service.save_test_run(run)


async def _run_test_in_background(
    *,
    test_id: UUID,
    server_id: str,
    user_id: str | None,
) -> None:
    from openhands.app_server.config import (
        get_httpx_client,
        get_sandbox_service,
        get_sandbox_spec_service,
    )
    from openhands.app_server.mcp.sql_mcp_test_run_service import (
        SQLMcpServerTestRunService,
    )
    from openhands.app_server.services.db_session import get_db_session
    from openhands.app_server.services.injector import InjectorState
    from openhands.app_server.user.auth_user_context import AuthUserContext
    from openhands.app_server.user_auth.default_user_auth import DefaultUserAuth
    from openhands.app_server.user_auth.user_auth import get_for_user

    state = InjectorState()
    timeout = STDIO_PROBE_TIMEOUT_S
    try:
        user_auth = await get_for_user(user_id) if user_id else DefaultUserAuth()
        user_context = AuthUserContext(user_auth=user_auth)

        async with (
            get_db_session(state) as db_session,
            get_sandbox_service(state) as sandbox_service,
            get_sandbox_spec_service(state) as sandbox_spec_service,
            get_httpx_client(state) as httpx_client,
        ):
            test_run_service = SQLMcpServerTestRunService(
                session=db_session, user_id=user_id
            )
            service = McpTestService(
                test_run_service=test_run_service,
                sandbox_service=sandbox_service,
                sandbox_spec_service=sandbox_spec_service,
                httpx_client=httpx_client,
                user_context=user_context,
            )
            settings = await user_auth.get_user_settings()
            if settings is None:
                await service.finish_test_run(
                    test_id,
                    McpProbeResult(
                        success=False,
                        category=MCPServerFailureCategory.CONFIGURATION,
                        message='User settings not found',
                    ),
                )
                return

            resolved = resolve_mcp_server(settings, server_id)
            if resolved.transport != McpServerTransport.STDIO:
                timeout = REMOTE_PROBE_TIMEOUT_S

            try:
                async with asyncio.timeout(timeout):
                    result = await service.run_test(
                        settings=settings, resolved=resolved
                    )
            except TimeoutError:
                result = McpProbeResult(
                    success=False,
                    category=MCPServerFailureCategory.TIMEOUT,
                    message='MCP test timed out',
                )
            except Exception as exc:
                logger.exception('Unexpected MCP test failure for server %s', server_id)
                result = McpProbeResult(
                    success=False,
                    category=MCPServerFailureCategory.INTERNAL,
                    message=str(exc)[:500],
                )

            await service.finish_test_run(test_id, result)
    except Exception:
        logger.exception(
            'Background MCP test task failed for test_id=%s server_id=%s',
            test_id,
            server_id,
        )
        try:
            async with get_db_session(state) as db_session:
                test_run_service = SQLMcpServerTestRunService(
                    session=db_session, user_id=user_id
                )
                run = await test_run_service.get_test_run(test_id)
                if run and run.status == McpServerTestRunStatus.RUNNING:
                    run.status = McpServerTestRunStatus.FAILED
                    run.category = MCPServerFailureCategory.INTERNAL
                    run.message = 'MCP test failed unexpectedly'
                    run.finished_at = utc_now()
                    await test_run_service.save_test_run(run)
        except Exception:
            logger.exception('Failed to mark MCP test run %s as failed', test_id)
