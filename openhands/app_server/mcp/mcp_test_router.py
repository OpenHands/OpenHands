"""API routes for MCP server connection tests."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from openhands.app_server.config import (
    depends_httpx_client,
    depends_mcp_test_run_service,
    depends_sandbox_service,
    depends_sandbox_spec_service,
    depends_user_context,
)
from openhands.app_server.mcp.mcp_test_models import (
    McpServerHealthResponse,
    McpServerTestRun,
    McpServerTestRunPage,
    McpServerTestRunSortOrder,
    StartMcpServerTestResponse,
    health_from_test_run,
)
from openhands.app_server.mcp.mcp_test_run_service import McpServerTestRunService
from openhands.app_server.mcp.mcp_test_service import (
    McpServerConfigurationError,
    McpServerNotFoundError,
    McpTestService,
    resolve_mcp_server,
)
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user_auth import get_user_settings
from openhands.app_server.utils.dependencies import get_dependencies

router = APIRouter(
    prefix='/settings/mcp',
    tags=['MCP Settings'],
    dependencies=get_dependencies(),
)

mcp_test_run_service_dependency = depends_mcp_test_run_service()
sandbox_service_dependency = depends_sandbox_service()
sandbox_spec_service_dependency = depends_sandbox_spec_service()
httpx_client_dependency = depends_httpx_client()
user_context_dependency = depends_user_context()


def _mcp_test_service(
    test_run_service: McpServerTestRunService,
    sandbox_service: SandboxService,
    sandbox_spec_service: SandboxSpecService,
    httpx_client,
    user_context: UserContext,
) -> McpTestService:
    return McpTestService(
        test_run_service=test_run_service,
        sandbox_service=sandbox_service,
        sandbox_spec_service=sandbox_spec_service,
        httpx_client=httpx_client,
        user_context=user_context,
    )


@router.post(
    '/servers/{server_id}/test',
    response_model=StartMcpServerTestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_mcp_server_test(
    server_id: str,
    test_run_service: McpServerTestRunService = mcp_test_run_service_dependency,
    sandbox_service: SandboxService = sandbox_service_dependency,
    sandbox_spec_service: SandboxSpecService = sandbox_spec_service_dependency,
    httpx_client=httpx_client_dependency,
    user_context: UserContext = user_context_dependency,
    settings: Settings | None = Depends(get_user_settings),
) -> StartMcpServerTestResponse:
    if settings is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Settings not found')
    try:
        resolve_mcp_server(settings, server_id)
    except McpServerNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f'MCP server {server_id!r} not found'
        )
    except McpServerConfigurationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    service = _mcp_test_service(
        test_run_service,
        sandbox_service,
        sandbox_spec_service,
        httpx_client,
        user_context,
    )
    run = await service.start_test(server_id, settings)
    return StartMcpServerTestResponse(test_id=run.id, status=run.status)


@router.get('/test-runs/search', response_model=McpServerTestRunPage)
async def search_mcp_test_runs(
    page_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(gt=0, le=100)] = 20,
    test_run_service: McpServerTestRunService = mcp_test_run_service_dependency,
) -> McpServerTestRunPage:
    return await test_run_service.search_test_runs(
        server_id=None,
        sort_order=McpServerTestRunSortOrder.CREATED_AT_DESC,
        page_id=page_id,
        limit=limit,
    )


@router.get('/test-runs/{test_id}', response_model=McpServerTestRun)
async def get_mcp_test_run(
    test_id: UUID,
    test_run_service: McpServerTestRunService = mcp_test_run_service_dependency,
) -> McpServerTestRun:
    run = await test_run_service.get_test_run(test_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Test run not found')
    return run


@router.get('/servers/{server_id}/test-runs', response_model=McpServerTestRunPage)
async def list_mcp_server_test_runs(
    server_id: str,
    page_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(gt=0, le=100)] = 20,
    test_run_service: McpServerTestRunService = mcp_test_run_service_dependency,
    settings: Settings | None = Depends(get_user_settings),
) -> McpServerTestRunPage:
    if settings is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Settings not found')
    try:
        resolve_mcp_server(settings, server_id)
    except McpServerNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f'MCP server {server_id!r} not found'
        )
    return await test_run_service.search_test_runs(
        server_id=server_id,
        sort_order=McpServerTestRunSortOrder.CREATED_AT_DESC,
        page_id=page_id,
        limit=limit,
    )


@router.get('/servers/{server_id}/health', response_model=McpServerHealthResponse)
async def get_mcp_server_health(
    server_id: str,
    test_run_service: McpServerTestRunService = mcp_test_run_service_dependency,
    settings: Settings | None = Depends(get_user_settings),
) -> McpServerHealthResponse:
    if settings is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Settings not found')
    try:
        resolve_mcp_server(settings, server_id)
    except McpServerNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f'MCP server {server_id!r} not found'
        )
    latest = await test_run_service.get_latest_test_run(server_id)
    return health_from_test_run(latest, server_id)
