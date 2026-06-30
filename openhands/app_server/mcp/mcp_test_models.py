"""Models for MCP server connection test runs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import Field

from openhands.agent_server.utils import utc_now
from openhands.sdk.utils.models import OpenHandsModel


class McpServerTestRunStatus(str, Enum):
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class MCPServerFailureCategory(str, Enum):
    CONFIGURATION = 'configuration'
    CONNECTION = 'connection'
    AUTHENTICATION = 'authentication'
    PROTOCOL = 'protocol'
    TOOL_DISCOVERY = 'tool_discovery'
    EXECUTION = 'execution'
    SANDBOX = 'sandbox'
    TIMEOUT = 'timeout'
    INTERNAL = 'internal'


class McpServerTransport(str, Enum):
    STDIO = 'stdio'
    SSE = 'sse'
    SHTTP = 'shttp'


class McpServerHealthStatus(str, Enum):
    UNKNOWN = 'unknown'
    TESTING = 'testing'
    HEALTHY = 'healthy'
    UNHEALTHY = 'unhealthy'


class McpServerTestRun(OpenHandsModel):
    """A single MCP server connection test run."""

    id: UUID = Field(default_factory=uuid4)
    created_by_user_id: str | None = None
    server_id: str
    transport: McpServerTransport
    status: McpServerTestRunStatus = McpServerTestRunStatus.RUNNING
    category: MCPServerFailureCategory | None = None
    message: str | None = None
    tool_count: int | None = None
    latency_ms: int | None = None
    sandbox_id: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class McpServerTestRunPage(OpenHandsModel):
    items: list[McpServerTestRun]
    next_page_id: str | None = None


class McpServerTestRunSortOrder(str, Enum):
    CREATED_AT = 'CREATED_AT'
    CREATED_AT_DESC = 'CREATED_AT_DESC'


class StartMcpServerTestResponse(OpenHandsModel):
    test_id: UUID
    status: McpServerTestRunStatus


class McpServerHealthResponse(OpenHandsModel):
    server_id: str
    status: McpServerHealthStatus
    category: MCPServerFailureCategory | None = None
    message: str | None = None
    tool_count: int | None = None
    latency_ms: int | None = None
    tested_at: datetime | None = None
    test_id: UUID | None = None


class McpProbeResult(OpenHandsModel):
    success: bool
    tool_count: int | None = None
    latency_ms: int | None = None
    category: MCPServerFailureCategory | None = None
    message: str | None = None
    sandbox_id: str | None = None


def health_from_test_run(
    run: McpServerTestRun | None, server_id: str
) -> McpServerHealthResponse:
    """Map the latest test run to a UI-facing health summary."""
    if run is None:
        return McpServerHealthResponse(
            server_id=server_id, status=McpServerHealthStatus.UNKNOWN
        )

    if run.status == McpServerTestRunStatus.RUNNING:
        return McpServerHealthResponse(
            server_id=server_id,
            status=McpServerHealthStatus.TESTING,
            test_id=run.id,
            tested_at=run.started_at,
        )

    if run.status == McpServerTestRunStatus.SUCCEEDED:
        return McpServerHealthResponse(
            server_id=server_id,
            status=McpServerHealthStatus.HEALTHY,
            tool_count=run.tool_count,
            latency_ms=run.latency_ms,
            tested_at=run.finished_at or run.created_at,
            test_id=run.id,
        )

    return McpServerHealthResponse(
        server_id=server_id,
        status=McpServerHealthStatus.UNHEALTHY,
        category=run.category,
        message=run.message,
        tool_count=run.tool_count,
        latency_ms=run.latency_ms,
        tested_at=run.finished_at or run.created_at,
        test_id=run.id,
    )
