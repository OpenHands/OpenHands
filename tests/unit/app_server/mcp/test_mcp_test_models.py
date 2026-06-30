"""Unit tests for MCP test run health mapping."""

from __future__ import annotations

from uuid import uuid4

from openhands.app_server.mcp.mcp_test_models import (
    MCPServerFailureCategory,
    McpServerHealthStatus,
    McpServerTestRun,
    McpServerTestRunStatus,
    McpServerTransport,
    health_from_test_run,
)


def test_health_unknown_when_no_run():
    health = health_from_test_run(None, 'demo')
    assert health.status == McpServerHealthStatus.UNKNOWN


def test_health_testing_when_running():
    run = McpServerTestRun(
        id=uuid4(),
        server_id='demo',
        transport=McpServerTransport.STDIO,
        status=McpServerTestRunStatus.RUNNING,
    )
    health = health_from_test_run(run, 'demo')
    assert health.status == McpServerHealthStatus.TESTING
    assert health.test_id == run.id


def test_health_healthy_on_success():
    run = McpServerTestRun(
        id=uuid4(),
        server_id='demo',
        transport=McpServerTransport.SSE,
        status=McpServerTestRunStatus.SUCCEEDED,
        tool_count=3,
        latency_ms=120,
    )
    health = health_from_test_run(run, 'demo')
    assert health.status == McpServerHealthStatus.HEALTHY
    assert health.tool_count == 3


def test_health_unhealthy_on_failure():
    run = McpServerTestRun(
        id=uuid4(),
        server_id='demo',
        transport=McpServerTransport.SHTTP,
        status=McpServerTestRunStatus.FAILED,
        category=MCPServerFailureCategory.AUTHENTICATION,
        message='Authentication failed',
    )
    health = health_from_test_run(run, 'demo')
    assert health.status == McpServerHealthStatus.UNHEALTHY
    assert health.category == MCPServerFailureCategory.AUTHENTICATION
