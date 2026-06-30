"""Tests for SQLMcpServerTestRunService."""

from __future__ import annotations

from typing import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.mcp.mcp_test_models import (
    McpServerTestRun,
    McpServerTestRunStatus,
    McpServerTransport,
)
from openhands.app_server.mcp.sql_mcp_test_run_service import SQLMcpServerTestRunService
from openhands.app_server.utils.sql_utils import Base


@pytest.fixture
async def async_engine():
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def service(async_session: AsyncSession) -> SQLMcpServerTestRunService:
    return SQLMcpServerTestRunService(session=async_session, user_id='user-1')


@pytest.mark.asyncio
async def test_create_and_get_test_run(service: SQLMcpServerTestRunService):
    run = McpServerTestRun(
        id=uuid4(),
        created_by_user_id='user-1',
        server_id='demo',
        transport=McpServerTransport.STDIO,
        status=McpServerTestRunStatus.RUNNING,
    )
    saved = await service.create_test_run(run)
    fetched = await service.get_test_run(saved.id)
    assert fetched is not None
    assert fetched.server_id == 'demo'
    assert fetched.status == McpServerTestRunStatus.RUNNING


@pytest.mark.asyncio
async def test_get_running_and_latest(service: SQLMcpServerTestRunService):
    run1 = McpServerTestRun(
        id=uuid4(),
        created_by_user_id='user-1',
        server_id='demo',
        transport=McpServerTransport.SSE,
        status=McpServerTestRunStatus.SUCCEEDED,
    )
    run2 = McpServerTestRun(
        id=uuid4(),
        created_by_user_id='user-1',
        server_id='demo',
        transport=McpServerTransport.SSE,
        status=McpServerTestRunStatus.RUNNING,
    )
    await service.create_test_run(run1)
    await service.create_test_run(run2)

    running = await service.get_running_test_run('demo')
    assert running is not None
    assert running.status == McpServerTestRunStatus.RUNNING

    latest = await service.get_latest_test_run('demo')
    assert latest is not None
    assert latest.id == run2.id


@pytest.mark.asyncio
async def test_search_test_runs_pagination(service: SQLMcpServerTestRunService):
    for idx in range(3):
        await service.create_test_run(
            McpServerTestRun(
                id=uuid4(),
                created_by_user_id='user-1',
                server_id='demo',
                transport=McpServerTransport.SHTTP,
                status=McpServerTestRunStatus.FAILED,
            )
        )

    page = await service.search_test_runs(server_id='demo', limit=2)
    assert len(page.items) == 2
    assert page.next_page_id == '2'

    page2 = await service.search_test_runs(
        server_id='demo', limit=2, page_id=page.next_page_id
    )
    assert len(page2.items) == 1
    assert page2.next_page_id is None
