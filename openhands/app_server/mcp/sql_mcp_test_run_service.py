# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""SQL implementation of McpServerTestRunService."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from sqlalchemy import Enum, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from openhands.agent_server.models import utc_now
from openhands.app_server.mcp.mcp_test_models import (
    MCPServerFailureCategory,
    McpServerTestRun,
    McpServerTestRunPage,
    McpServerTestRunSortOrder,
    McpServerTestRunStatus,
    McpServerTransport,
)
from openhands.app_server.mcp.mcp_test_run_service import (
    McpServerTestRunService,
    McpServerTestRunServiceInjector,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.sql_utils import Base, UtcDateTime, row2dict

logger = logging.getLogger(__name__)


class StoredMcpServerTestRun(Base):
    __tablename__ = 'mcp_server_test_run'

    id: Mapped[UUID] = mapped_column(primary_key=True)
    created_by_user_id: Mapped[str | None] = mapped_column(String, index=True)
    server_id: Mapped[str] = mapped_column(String, index=True)
    transport: Mapped[McpServerTransport] = mapped_column(Enum(McpServerTransport))
    status: Mapped[McpServerTestRunStatus] = mapped_column(
        Enum(McpServerTestRunStatus), index=True
    )
    category: Mapped[MCPServerFailureCategory | None] = mapped_column(
        Enum(MCPServerFailureCategory), nullable=True
    )
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sandbox_id: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), index=True
    )


@dataclass
class SQLMcpServerTestRunService(McpServerTestRunService):
    session: AsyncSession
    user_id: str | None = None

    def _base_query(self):
        query = select(StoredMcpServerTestRun)
        if self.user_id:
            query = query.where(
                StoredMcpServerTestRun.created_by_user_id == self.user_id
            )
        return query

    async def create_test_run(self, run: McpServerTestRun) -> McpServerTestRun:
        if self.user_id:
            run.created_by_user_id = self.user_id
        self.session.add(StoredMcpServerTestRun(**run.model_dump()))
        await self.session.commit()
        return run

    async def save_test_run(self, run: McpServerTestRun) -> McpServerTestRun:
        if self.user_id:
            assert run.created_by_user_id == self.user_id
        await self.session.merge(StoredMcpServerTestRun(**run.model_dump()))
        await self.session.commit()
        return run

    async def get_test_run(self, test_id: UUID) -> McpServerTestRun | None:
        query = self._base_query().where(StoredMcpServerTestRun.id == test_id)
        result = await self.session.execute(query)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return McpServerTestRun.model_validate(row2dict(row))

    async def get_running_test_run(self, server_id: str) -> McpServerTestRun | None:
        query = (
            self._base_query()
            .where(StoredMcpServerTestRun.server_id == server_id)
            .where(StoredMcpServerTestRun.status == McpServerTestRunStatus.RUNNING)
            .order_by(StoredMcpServerTestRun.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return McpServerTestRun.model_validate(row2dict(row))

    async def get_latest_test_run(self, server_id: str) -> McpServerTestRun | None:
        query = (
            self._base_query()
            .where(StoredMcpServerTestRun.server_id == server_id)
            .order_by(StoredMcpServerTestRun.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return McpServerTestRun.model_validate(row2dict(row))

    async def search_test_runs(
        self,
        server_id: str | None = None,
        sort_order: McpServerTestRunSortOrder = McpServerTestRunSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 20,
    ) -> McpServerTestRunPage:
        query = self._base_query()
        if server_id is not None:
            query = query.where(StoredMcpServerTestRun.server_id == server_id)

        if sort_order == McpServerTestRunSortOrder.CREATED_AT:
            query = query.order_by(StoredMcpServerTestRun.created_at)
        else:
            query = query.order_by(StoredMcpServerTestRun.created_at.desc())

        offset = 0
        if page_id is not None:
            try:
                offset = int(page_id)
            except ValueError:
                offset = 0
        query = query.offset(offset).limit(limit + 1)

        result = await self.session.execute(query)
        rows = result.scalars().all()
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [McpServerTestRun.model_validate(row2dict(row)) for row in rows]
        next_page_id = str(offset + limit) if has_more else None
        return McpServerTestRunPage(items=items, next_page_id=next_page_id)


class SQLMcpServerTestRunServiceInjector(McpServerTestRunServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[McpServerTestRunService, None]:
        from openhands.app_server.config import get_db_session, get_user_context

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            user_id = await user_context.get_user_id()
            yield SQLMcpServerTestRunService(session=db_session, user_id=user_id)


def mark_stale_running_tests_failed(
    runs: list[McpServerTestRun], stale_after_seconds: float
) -> list[McpServerTestRun]:
    """Mark RUNNING tests older than stale_after_seconds as FAILED/TIMEOUT."""
    now = utc_now()
    updated: list[McpServerTestRun] = []
    for run in runs:
        if run.status != McpServerTestRunStatus.RUNNING:
            continue
        age = (now - run.started_at).total_seconds()
        if age <= stale_after_seconds:
            continue
        run.status = McpServerTestRunStatus.FAILED
        run.category = MCPServerFailureCategory.TIMEOUT
        run.message = 'Test run timed out'
        run.finished_at = now
        updated.append(run)
    return updated
