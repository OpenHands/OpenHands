# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""SQL implementation of AppConversationStartTaskService.

This implementation provides CRUD operations for conversation start tasks focused purely
on SQL operations:
- Direct database access without permission checks
- Batch operations for efficient data retrieval
- Full async/await support using SQL async sessions

Security and permission checks are handled by wrapper services.

Key components:
- SQLAppConversationStartTaskService: Main service class implementing all operations
- SQLAppConversationStartTaskServiceInjector: Dependency injection resolver for FastAPI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from sqlalchemy import UUID as SQLUUID
from sqlalchemy import Column, Enum, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.agent_server.models import utc_now
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskPage,
    AppConversationStartTaskSortOrder,
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.app_conversation_start_task_service import (
    AppConversationStartTaskService,
    AppConversationStartTaskServiceInjector,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.sql_utils import (
    Base,
    UtcDateTime,
    create_json_type_decorator,
    row2dict,
)

logger = logging.getLogger(__name__)


class StoredAppConversationStartTask(Base):  # type: ignore
    __tablename__ = 'app_conversation_start_task'
    id = Column(SQLUUID, primary_key=True)
    created_by_user_id = Column(String, index=True)
    status = Column(Enum(AppConversationStartTaskStatus), nullable=True)
    detail = Column(String, nullable=True)
    app_conversation_id = Column(SQLUUID, nullable=True)
    sandbox_id = Column(String, nullable=True)
    agent_server_url = Column(String, nullable=True)
    request = Column(create_json_type_decorator(AppConversationStartRequest))
    created_at = Column(UtcDateTime, server_default=func.now(), index=True)
    updated_at = Column(UtcDateTime, onupdate=func.now(), index=True)


@dataclass
class SQLAppConversationStartTaskService(AppConversationStartTaskService):
    """SQL implementation of AppConversationStartTaskService focused on db operations.

    This allows storing and retrieving conversation start tasks from the database.
    """

    session: AsyncSession
    user_id: str | None = None

    async def search_app_conversation_start_tasks(
        self,
        conversation_id__eq: UUID | None = None,
        created_at__gte: datetime | None = None,
        sort_order: AppConversationStartTaskSortOrder = AppConversationStartTaskSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 100,
    ) -> AppConversationStartTaskPage:
        """Search for conversation start tasks."""
        query = select(StoredAppConversationStartTask)

        # Apply user filter if user_id is set
        if self.user_id:
            query = query.where(
                StoredAppConversationStartTask.created_by_user_id == self.user_id
            )

        # Apply conversation_id filter
        if conversation_id__eq is not None:
            query = query.where(
                StoredAppConversationStartTask.app_conversation_id
                == conversation_id__eq
            )

        # Apply created_at__gte filter
        if created_at__gte is not None:
            query = query.where(
                StoredAppConversationStartTask.created_at >= created_at__gte
            )

        # Add sort order
        if sort_order == AppConversationStartTaskSortOrder.CREATED_AT:
            query = query.order_by(StoredAppConversationStartTask.created_at)
        elif sort_order == AppConversationStartTaskSortOrder.CREATED_AT_DESC:
            query = query.order_by(StoredAppConversationStartTask.created_at.desc())
        elif sort_order == AppConversationStartTaskSortOrder.UPDATED_AT:
            query = query.order_by(StoredAppConversationStartTask.updated_at)
        elif sort_order == AppConversationStartTaskSortOrder.UPDATED_AT_DESC:
            query = query.order_by(StoredAppConversationStartTask.updated_at.desc())

        # Apply pagination
        if page_id is not None:
            try:
                offset = int(page_id)
                query = query.offset(offset)
            except ValueError:
                # If page_id is not a valid integer, start from beginning
                offset = 0
        else:
            offset = 0

        # Apply limit and get one extra to check if there are more results
        query = query.limit(limit + 1)

        result = await self.session.execute(query)
        rows = result.scalars().all()

        # Check if there are more results
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [AppConversationStartTask.model_validate(row2dict(row)) for row in rows]

        # Calculate next page ID
        next_page_id = None
        if has_more:
            next_page_id = str(offset + limit)

        return AppConversationStartTaskPage(items=items, next_page_id=next_page_id)

    async def count_app_conversation_start_tasks(
        self,
        conversation_id__eq: UUID | None = None,
        created_at__gte: datetime | None = None,
    ) -> int:
        """Count conversation start tasks."""
        query = select(func.count(StoredAppConversationStartTask.id))

        # Apply user filter if user_id is set
        if self.user_id:
            query = query.where(
                StoredAppConversationStartTask.created_by_user_id == self.user_id
            )

        # Apply conversation_id filter
        if conversation_id__eq is not None:
            query = query.where(
                StoredAppConversationStartTask.app_conversation_id
                == conversation_id__eq
            )

        # Apply created_at__gte filter
        if created_at__gte is not None:
            query = query.where(
                StoredAppConversationStartTask.created_at >= created_at__gte
            )

        result = await self.session.execute(query)
        count = result.scalar()
        return count or 0

    async def batch_get_app_conversation_start_tasks(
        self, task_ids: list[UUID]
    ) -> list[AppConversationStartTask | None]:
        """Get a batch of start tasks, return None for any missing."""
        if not task_ids:
            return []

        query = select(StoredAppConversationStartTask).where(
            StoredAppConversationStartTask.id.in_(task_ids)
        )
        if self.user_id:
            query = query.where(
                StoredAppConversationStartTask.created_by_user_id == self.user_id
            )

        result = await self.session.execute(query)
        tasks_by_id = {task.id: task for task in result.scalars().all()}

        # Return tasks in the same order as requested, with None for missing ones
        return [
            (
                AppConversationStartTask.model_validate(row2dict(tasks_by_id[task_id]))
                if task_id in tasks_by_id
                else None
            )
            for task_id in task_ids
        ]

    async def get_app_conversation_start_task(
        self, task_id: UUID
    ) -> AppConversationStartTask | None:
        """Get a single start task, returning None if missing."""
        query = select(StoredAppConversationStartTask).where(
            StoredAppConversationStartTask.id == task_id
        )
        if self.user_id:
            query = query.where(
                StoredAppConversationStartTask.created_by_user_id == self.user_id
            )

        result = await self.session.execute(query)
        stored_task = result.scalar_one_or_none()
        if stored_task:
            return AppConversationStartTask.model_validate(row2dict(stored_task))
        return None

    async def save_app_conversation_start_task(
        self, task: AppConversationStartTask
    ) -> AppConversationStartTask:
        if self.user_id:
            query = select(StoredAppConversationStartTask).where(
                StoredAppConversationStartTask.id == task.id
            )
            result = await self.session.execute(query)
            existing = result.scalar_one_or_none()
            assert existing is None or existing.created_by_user_id == self.user_id
        task.updated_at = utc_now()
        await self.session.merge(StoredAppConversationStartTask(**task.model_dump()))
        await self.session.commit()
        return task

    async def delete_app_conversation_start_tasks(self, conversation_id: UUID) -> bool:
        """Delete all start tasks associated with a conversation.

        Args:
            conversation_id: The ID of the conversation to delete tasks for.
        """
        from sqlalchemy import delete

        # Build secure delete query with user filter if user_id is set
        delete_query = delete(StoredAppConversationStartTask).where(
            StoredAppConversationStartTask.app_conversation_id == conversation_id
        )

        if self.user_id:
            delete_query = delete_query.where(
                StoredAppConversationStartTask.created_by_user_id == self.user_id
            )

        result = await self.session.execute(delete_query)

        # Return True if any rows were affected
        return result.rowcount > 0


class SQLAppConversationStartTaskServiceInjector(
    AppConversationStartTaskServiceInjector
):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[AppConversationStartTaskService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import (
            get_db_session,
            get_user_context,
        )

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            user_id = await user_context.get_user_id()
            service = SQLAppConversationStartTaskService(
                session=db_session, user_id=user_id
            )
            yield service
