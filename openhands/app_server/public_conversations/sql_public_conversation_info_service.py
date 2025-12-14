"""SQL implementation of PublicConversationInfoService.

This implementation provides read-only operations for public conversations:
- Direct database access filtering only public conversations
- Batch operations for efficient data retrieval
- Full async/await support using SQL async db_sessions

Key components:
- SQLPublicConversationInfoService: Main service class implementing all operations
- SQLPublicConversationInfoServiceInjector: Dependency injection resolver for FastAPI
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    Select,
    String,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.agent_server.utils import utc_now
from openhands.app_server.public_conversations.public_conversation_info_service import (
    PublicConversationInfoService,
)
from openhands.app_server.public_conversations.public_conversation_models import (
    PublicConversationInfo,
    PublicConversationInfoPage,
    PublicConversationSortOrder,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    StoredConversationMetadata,
)
from openhands.app_server.services.db_session_injector import (
    DBSessionInjector,
    depends_db_session,
)
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.user.user_context import UserContext

_logger = logging.getLogger(__name__)


@dataclass
class SQLPublicConversationInfoService(PublicConversationInfoService):
    """SQL implementation of PublicConversationInfoService."""

    db_session: AsyncSession

    async def search_public_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sort_order: PublicConversationSortOrder = PublicConversationSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 100,
        include_sub_conversations: bool = False,
    ) -> PublicConversationInfoPage:
        """Search for public conversations."""
        query = select(StoredConversationMetadata).where(
            StoredConversationMetadata.is_public == True  # noqa: E712
        )

        # Apply filters
        if title__contains:
            query = query.where(StoredConversationMetadata.title.contains(title__contains))
        if created_at__gte:
            query = query.where(StoredConversationMetadata.created_at >= created_at__gte)
        if created_at__lt:
            query = query.where(StoredConversationMetadata.created_at < created_at__lt)
        if updated_at__gte:
            query = query.where(StoredConversationMetadata.last_updated_at >= updated_at__gte)
        if updated_at__lt:
            query = query.where(StoredConversationMetadata.last_updated_at < updated_at__lt)

        if not include_sub_conversations:
            query = query.where(StoredConversationMetadata.parent_conversation_id.is_(None))

        # Apply sorting
        if sort_order == PublicConversationSortOrder.CREATED_AT:
            query = query.order_by(StoredConversationMetadata.created_at.asc())
        elif sort_order == PublicConversationSortOrder.CREATED_AT_DESC:
            query = query.order_by(StoredConversationMetadata.created_at.desc())
        elif sort_order == PublicConversationSortOrder.UPDATED_AT:
            query = query.order_by(StoredConversationMetadata.last_updated_at.asc())
        elif sort_order == PublicConversationSortOrder.UPDATED_AT_DESC:
            query = query.order_by(StoredConversationMetadata.last_updated_at.desc())
        elif sort_order == PublicConversationSortOrder.TITLE:
            query = query.order_by(StoredConversationMetadata.title.asc())
        elif sort_order == PublicConversationSortOrder.TITLE_DESC:
            query = query.order_by(StoredConversationMetadata.title.desc())

        # Apply pagination
        if page_id:
            # Decode page_id to get the last seen value for cursor-based pagination
            try:
                last_value = datetime.fromisoformat(page_id)
                if sort_order in [
                    PublicConversationSortOrder.CREATED_AT,
                    PublicConversationSortOrder.CREATED_AT_DESC,
                ]:
                    if sort_order == PublicConversationSortOrder.CREATED_AT:
                        query = query.where(StoredConversationMetadata.created_at > last_value)
                    else:
                        query = query.where(StoredConversationMetadata.created_at < last_value)
                elif sort_order in [
                    PublicConversationSortOrder.UPDATED_AT,
                    PublicConversationSortOrder.UPDATED_AT_DESC,
                ]:
                    if sort_order == PublicConversationSortOrder.UPDATED_AT:
                        query = query.where(StoredConversationMetadata.last_updated_at > last_value)
                    else:
                        query = query.where(StoredConversationMetadata.last_updated_at < last_value)
            except ValueError:
                # Invalid page_id, ignore pagination
                pass

        query = query.limit(limit + 1)  # Get one extra to check if there's a next page

        result = await self.db_session.execute(query)
        conversations = result.scalars().all()

        # Check if there's a next page
        next_page_id = None
        if len(conversations) > limit:
            conversations = conversations[:limit]
            last_conversation = conversations[-1]
            if sort_order in [
                PublicConversationSortOrder.CREATED_AT,
                PublicConversationSortOrder.CREATED_AT_DESC,
            ]:
                next_page_id = last_conversation.created_at.isoformat()
            elif sort_order in [
                PublicConversationSortOrder.UPDATED_AT,
                PublicConversationSortOrder.UPDATED_AT_DESC,
            ]:
                next_page_id = last_conversation.last_updated_at.isoformat()

        # Convert to PublicConversationInfo
        items = [self._convert_to_public_conversation_info(conv) for conv in conversations]

        return PublicConversationInfoPage(items=items, next_page_id=next_page_id)

    async def count_public_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
    ) -> int:
        """Count public conversations."""
        query = select(func.count(StoredConversationMetadata.conversation_id)).where(
            StoredConversationMetadata.is_public == True  # noqa: E712
        )

        # Apply filters
        if title__contains:
            query = query.where(StoredConversationMetadata.title.contains(title__contains))
        if created_at__gte:
            query = query.where(StoredConversationMetadata.created_at >= created_at__gte)
        if created_at__lt:
            query = query.where(StoredConversationMetadata.created_at < created_at__lt)
        if updated_at__gte:
            query = query.where(StoredConversationMetadata.last_updated_at >= updated_at__gte)
        if updated_at__lt:
            query = query.where(StoredConversationMetadata.last_updated_at < updated_at__lt)

        result = await self.db_session.execute(query)
        return result.scalar() or 0

    async def get_public_conversation_info(
        self, conversation_id: UUID
    ) -> PublicConversationInfo | None:
        """Get a single public conversation info, returning None if missing or not public."""
        query = select(StoredConversationMetadata).where(
            StoredConversationMetadata.conversation_id == str(conversation_id),
            StoredConversationMetadata.is_public == True,  # noqa: E712
        )

        result = await self.db_session.execute(query)
        conversation = result.scalar_one_or_none()

        if conversation is None:
            return None

        return self._convert_to_public_conversation_info(conversation)

    async def get_public_conversation_info_by_token(
        self, token: str
    ) -> PublicConversationInfo | None:
        """Get a single public conversation info by share token, returning None if missing or not public."""
        query = select(StoredConversationMetadata).where(
            StoredConversationMetadata.public_share_token == token,
            StoredConversationMetadata.is_public == True,  # noqa: E712
        )

        result = await self.db_session.execute(query)
        conversation = result.scalar_one_or_none()

        if conversation is None:
            return None

        return self._convert_to_public_conversation_info(conversation)

    def _convert_to_public_conversation_info(
        self, conversation: StoredConversationMetadata
    ) -> PublicConversationInfo:
        """Convert StoredConversationMetadata to PublicConversationInfo."""
        from uuid import UUID
        from openhands.storage.data_models.conversation_metadata import ConversationTrigger
        
        # Convert string IDs to UUIDs
        conversation_id = UUID(conversation.conversation_id)
        parent_id = UUID(conversation.parent_conversation_id) if conversation.parent_conversation_id else None
        
        # Convert trigger string to enum
        trigger = None
        if conversation.trigger:
            try:
                trigger = ConversationTrigger(conversation.trigger)
            except ValueError:
                trigger = None
        
        return PublicConversationInfo(
            id=conversation_id,
            sandbox_id=conversation.sandbox_id,
            selected_repository=conversation.selected_repository,
            selected_branch=conversation.selected_branch,
            git_provider=conversation.git_provider,
            title=conversation.title,
            trigger=trigger,
            pr_number=conversation.pr_number or [],
            llm_model=conversation.llm_model,
            metrics=None,  # TODO: Convert metrics if needed
            parent_conversation_id=parent_id,
            sub_conversation_ids=[],  # TODO: Query sub-conversations if needed
            created_at=conversation.created_at,
            updated_at=conversation.last_updated_at,
        )


class SQLPublicConversationInfoServiceInjector:
    """Dependency injection for SQLPublicConversationInfoService."""

    async def __call__(self, request: Request) -> AsyncGenerator[SQLPublicConversationInfoService, None]:
        """Create and yield a SQLPublicConversationInfoService instance."""
        db_session_injector = DBSessionInjector()
        async with db_session_injector(request) as db_session:
            yield SQLPublicConversationInfoService(db_session=db_session)


def depends_sql_public_conversation_info_service() -> SQLPublicConversationInfoService:
    """Dependency injection for SQLPublicConversationInfoService."""
    return Injector.get(PublicConversationInfoService)


def register_sql_public_conversation_info_service():
    """Register the SQL implementation of PublicConversationInfoService."""
    Injector.register(
        PublicConversationInfoService,
        SQLPublicConversationInfoServiceInjector(),
        InjectorState.ASYNC_GENERATOR,
    )