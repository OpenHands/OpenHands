"""SQL implementation of AppConversationService.

This implementation provides CRUD operations for sandboxed conversations focused purely
on SQL operations:
- Direct database access without permission checks
- Batch operations for efficient data retrieval
- Integration with SandboxService for sandbox information
- HTTP client integration for agent status retrieval
- Full async/await support using SQL async db_sessions

Security and permission checks are handled by wrapper services.

Key components:
- SQLAppConversationService: Main service class implementing all operations
- SQLAppConversationInfoServiceInjector: Dependency injection resolver for FastAPI
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
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
    AppConversationInfoServiceInjector,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationInfoPage,
    AppConversationSortOrder,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.sql_utils import (
    Base,
    create_json_type_decorator,
)
from openhands.integrations.provider import ProviderType
from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.sdk.llm import MetricsSnapshot
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

logger = logging.getLogger(__name__)


class StoredConversationMetadata(Base):  # type: ignore
    __tablename__ = 'conversation_metadata'
    conversation_id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    selected_repository = Column(String, nullable=True)
    selected_branch = Column(String, nullable=True)
    git_provider = Column(
        String, nullable=True
    )  # The git provider (GitHub, GitLab, etc.)
    title = Column(String, nullable=True)
    last_updated_at = Column(DateTime(timezone=True), default=utc_now)  # type: ignore[attr-defined]
    created_at = Column(DateTime(timezone=True), default=utc_now)  # type: ignore[attr-defined]

    trigger = Column(String, nullable=True)
    pr_number = Column(create_json_type_decorator(list[int]))

    # Cost and token metrics
    accumulated_cost = Column(Float, default=0.0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    max_budget_per_task = Column(Float, nullable=True)
    cache_read_tokens = Column(Integer, default=0)
    cache_write_tokens = Column(Integer, default=0)
    reasoning_tokens = Column(Integer, default=0)
    context_window = Column(Integer, default=0)
    per_turn_token = Column(Integer, default=0)

    # LLM model used for the conversation
    llm_model = Column(String, nullable=True)

    conversation_version = Column(String, nullable=False, default='V0', index=True)
    sandbox_id = Column(String, nullable=True, index=True)
    parent_conversation_id = Column(String, nullable=True, index=True)
    public = Column(Boolean, nullable=True, index=True)


@dataclass
class SQLAppConversationInfoService(AppConversationInfoService):
    """SQL implementation of AppConversationInfoService focused on db operations.

    This allows storing a record of a conversation even after its sandbox ceases to exist
    """

    db_session: AsyncSession
    user_context: UserContext

    async def search_app_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sort_order: AppConversationSortOrder = AppConversationSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 100,
        include_sub_conversations: bool = False,
    ) -> AppConversationInfoPage:
        """Search for sandboxed conversations without permission checks."""
        query = await self._secure_select()

        # Conditionally exclude sub-conversations based on the parameter
        if not include_sub_conversations:
            # Exclude sub-conversations (only include top-level conversations)
            query = query.where(
                StoredConversationMetadata.parent_conversation_id.is_(None)
            )

        query = self._apply_filters(
            query=query,
            title__contains=title__contains,
            created_at__gte=created_at__gte,
            created_at__lt=created_at__lt,
            updated_at__gte=updated_at__gte,
            updated_at__lt=updated_at__lt,
        )

        # Add sort order
        if sort_order == AppConversationSortOrder.CREATED_AT:
            query = query.order_by(StoredConversationMetadata.created_at)
        elif sort_order == AppConversationSortOrder.CREATED_AT_DESC:
            query = query.order_by(StoredConversationMetadata.created_at.desc())
        elif sort_order == AppConversationSortOrder.UPDATED_AT:
            query = query.order_by(StoredConversationMetadata.last_updated_at)
        elif sort_order == AppConversationSortOrder.UPDATED_AT_DESC:
            query = query.order_by(StoredConversationMetadata.last_updated_at.desc())
        elif sort_order == AppConversationSortOrder.TITLE:
            query = query.order_by(StoredConversationMetadata.title)
        elif sort_order == AppConversationSortOrder.TITLE_DESC:
            query = query.order_by(StoredConversationMetadata.title.desc())

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

        result = await self.db_session.execute(query)
        rows = result.scalars().all()

        # Check if there are more results
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [self._to_info(row) for row in rows]

        # Calculate next page ID
        next_page_id = None
        if has_more:
            next_page_id = str(offset + limit)

        return AppConversationInfoPage(items=items, next_page_id=next_page_id)

    async def count_app_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
    ) -> int:
        """Count sandboxed conversations matching the given filters."""
        query = select(func.count(StoredConversationMetadata.conversation_id)).where(
            StoredConversationMetadata.conversation_version == 'V1'
        )

        query = self._apply_filters(
            query=query,
            title__contains=title__contains,
            created_at__gte=created_at__gte,
            created_at__lt=created_at__lt,
            updated_at__gte=updated_at__gte,
            updated_at__lt=updated_at__lt,
        )

        result = await self.db_session.execute(query)
        count = result.scalar()
        return count or 0

    def _apply_filters(
        self,
        query: Select,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
    ) -> Select:
        # Apply the same filters as search_app_conversations
        conditions = []
        if title__contains is not None:
            conditions.append(
                StoredConversationMetadata.title.like(f'%{title__contains}%')
            )

        if created_at__gte is not None:
            conditions.append(StoredConversationMetadata.created_at >= created_at__gte)

        if created_at__lt is not None:
            conditions.append(StoredConversationMetadata.created_at < created_at__lt)

        if updated_at__gte is not None:
            conditions.append(
                StoredConversationMetadata.last_updated_at >= updated_at__gte
            )

        if updated_at__lt is not None:
            conditions.append(
                StoredConversationMetadata.last_updated_at < updated_at__lt
            )

        if conditions:
            query = query.where(*conditions)
        return query

    async def get_sub_conversation_ids(
        self, parent_conversation_id: UUID
    ) -> list[UUID]:
        """Get all sub-conversation IDs for a given parent conversation.

        Args:
            parent_conversation_id: The ID of the parent conversation

        Returns:
            List of sub-conversation IDs
        """
        query = await self._secure_select()
        query = query.where(
            StoredConversationMetadata.parent_conversation_id
            == str(parent_conversation_id)
        )
        result_set = await self.db_session.execute(query)
        rows = result_set.scalars().all()
        return [UUID(row.conversation_id) for row in rows]

    async def get_app_conversation_info(
        self, conversation_id: UUID
    ) -> AppConversationInfo | None:
        query = await self._secure_select()
        query = query.where(
            StoredConversationMetadata.conversation_id == str(conversation_id)
        )
        result_set = await self.db_session.execute(query)
        result = result_set.scalar_one_or_none()
        if result:
            # Fetch sub-conversation IDs
            sub_conversation_ids = await self.get_sub_conversation_ids(conversation_id)
            return self._to_info(result, sub_conversation_ids=sub_conversation_ids)
        return None

    async def batch_get_app_conversation_info(
        self, conversation_ids: list[UUID]
    ) -> list[AppConversationInfo | None]:
        conversation_id_strs = [
            str(conversation_id) for conversation_id in conversation_ids
        ]
        query = await self._secure_select()
        query = query.where(
            StoredConversationMetadata.conversation_id.in_(conversation_id_strs)
        )
        result = await self.db_session.execute(query)
        rows = result.scalars().all()
        info_by_id = {info.conversation_id: info for info in rows if info}
        results: list[AppConversationInfo | None] = []
        for conversation_id in conversation_id_strs:
            info = info_by_id.get(conversation_id)
            sub_conversation_ids = await self.get_sub_conversation_ids(
                UUID(conversation_id)
            )
            if info:
                results.append(
                    self._to_info(info, sub_conversation_ids=sub_conversation_ids)
                )
            else:
                results.append(None)

        return results

    async def save_app_conversation_info(
        self, info: AppConversationInfo
    ) -> AppConversationInfo:
        metrics = info.metrics or MetricsSnapshot()
        usage = metrics.accumulated_token_usage or TokenUsage()

        stored = StoredConversationMetadata(
            conversation_id=str(info.id),
            selected_repository=info.selected_repository,
            selected_branch=info.selected_branch,
            git_provider=info.git_provider.value if info.git_provider else None,
            title=info.title,
            last_updated_at=info.updated_at,
            created_at=info.created_at,
            trigger=info.trigger.value if info.trigger else None,
            pr_number=info.pr_number or [],
            # Cost and token metrics
            accumulated_cost=metrics.accumulated_cost,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=0,
            max_budget_per_task=metrics.max_budget_per_task,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            context_window=usage.context_window,
            per_turn_token=usage.per_turn_token,
            llm_model=info.llm_model,
            conversation_version='V1',
            sandbox_id=info.sandbox_id,
            parent_conversation_id=(
                str(info.parent_conversation_id)
                if info.parent_conversation_id
                else None
            ),
            public=info.public,
        )

        await self.db_session.merge(stored)
        await self.db_session.commit()
        return info

    async def update_conversation_statistics(
        self, conversation_id: UUID, stats: ConversationStats
    ) -> None:
        """Update conversation statistics from stats event data.

        Args:
            conversation_id: The ID of the conversation to update
            stats: ConversationStats object containing usage_to_metrics data from stats event
        """
        # Extract agent metrics from usage_to_metrics
        usage_to_metrics = stats.usage_to_metrics
        agent_metrics = usage_to_metrics.get('agent')

        if not agent_metrics:
            logger.debug(
                'No agent metrics found in stats for conversation %s', conversation_id
            )
            return

        # Query existing record using secure select (filters for V1 and user if available)
        query = await self._secure_select()
        query = query.where(
            StoredConversationMetadata.conversation_id == str(conversation_id)
        )
        result = await self.db_session.execute(query)
        stored = result.scalar_one_or_none()

        if not stored:
            logger.debug(
                'Conversation %s not found or not accessible, skipping statistics update',
                conversation_id,
            )
            return

        # Extract accumulated_cost and max_budget_per_task from Metrics object
        accumulated_cost = agent_metrics.accumulated_cost
        max_budget_per_task = agent_metrics.max_budget_per_task

        # Extract accumulated_token_usage from Metrics object
        accumulated_token_usage = agent_metrics.accumulated_token_usage
        if accumulated_token_usage:
            prompt_tokens = accumulated_token_usage.prompt_tokens
            completion_tokens = accumulated_token_usage.completion_tokens
            cache_read_tokens = accumulated_token_usage.cache_read_tokens
            cache_write_tokens = accumulated_token_usage.cache_write_tokens
            reasoning_tokens = accumulated_token_usage.reasoning_tokens
            context_window = accumulated_token_usage.context_window
            per_turn_token = accumulated_token_usage.per_turn_token
        else:
            prompt_tokens = None
            completion_tokens = None
            cache_read_tokens = None
            cache_write_tokens = None
            reasoning_tokens = None
            context_window = None
            per_turn_token = None

        # Update fields only if values are provided (not None)
        if accumulated_cost is not None:
            stored.accumulated_cost = accumulated_cost
        if max_budget_per_task is not None:
            stored.max_budget_per_task = max_budget_per_task
        if prompt_tokens is not None:
            stored.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            stored.completion_tokens = completion_tokens
        if cache_read_tokens is not None:
            stored.cache_read_tokens = cache_read_tokens
        if cache_write_tokens is not None:
            stored.cache_write_tokens = cache_write_tokens
        if reasoning_tokens is not None:
            stored.reasoning_tokens = reasoning_tokens
        if context_window is not None:
            stored.context_window = context_window
        if per_turn_token is not None:
            stored.per_turn_token = per_turn_token

        # Update last_updated_at timestamp
        stored.last_updated_at = utc_now()

        await self.db_session.commit()

    async def process_stats_event(
        self,
        event: ConversationStateUpdateEvent,
        conversation_id: UUID,
    ) -> None:
        """Process a stats event and update conversation statistics.

        Args:
            event: The ConversationStateUpdateEvent with key='stats'
            conversation_id: The ID of the conversation to update
        """
        try:
            # Parse event value into ConversationStats model for type safety
            # event.value can be a dict (from JSON deserialization) or a ConversationStats object
            event_value = event.value
            conversation_stats: ConversationStats | None = None

            if isinstance(event_value, ConversationStats):
                # Already a ConversationStats object
                conversation_stats = event_value
            elif isinstance(event_value, dict):
                # Parse dict into ConversationStats model
                # This validates the structure and ensures type safety
                conversation_stats = ConversationStats.model_validate(event_value)
            elif hasattr(event_value, 'usage_to_metrics'):
                # Handle objects with usage_to_metrics attribute (e.g., from tests)
                # Convert to dict first, then validate
                stats_dict = {'usage_to_metrics': event_value.usage_to_metrics}
                conversation_stats = ConversationStats.model_validate(stats_dict)

            if conversation_stats and conversation_stats.usage_to_metrics:
                # Pass ConversationStats object directly for type safety
                await self.update_conversation_statistics(
                    conversation_id, conversation_stats
                )
        except Exception:
            logger.exception(
                'Error updating conversation statistics for conversation %s',
                conversation_id,
                stack_info=True,
            )

    async def _secure_select(self):
        query = select(StoredConversationMetadata).where(
            StoredConversationMetadata.conversation_version == 'V1'
        )
        return query

    def _to_info(
        self,
        stored: StoredConversationMetadata,
        sub_conversation_ids: list[UUID] | None = None,
    ) -> AppConversationInfo:
        # V1 conversations should always have a sandbox_id
        sandbox_id = stored.sandbox_id
        assert sandbox_id is not None

        # Rebuild token usage
        token_usage = TokenUsage(
            prompt_tokens=stored.prompt_tokens,
            completion_tokens=stored.completion_tokens,
            cache_read_tokens=stored.cache_read_tokens,
            cache_write_tokens=stored.cache_write_tokens,
            context_window=stored.context_window,
            per_turn_token=stored.per_turn_token,
        )

        # Rebuild metrics object
        metrics = MetricsSnapshot(
            accumulated_cost=stored.accumulated_cost,
            max_budget_per_task=stored.max_budget_per_task,
            accumulated_token_usage=token_usage,
        )

        # Get timestamps
        created_at = self._fix_timezone(stored.created_at)
        updated_at = self._fix_timezone(stored.last_updated_at)

        return AppConversationInfo(
            id=UUID(stored.conversation_id),
            created_by_user_id=None,  # User ID is now stored in ConversationMetadataSaas
            sandbox_id=stored.sandbox_id,
            selected_repository=stored.selected_repository,
            selected_branch=stored.selected_branch,
            git_provider=(
                ProviderType(stored.git_provider) if stored.git_provider else None
            ),
            title=stored.title,
            trigger=ConversationTrigger(stored.trigger) if stored.trigger else None,
            pr_number=stored.pr_number,
            llm_model=stored.llm_model,
            metrics=metrics,
            parent_conversation_id=(
                UUID(stored.parent_conversation_id)
                if stored.parent_conversation_id
                else None
            ),
            sub_conversation_ids=sub_conversation_ids or [],
            public=stored.public,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _fix_timezone(self, value: datetime) -> datetime:
        """Sqlite does not stpre timezones - and since we can't update the existing models
        we assume UTC if the timezone is missing."""
        if not value.tzinfo:
            value = value.replace(tzinfo=UTC)
        return value

    async def delete_app_conversation_info(self, conversation_id: UUID) -> bool:
        """Delete a conversation info from the database.

        Args:
            conversation_id: The ID of the conversation to delete.

        Returns True if the conversation was deleted successfully, False otherwise.
        """
        from sqlalchemy import delete

        # Build secure delete query with user context filtering
        delete_query = delete(StoredConversationMetadata).where(
            StoredConversationMetadata.conversation_id == str(conversation_id)
        )

        # Execute the secure delete query
        result = await self.db_session.execute(delete_query)

        return result.rowcount > 0


class SQLAppConversationInfoServiceInjector(AppConversationInfoServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[AppConversationInfoService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import (
            get_db_session,
            get_user_context,
        )

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            service = SQLAppConversationInfoService(
                db_session=db_session, user_context=user_context
            )
            yield service
