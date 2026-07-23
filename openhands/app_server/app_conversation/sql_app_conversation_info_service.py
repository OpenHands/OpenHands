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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncGenerator, cast
from uuid import UUID

from fastapi import Request
from sqlalchemy import (
    ColumnElement,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Integer,
    Select,
    String,
    func,
    select,
)
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from openhands.agent_server.utils import utc_now
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
    AppConversationInfoServiceInjector,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationInfoPage,
    AppConversationSortOrder,
    ConversationTrigger,
)
from openhands.app_server.integrations.provider import ProviderType
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.sql_utils import (
    Base,
    create_json_type_decorator,
)
from openhands.sdk import ConversationStats
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.sdk.llm import MetricsSnapshot, TokenUsage


def _parse_event_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith('Z'):
            value = f'{value[:-1]}+00:00'
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _normalize_event_timestamp(value: datetime | None) -> datetime:
    if value is None:
        return utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _combine_usage_metrics(
    usage_to_metrics: Mapping[str, MetricsSnapshot],
) -> tuple[float, TokenUsage | None]:
    """Sum cost and token usage across every usage bucket in the registry."""
    total_cost = 0.0
    combined: TokenUsage | None = None
    for snapshot in usage_to_metrics.values():
        total_cost += snapshot.accumulated_cost or 0.0
        usage = snapshot.accumulated_token_usage
        if usage is None:
            continue
        if combined is None:
            combined = usage.model_copy()
        else:
            combined = TokenUsage(
                model=combined.model,
                prompt_tokens=combined.prompt_tokens + usage.prompt_tokens,
                completion_tokens=combined.completion_tokens + usage.completion_tokens,
                cache_read_tokens=combined.cache_read_tokens + usage.cache_read_tokens,
                cache_write_tokens=combined.cache_write_tokens
                + usage.cache_write_tokens,
                reasoning_tokens=combined.reasoning_tokens + usage.reasoning_tokens,
                context_window=max(combined.context_window, usage.context_window),
                per_turn_token=combined.per_turn_token,
            )
    # context_window/per_turn_token are per-model snapshots, not additive;
    # prefer the primary agent's when present.
    agent_snapshot = usage_to_metrics.get('agent')
    agent_usage = agent_snapshot.accumulated_token_usage if agent_snapshot else None
    if combined is not None and agent_usage is not None:
        combined = combined.model_copy(
            update={
                'context_window': agent_usage.context_window,
                'per_turn_token': agent_usage.per_turn_token,
            }
        )
    return total_cost, combined


logger = logging.getLogger(__name__)


class StoredConversationMetadata(Base):
    __tablename__ = 'conversation_metadata'

    conversation_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    selected_repository: Mapped[str | None] = mapped_column(String, nullable=True)
    selected_branch: Mapped[str | None] = mapped_column(String, nullable=True)
    git_provider: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # The git provider (GitHub, GitLab, etc.)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    last_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    trigger: Mapped[str | None] = mapped_column(String, nullable=True)
    pr_number: Mapped[list[int] | None] = mapped_column(
        create_json_type_decorator(list[int])
    )

    # Cost and token metrics
    accumulated_cost: Mapped[float | None] = mapped_column(default=0.0)
    prompt_tokens: Mapped[int | None] = mapped_column(default=0)
    completion_tokens: Mapped[int | None] = mapped_column(default=0)
    total_tokens: Mapped[int | None] = mapped_column(default=0)
    max_budget_per_task: Mapped[float | None] = mapped_column(nullable=True)
    cache_read_tokens: Mapped[int | None] = mapped_column(default=0)
    cache_write_tokens: Mapped[int | None] = mapped_column(default=0)
    reasoning_tokens: Mapped[int | None] = mapped_column(default=0)
    context_window: Mapped[int | None] = mapped_column(default=0)
    per_turn_token: Mapped[int | None] = mapped_column(default=0)

    # LLM model used for the conversation
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    agent_kind: Mapped[str | None] = mapped_column(String, nullable=True)

    conversation_version: Mapped[str] = mapped_column(
        String, nullable=False, default='V0', index=True
    )
    sandbox_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    parent_conversation_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    public: Mapped[bool | None] = mapped_column(nullable=True, index=True)

    # Execution status: idle, running, paused, finished, error, stuck, deleting
    execution_status: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )

    # Tags for conversation metadata (e.g., automation context, skills used)
    tags: Mapped[dict[str, str] | None] = mapped_column(
        create_json_type_decorator(dict[str, str]), nullable=True
    )


class StoredConversationCostEvent(Base):
    __tablename__ = 'conversation_cost_events'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String,
        ForeignKey('conversation_metadata.conversation_id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    cost_delta: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    # Attribution is nullable for rows written before these columns existed.
    usage_id: Mapped[str | None] = mapped_column(String, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)


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
        sandbox_id__eq: str | None = None,
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
            sandbox_id__eq=sandbox_id__eq,
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
        sandbox_id__eq: str | None = None,
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
            sandbox_id__eq=sandbox_id__eq,
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
        sandbox_id__eq: str | None = None,
    ) -> Select:
        # Apply the same filters as search_app_conversations
        conditions: list[ColumnElement[bool]] = []
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

        if sandbox_id__eq is not None:
            conditions.append(StoredConversationMetadata.sandbox_id == sandbox_id__eq)

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

    async def count_conversations_by_sandbox_id(self, sandbox_id: str) -> int:
        query = await self._secure_select()
        query = query.where(StoredConversationMetadata.sandbox_id == sandbox_id)
        count_query = select(func.count()).select_from(query.subquery())
        result = await self.db_session.execute(count_query)
        count = result.scalar()
        return count or 0

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

        # Preserve the original creation time on update.
        #
        # ``save`` is an upsert (``merge`` on the primary key), and several
        # callers rebuild an ``AppConversationInfo`` from scratch rather than
        # mutating the loaded row (e.g. the ``on_conversation_update`` webhook,
        # which fires on every start/pause/resume/interrupt). Those callers do
        # not carry ``created_at`` forward, so ``AppConversationInfo`` fills it
        # from ``default_factory=utc_now``. Without this guard the merge would
        # overwrite the persisted ``created_at`` with "now" on every lifecycle
        # webhook, corrupting created-at ordering and the usage/billing
        # dashboards that bucket conversations by creation time. Look up the
        # stored value directly by primary key (not via ``_secure_select``) so
        # this works under the ADMIN webhook context as well.
        created_at = info.created_at
        existing_created_at = await self.db_session.scalar(
            select(StoredConversationMetadata.created_at).where(
                StoredConversationMetadata.conversation_id == str(info.id)
            )
        )
        if existing_created_at is not None:
            created_at = existing_created_at

        stored = StoredConversationMetadata(
            conversation_id=str(info.id),
            selected_repository=info.selected_repository,
            selected_branch=info.selected_branch,
            git_provider=info.git_provider.value if info.git_provider else None,
            title=info.title,
            last_updated_at=info.updated_at,
            created_at=created_at,
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
            agent_kind=info.agent_kind,
            conversation_version='V1',
            sandbox_id=info.sandbox_id,
            parent_conversation_id=(
                str(info.parent_conversation_id)
                if info.parent_conversation_id
                else None
            ),
            public=info.public,
            tags=info.tags if info.tags else None,
        )

        await self.db_session.merge(stored)
        await self.db_session.commit()
        return info

    async def update_conversation_statistics(
        self,
        conversation_id: UUID,
        stats: ConversationStats,
        event_timestamp: datetime | None = None,
    ) -> None:
        """Update conversation statistics from stats event data.

        Args:
            conversation_id: The ID of the conversation to update
            stats: ConversationStats object containing usage_to_metrics data from stats event
            event_timestamp: Timestamp of the stats event (UTC if naive)
        """
        usage_to_metrics = stats.usage_to_metrics
        if not usage_to_metrics:
            logger.debug(
                'No usage metrics found in stats for conversation %s', conversation_id
            )
            return

        # Query existing record using secure select (filters for V1 and user if available)
        # Row-lock so concurrent snapshots (stats events, run-end pull)
        # serialize per conversation instead of racing the guard/ledger.
        # No-op on SQLite.
        query = await self._secure_select()
        query = query.where(
            StoredConversationMetadata.conversation_id == str(conversation_id)
        ).with_for_update()
        result = await self.db_session.execute(query)
        stored = result.scalar_one_or_none()

        if not stored:
            logger.debug(
                'Conversation %s not found or not accessible, skipping statistics update',
                conversation_id,
            )
            return

        event_timestamp = _normalize_event_timestamp(event_timestamp)

        # Combine ALL usage buckets: switched-in LLMs ("profile:*") and ACP
        # agents ("acp-managed") accrue spend outside the "agent" bucket, so
        # an agent-only read freezes persisted cost at the pre-switch value.
        accumulated_cost, accumulated_token_usage = _combine_usage_metrics(
            usage_to_metrics
        )
        agent_metrics = usage_to_metrics.get('agent')

        previous_cost = stored.accumulated_cost or 0.0
        delta_cost = accumulated_cost - float(previous_cost)
        if delta_cost < 0:
            # Stale/out-of-order snapshot; never regress the running total.
            logger.debug(
                'Accumulated cost decreased for conversation %s (prev=%s, new=%s)',
                conversation_id,
                previous_cost,
                accumulated_cost,
            )
            return
        # Ledger deltas are computed against the ledger's own per-bucket sums
        # (not the column), so it self-heals if the column ever advances
        # through a path that skips the ledger.
        await self._record_bucket_cost_deltas(stored, usage_to_metrics, event_timestamp)

        stored.accumulated_cost = accumulated_cost
        if agent_metrics is not None and agent_metrics.max_budget_per_task is not None:
            stored.max_budget_per_task = agent_metrics.max_budget_per_task
        # Preserve each cumulative counter independently when snapshots arrive
        # partially out of order.
        if accumulated_token_usage is not None:
            stored.prompt_tokens = max(
                accumulated_token_usage.prompt_tokens, stored.prompt_tokens or 0
            )
            stored.completion_tokens = max(
                accumulated_token_usage.completion_tokens,
                stored.completion_tokens or 0,
            )
            stored.cache_read_tokens = max(
                accumulated_token_usage.cache_read_tokens,
                stored.cache_read_tokens or 0,
            )
            stored.cache_write_tokens = max(
                accumulated_token_usage.cache_write_tokens,
                stored.cache_write_tokens or 0,
            )
            stored.reasoning_tokens = max(
                accumulated_token_usage.reasoning_tokens,
                stored.reasoning_tokens or 0,
            )
            # Gauges, not counters: only follow snapshots that are at least
            # as fresh as the stored totals.
            if (
                accumulated_token_usage.prompt_tokens
                + accumulated_token_usage.completion_tokens
                >= (stored.prompt_tokens or 0) + (stored.completion_tokens or 0)
            ):
                stored.context_window = accumulated_token_usage.context_window
                stored.per_turn_token = accumulated_token_usage.per_turn_token

        # Update last_updated_at timestamp
        stored.last_updated_at = utc_now()

        await self.db_session.commit()

    async def _record_bucket_cost_deltas(
        self,
        stored: StoredConversationMetadata,
        usage_to_metrics: Mapping[str, MetricsSnapshot],
        event_timestamp: datetime,
    ) -> None:
        """Write per-bucket deltas without replaying legacy unattributed cost."""
        result = await self.db_session.execute(
            select(
                StoredConversationCostEvent.usage_id,
                func.sum(StoredConversationCostEvent.cost_delta),
                func.sum(StoredConversationCostEvent.prompt_tokens),
                func.sum(StoredConversationCostEvent.completion_tokens),
            )
            .where(
                StoredConversationCostEvent.conversation_id == stored.conversation_id
            )
            .group_by(StoredConversationCostEvent.usage_id)
        )
        prior = {row[0]: row for row in result.all()}
        unattributed = prior.pop(None, None)
        cost_drain = float(unattributed[1] or 0.0) if unattributed is not None else 0.0

        pending: list[dict] = []
        for usage_id, snapshot in usage_to_metrics.items():
            prior_row = prior.get(usage_id)
            prior_cost = float(prior_row[1] or 0.0) if prior_row is not None else 0.0
            prior_prompt = int(prior_row[2] or 0) if prior_row is not None else 0
            prior_completion = int(prior_row[3] or 0) if prior_row is not None else 0
            usage = snapshot.accumulated_token_usage
            pending.append(
                {
                    'usage_id': usage_id,
                    'snapshot': snapshot,
                    'cost': max(
                        float(snapshot.accumulated_cost or 0.0) - prior_cost, 0.0
                    ),
                    'prompt': max(usage.prompt_tokens - prior_prompt, 0)
                    if usage
                    else 0,
                    'completion': max(usage.completion_tokens - prior_completion, 0)
                    if usage
                    else 0,
                    'has_usage': usage is not None,
                }
            )

        for item in sorted(pending, key=lambda i: i['cost'], reverse=True):
            covered = min(item['cost'], cost_drain)
            item['cost'] -= covered
            cost_drain -= covered

        for item in pending:
            if item['cost'] <= 0 and item['prompt'] <= 0 and item['completion'] <= 0:
                continue
            snapshot = item['snapshot']
            model: str | None = snapshot.model_name
            if not model or model == 'default':
                model = stored.llm_model if item['usage_id'] == 'agent' else None
            self.db_session.add(
                StoredConversationCostEvent(
                    conversation_id=stored.conversation_id,
                    cost_delta=item['cost'],
                    occurred_at=event_timestamp,
                    usage_id=item['usage_id'],
                    llm_model=model,
                    prompt_tokens=item['prompt'] if item['has_usage'] else None,
                    completion_tokens=item['completion'] if item['has_usage'] else None,
                )
            )

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

            event_timestamp = _parse_event_timestamp(event.timestamp)

            if conversation_stats and conversation_stats.usage_to_metrics:
                # Pass ConversationStats object directly for type safety
                await self.update_conversation_statistics(
                    conversation_id,
                    conversation_stats,
                    event_timestamp=event_timestamp,
                )
        except Exception:
            logger.exception(
                'Error updating conversation statistics for conversation %s',
                conversation_id,
                stack_info=True,
            )

    async def update_execution_status(
        self,
        conversation_id: UUID,
        execution_status: str,
    ) -> None:
        """Update the execution status for a conversation.

        Args:
            conversation_id: The ID of the conversation to update
            execution_status: The new execution status value
        """
        query = await self._secure_select()
        query = query.where(
            StoredConversationMetadata.conversation_id == str(conversation_id)
        )
        result = await self.db_session.execute(query)
        stored = result.scalar_one_or_none()

        if not stored:
            logger.debug(
                'Conversation %s not found or not accessible, skipping execution status update',
                conversation_id,
            )
            return

        stored.execution_status = execution_status
        stored.last_updated_at = utc_now()
        await self.db_session.commit()

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

        # Rebuild token usage (use 0 as default for nullable int columns)
        token_usage = TokenUsage(
            prompt_tokens=stored.prompt_tokens or 0,
            completion_tokens=stored.completion_tokens or 0,
            cache_read_tokens=stored.cache_read_tokens or 0,
            cache_write_tokens=stored.cache_write_tokens or 0,
            context_window=stored.context_window or 0,
            per_turn_token=stored.per_turn_token or 0,
        )

        # Rebuild metrics object (use 0.0 as default for nullable float columns)
        metrics = MetricsSnapshot(
            accumulated_cost=stored.accumulated_cost or 0.0,
            max_budget_per_task=stored.max_budget_per_task,
            accumulated_token_usage=token_usage,
        )

        # Get timestamps
        created_at = self._fix_timezone(stored.created_at)
        updated_at = self._fix_timezone(stored.last_updated_at)

        return AppConversationInfo(
            id=UUID(stored.conversation_id),
            created_by_user_id=None,  # User ID is now stored in ConversationMetadataSaas
            sandbox_id=sandbox_id,  # Use the asserted non-None value
            selected_repository=stored.selected_repository,
            selected_branch=stored.selected_branch,
            git_provider=(
                ProviderType(stored.git_provider) if stored.git_provider else None
            ),
            title=stored.title,
            trigger=ConversationTrigger(stored.trigger) if stored.trigger else None,
            pr_number=stored.pr_number or [],
            llm_model=stored.llm_model,
            agent_kind=stored.agent_kind or 'openhands',
            metrics=metrics,
            parent_conversation_id=(
                UUID(stored.parent_conversation_id)
                if stored.parent_conversation_id
                else None
            ),
            sub_conversation_ids=sub_conversation_ids or [],
            public=stored.public,
            tags=stored.tags or {},
            created_at=created_at,
            updated_at=updated_at,
        )

    def _fix_timezone(self, value: datetime | None) -> datetime:
        """Sqlite does not store timezones - and since we can't update the existing models
        we assume UTC if the timezone is missing. Returns current UTC time if value is None.
        """
        if value is None:
            # Fallback for legacy data: use current time to match model defaults.
            # The DB columns have default=utc_now, so None only occurs in legacy records.
            # Using utc_now() keeps the API model non-nullable and matches new record behavior.
            return utc_now()
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
        result = cast(CursorResult, await self.db_session.execute(delete_query))

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
