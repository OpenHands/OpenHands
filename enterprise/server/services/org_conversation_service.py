"""Service class for organization conversation listing.

Separates business logic from route handlers.
Uses dependency injection for db_session and sandbox_service.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from server.routes.org_models import (
    AgentUsageData,
    DailyUsageData,
    ModelUsageData,
    OrgConversationPage,
    OrgConversationResponse,
    OrgConversationStats,
    OrgUsageStats,
    OrgUserUsageRow,
    OrgUserUsageStats,
    TeamUsageData,
)
from sqlalchemy import case, func, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from storage.openhands_pr import OpenhandsPR
from storage.org_budget_settings import OrgBudgetSettings
from storage.org_user_budget_override import OrgUserBudgetOverride
from storage.stored_conversation_cost_event import StoredConversationCostEvent
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user import User

from openhands.app_server.sandbox.sandbox_models import AGENT_SERVER, SandboxInfo
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.llm import MetricsSnapshot, TokenUsage

# Valid sort fields
VALID_SORT_FIELDS = {
    'created_at': StoredConversationMetadata.created_at,
    'updated_at': StoredConversationMetadata.last_updated_at,
    'llm_model': StoredConversationMetadata.llm_model,
    'accumulated_cost': StoredConversationMetadata.accumulated_cost,
    'title': StoredConversationMetadata.title,
}

# Time window options (in days)
TIME_WINDOW_OPTIONS = {
    '7d': 7,
    '30d': 30,
    '90d': 90,
}


AGENT_LABELS = {
    'openhands': 'OpenHands',
    'acp': 'ACP',
}


def _format_acp_agent_label(llm_model: str | None) -> str:
    if not llm_model:
        return AGENT_LABELS['acp']
    llm_model_lower = llm_model.lower()
    if 'claude' in llm_model_lower:
        return 'Claude'
    if 'codex' in llm_model_lower:
        return 'Codex'
    if 'gpt' in llm_model_lower or 'openai' in llm_model_lower:
        return 'OpenAI'
    if 'gemini' in llm_model_lower:
        return 'Gemini'
    return llm_model


def _format_agent_label(agent_kind: str | None, llm_model: str | None) -> str:
    if agent_kind == 'acp':
        return _format_acp_agent_label(llm_model)
    if not agent_kind:
        return AGENT_LABELS['openhands']
    return AGENT_LABELS.get(agent_kind, agent_kind)


MAX_SANDBOX_STATUS_FILTER_ROWS = 5000


class OrgConversationFilterError(ValueError):
    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass
class OrgConversationService:
    """Service for organization conversations with injected dependencies."""

    db_session: AsyncSession
    sandbox_service = None  # Optional: SandboxService for live status

    def set_sandbox_service(self, sandbox_service):
        """Set the sandbox service for live status fetching."""
        self.sandbox_service = sandbox_service

    def _build_conversation_response(
        self,
        metadata: StoredConversationMetadata,
        saas_metadata: StoredConversationMetadataSaas,
        user: User | None,
        sandbox_info: SandboxInfo | None,
        pr_merged: bool | None = None,
    ) -> OrgConversationResponse:
        """Build an OrgConversationResponse from a row and sandbox info."""
        resolved_sandbox_status = sandbox_info.status.value if sandbox_info else None

        # Construct runtime URL from exposed URLs
        runtime_url = None
        if sandbox_info and sandbox_info.exposed_urls:
            agent_server_url = next(
                (
                    exposed_url.url
                    for exposed_url in sandbox_info.exposed_urls
                    if exposed_url.name == AGENT_SERVER
                ),
                None,
            )
            if agent_server_url:
                runtime_url = (
                    f'{agent_server_url}/api/conversations/{metadata.conversation_id}'
                )

        # Build metrics
        token_usage = TokenUsage(
            prompt_tokens=metadata.prompt_tokens or 0,
            completion_tokens=metadata.completion_tokens or 0,
            cache_read_tokens=metadata.cache_read_tokens or 0,
            cache_write_tokens=metadata.cache_write_tokens or 0,
            context_window=metadata.context_window or 0,
            per_turn_token=metadata.per_turn_token or 0,
        )
        metrics = MetricsSnapshot(
            accumulated_cost=metadata.accumulated_cost or 0.0,
            max_budget_per_task=metadata.max_budget_per_task,
            accumulated_token_usage=token_usage,
        )

        return OrgConversationResponse(
            id=metadata.conversation_id,
            title=metadata.title,
            llm_model=metadata.llm_model,
            agent_kind=metadata.agent_kind or 'openhands',
            user_id=str(saas_metadata.user_id),
            user_email=user.email if user else None,
            created_at=metadata.created_at,
            updated_at=metadata.last_updated_at,
            sandbox_id=metadata.sandbox_id,
            sandbox_status=resolved_sandbox_status,
            runtime_url=runtime_url,
            execution_status=metadata.execution_status,
            selected_repository=metadata.selected_repository,
            selected_branch=metadata.selected_branch,
            git_provider=metadata.git_provider,
            trigger=metadata.trigger,
            pr_number=metadata.pr_number or [],
            pr_merged=pr_merged,
            tags=metadata.tags or {},
            accumulated_cost=metrics.accumulated_cost,
            prompt_tokens=metrics.accumulated_token_usage.prompt_tokens,  # type: ignore[union-attr]
            completion_tokens=metrics.accumulated_token_usage.completion_tokens,  # type: ignore[union-attr]
            total_tokens=(
                metrics.accumulated_token_usage.prompt_tokens  # type: ignore[union-attr]
                + metrics.accumulated_token_usage.completion_tokens  # type: ignore[union-attr]
            ),
            cache_read_tokens=metrics.accumulated_token_usage.cache_read_tokens,  # type: ignore[union-attr]
            cache_write_tokens=metrics.accumulated_token_usage.cache_write_tokens,  # type: ignore[union-attr]
        )

    async def _count_conversations_by_sandbox_id(self, sandbox_id: str) -> int:
        count_query = (
            select(func.count())
            .select_from(StoredConversationMetadata)
            .where(
                StoredConversationMetadata.conversation_version == 'V1',
                StoredConversationMetadata.sandbox_id == sandbox_id,
            )
        )
        result = await self.db_session.execute(count_query)
        return result.scalar() or 0

    async def _load_pr_merge_map(
        self, pr_keys: set[tuple[str, str, int]]
    ) -> dict[tuple[str, str, int], bool | None]:
        if not pr_keys:
            return {}

        query = select(
            OpenhandsPR.provider,
            OpenhandsPR.repo_name,
            OpenhandsPR.pr_number,
            OpenhandsPR.merged,
        ).where(
            tuple_(
                OpenhandsPR.provider, OpenhandsPR.repo_name, OpenhandsPR.pr_number
            ).in_(pr_keys)
        )
        result = await self.db_session.execute(query)
        return {
            (provider, repo_name, pr_number): merged
            for provider, repo_name, pr_number, merged in result.all()
        }

    def _resolve_pr_merged(
        self,
        metadata: StoredConversationMetadata,
        pr_map: dict[tuple[str, str, int], bool | None],
    ) -> bool | None:
        if not metadata.pr_number:
            return None
        if not metadata.selected_repository or not metadata.git_provider:
            return None

        statuses: list[bool | None] = []
        for pr_number in metadata.pr_number or []:
            key = (metadata.git_provider, metadata.selected_repository, pr_number)
            if key in pr_map:
                statuses.append(pr_map[key])

        if not statuses:
            return None
        if any(status is True for status in statuses):
            return True
        if any(status is False for status in statuses):
            return False
        return None

    async def list_org_conversations(
        self,
        org_id: UUID,
        search: str | None = None,
        sort_by: str = 'updated_at',
        sort_order: str = 'desc',
        execution_status: list[str] | None = None,
        sandbox_status: list[str] | None = None,
        time_window: str | None = None,
        page: int = 1,
        per_page: int = 20,
        include_sub_conversations: bool = False,
    ) -> OrgConversationPage:
        """List all conversations for an organization with filtering, sorting, and pagination.

        Args:
            org_id: The organization ID
            search: Search text matching conversation name, creator name, email, or sandbox ID
            sort_by: Field to sort by (created_at, updated_at, llm_model, accumulated_cost, title)
            sort_order: Sort order ('desc' or 'asc')
            execution_status: Filter by execution status values
            sandbox_status: Filter by sandbox status values
            time_window: Time window filter ('7d', '30d', '90d', or None for all)
            page: Page number (1-indexed)
            per_page: Items per page (1-100)
            include_sub_conversations: If True, include sub-conversations

        Returns:
            OrgConversationPage: Paginated list of conversations with total count
        """
        # Base query with joins
        query = (
            select(StoredConversationMetadata, StoredConversationMetadataSaas, User)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .outerjoin(User, StoredConversationMetadataSaas.user_id == User.id)
            .where(StoredConversationMetadata.conversation_version == 'V1')
            .where(StoredConversationMetadataSaas.org_id == org_id)
        )

        # Exclude sub-conversations unless explicitly requested
        if not include_sub_conversations:
            query = query.where(
                StoredConversationMetadata.parent_conversation_id.is_(None)
            )

        # Apply search filter
        if search:
            search_pattern = f'%{search}%'
            query = query.where(
                or_(
                    StoredConversationMetadata.title.ilike(search_pattern),
                    StoredConversationMetadata.sandbox_id.ilike(search_pattern),
                    User.email.ilike(search_pattern),
                    User.git_user_name.ilike(search_pattern),
                )
            )

        # Apply execution_status filter
        if execution_status:
            query = query.where(
                StoredConversationMetadata.execution_status.in_(execution_status)
            )

        # Apply time_window filter
        if time_window and time_window in TIME_WINDOW_OPTIONS:
            days = TIME_WINDOW_OPTIONS[time_window]
            cutoff_date = datetime.now(UTC) - timedelta(days=days)
            query = query.where(StoredConversationMetadata.created_at >= cutoff_date)

        # Apply sorting
        sort_column = VALID_SORT_FIELDS.get(
            sort_by, StoredConversationMetadata.last_updated_at
        )
        if sort_order.lower() == 'asc':
            query = query.order_by(sort_column.asc().nullslast())
        else:
            query = query.order_by(sort_column.desc().nullslast())

        pagination_accurate = not (sandbox_status and not self.sandbox_service)
        sandbox_info_map: dict[str, SandboxInfo | None] = {}

        if sandbox_status:
            limited_query = query.limit(MAX_SANDBOX_STATUS_FILTER_ROWS + 1)
            result = await self.db_session.execute(limited_query)
            rows = result.all()

            if len(rows) > MAX_SANDBOX_STATUS_FILTER_ROWS:
                raise OrgConversationFilterError(
                    'Sandbox status filter is too broad; refine filters or remove '
                    'the sandbox_status filter.',
                    error_code='sandbox_status_too_broad',
                )

            sandbox_ids = [
                metadata.sandbox_id for metadata, _, _ in rows if metadata.sandbox_id
            ]

            if sandbox_ids and self.sandbox_service:
                try:
                    sandbox_results = await self.sandbox_service.batch_get_sandboxes(
                        sandbox_ids
                    )
                    for sandbox_id, sandbox_info in zip(sandbox_ids, sandbox_results):
                        sandbox_info_map[sandbox_id] = sandbox_info
                except Exception as e:
                    logger.warning(
                        'Failed to fetch sandbox info for org conversations',
                        extra={'org_id': str(org_id), 'error': str(e)},
                    )

            rows = [
                row
                for row in rows
                if row[0].sandbox_id
                and sandbox_info_map.get(row[0].sandbox_id)
                and sandbox_info_map[row[0].sandbox_id].status.value in sandbox_status  # type: ignore[union-attr]
            ]

            total_items = len(rows)
            total_pages = math.ceil(total_items / per_page) if total_items > 0 else 0

            offset = (page - 1) * per_page
            rows = rows[offset : offset + per_page]
        else:
            count_query = select(func.count()).select_from(query.subquery())
            count_result = await self.db_session.execute(count_query)
            total_items = count_result.scalar() or 0
            total_pages = math.ceil(total_items / per_page) if total_items > 0 else 0

            offset = (page - 1) * per_page
            query = query.offset(offset).limit(per_page)

            result = await self.db_session.execute(query)
            rows = result.all()

            sandbox_ids = [
                metadata.sandbox_id for metadata, _, _ in rows if metadata.sandbox_id
            ]

            if sandbox_ids and self.sandbox_service:
                try:
                    sandbox_results = await self.sandbox_service.batch_get_sandboxes(
                        sandbox_ids
                    )
                    for sandbox_id, sandbox_info in zip(sandbox_ids, sandbox_results):
                        sandbox_info_map[sandbox_id] = sandbox_info
                except Exception as e:
                    logger.warning(
                        'Failed to fetch sandbox info for org conversations',
                        extra={'org_id': str(org_id), 'error': str(e)},
                    )

        pr_keys: set[tuple[str, str, int]] = set()
        for metadata, _, _ in rows:
            if (
                metadata.pr_number
                and metadata.selected_repository
                and metadata.git_provider
            ):
                for pr_number in metadata.pr_number:
                    pr_keys.add(
                        (metadata.git_provider, metadata.selected_repository, pr_number)
                    )

        pr_merge_map = await self._load_pr_merge_map(pr_keys)

        # Build response items
        items: list[OrgConversationResponse] = []
        for metadata, saas_metadata, user in rows:
            sandbox_info = sandbox_info_map.get(metadata.sandbox_id)
            items.append(
                self._build_conversation_response(
                    metadata,
                    saas_metadata,
                    user,
                    sandbox_info,
                    pr_merged=self._resolve_pr_merged(metadata, pr_merge_map),
                )
            )

        logger.info(
            'Listed organization conversations',
            extra={
                'org_id': str(org_id),
                'count': len(items),
                'total_items': total_items,
                'page': page,
                'per_page': per_page,
            },
        )

        return OrgConversationPage(
            items=items,
            total_items=total_items,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            pagination_accurate=pagination_accurate,
        )

    async def get_stats(
        self,
        org_id: UUID,
    ) -> OrgConversationStats:
        """Get aggregated statistics for organization conversations.

        Args:
            org_id: The organization ID

        Returns:
            OrgConversationStats with aggregated stats for the org
        """
        now = datetime.now(UTC)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_7d = now - timedelta(days=7)
        cutoff_30d = now - timedelta(days=30)

        # Base query for org conversations
        base_filter = [
            StoredConversationMetadata.conversation_version == 'V1',
            StoredConversationMetadataSaas.org_id == org_id,
        ]

        # 1. Active conversations (execution_status = 'running')
        active_query = (
            select(func.count(StoredConversationMetadata.conversation_id))
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.execution_status == 'running')
        )
        result = await self.db_session.execute(active_query)
        active_conversations = result.scalar() or 0

        # 2. Aggregate cost and tokens (all time)
        aggregate_query = (
            select(
                func.coalesce(func.sum(StoredConversationMetadata.accumulated_cost), 0),
                func.coalesce(func.sum(StoredConversationMetadata.prompt_tokens), 0),
                func.coalesce(
                    func.sum(StoredConversationMetadata.completion_tokens), 0
                ),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
        )
        result = await self.db_session.execute(aggregate_query)
        total_cost, total_prompt, total_completion = result.one()

        # 3. Completed in 24h, 7d, 30d (terminal status = finished, error, stuck)
        terminal_statuses = ['finished', 'error', 'stuck']

        # Helper to count completed conversations since a given cutoff
        async def count_completed(cutoff: datetime) -> int:
            completed_query = (
                select(func.count(StoredConversationMetadata.conversation_id))
                .select_from(StoredConversationMetadata)
                .join(
                    StoredConversationMetadataSaas,
                    StoredConversationMetadata.conversation_id
                    == StoredConversationMetadataSaas.conversation_id,
                )
                .where(*base_filter)
                .where(
                    StoredConversationMetadata.execution_status.in_(terminal_statuses)
                )
                .where(StoredConversationMetadata.last_updated_at >= cutoff)
            )
            res = await self.db_session.execute(completed_query)
            return res.scalar() or 0

        completed_24h = await count_completed(cutoff_24h)
        completed_7d = await count_completed(cutoff_7d)
        completed_30d = await count_completed(cutoff_30d)

        # 4. Running runtimes (requires sandbox service)
        running_runtimes = 0
        if self.sandbox_service:
            try:
                # Get distinct sandbox IDs from active conversations
                sandbox_ids_query = (
                    select(StoredConversationMetadata.sandbox_id)
                    .select_from(StoredConversationMetadata)
                    .join(
                        StoredConversationMetadataSaas,
                        StoredConversationMetadata.conversation_id
                        == StoredConversationMetadataSaas.conversation_id,
                    )
                    .where(*base_filter)
                    .where(StoredConversationMetadata.execution_status == 'running')
                    .where(StoredConversationMetadata.sandbox_id.isnot(None))
                    .distinct()
                )
                result = await self.db_session.execute(sandbox_ids_query)
                sandbox_ids = [row[0] for row in result.all()]

                # Batch fetch all sandbox statuses in one call
                if sandbox_ids:
                    sandbox_results = await self.sandbox_service.batch_get_sandboxes(
                        sandbox_ids
                    )
                    for sandbox in sandbox_results:
                        if sandbox and sandbox.status.value == 'RUNNING':
                            running_runtimes += 1
            except Exception as e:
                logger.warning(
                    'Failed to get running runtimes count',
                    extra={'org_id': str(org_id), 'error': str(e)},
                )

        return OrgConversationStats(
            active_conversations=active_conversations,
            running_runtimes=running_runtimes,
            completed_24h=completed_24h,
            completed_7d=completed_7d,
            completed_30d=completed_30d,
            total_cost=float(total_cost or 0),
            total_prompt_tokens=int(total_prompt or 0),
            total_completion_tokens=int(total_completion or 0),
            total_tokens=int((total_prompt or 0) + (total_completion or 0)),
        )

    async def get_usage_stats(
        self,
        org_id: UUID,
        days: int = 7,
    ) -> OrgUsageStats:
        """Get detailed usage statistics for organization dashboard.

        Args:
            org_id: The organization ID
            days: Number of days to look back (default 7)

        Returns:
            OrgUsageStats with detailed usage data
        """
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=days)

        # Base filter for org conversations
        base_filter = [
            StoredConversationMetadata.conversation_version == 'V1',
            StoredConversationMetadataSaas.org_id == org_id,
        ]

        # 1. Active users (users with activity in the time window)
        active_users_query = (
            select(func.count(func.distinct(StoredConversationMetadataSaas.user_id)))
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.created_at >= cutoff)
        )
        result = await self.db_session.execute(active_users_query)
        active_users = result.scalar() or 0

        # 2. Total agent runs (conversations) in time window
        agent_runs_query = (
            select(func.count(StoredConversationMetadata.conversation_id))
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.created_at >= cutoff)
        )
        result = await self.db_session.execute(agent_runs_query)
        agent_runs = result.scalar() or 0

        # 3. Total tokens and cost in time window
        totals_query = (
            select(
                func.coalesce(func.sum(StoredConversationMetadata.accumulated_cost), 0),
                func.coalesce(func.sum(StoredConversationMetadata.prompt_tokens), 0),
                func.coalesce(
                    func.sum(StoredConversationMetadata.completion_tokens), 0
                ),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.created_at >= cutoff)
        )
        result = await self.db_session.execute(totals_query)
        total_cost, total_prompt_tokens, total_completion_tokens = result.one()

        # 4. Daily usage breakdown (single query instead of N queries)
        from sqlalchemy import Date, cast

        daily_query = (
            select(
                cast(StoredConversationMetadata.created_at, Date).label('day'),
                func.count(StoredConversationMetadata.conversation_id),
                func.coalesce(
                    func.sum(
                        StoredConversationMetadata.prompt_tokens
                        + StoredConversationMetadata.completion_tokens
                    ),
                    0,
                ),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.created_at >= cutoff)
            .group_by(cast(StoredConversationMetadata.created_at, Date))
            .order_by(cast(StoredConversationMetadata.created_at, Date).asc())
        )
        result = await self.db_session.execute(daily_query)
        daily_rows = result.all()

        # Build a map of date -> (conv_count, token_count)
        daily_map = {row[0]: (row[1], row[2]) for row in daily_rows}

        # Generate entries for all days in range (including days with no data)
        daily_usage = []
        for i in range(days - 1, -1, -1):
            day_start = (now - timedelta(days=i)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            day_date = day_start.date() if hasattr(day_start, 'date') else day_start
            conv_count, token_count = daily_map.get(day_date, (0, 0))
            daily_usage.append(
                DailyUsageData(
                    date=day_start.strftime('%Y-%m-%d'),
                    tokens=int(token_count or 0),
                    conversations=int(conv_count or 0),
                )
            )

        # 5. Team usage (by user)
        team_query = (
            select(
                StoredConversationMetadataSaas.user_id,
                User.email,
                User.git_user_name,
                func.count(StoredConversationMetadata.conversation_id).label(
                    'conv_count'
                ),
                func.coalesce(
                    func.sum(
                        StoredConversationMetadata.prompt_tokens
                        + StoredConversationMetadata.completion_tokens
                    ),
                    0,
                ).label('token_count'),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .outerjoin(User, StoredConversationMetadataSaas.user_id == User.id)
            .where(*base_filter)
            .where(StoredConversationMetadata.created_at >= cutoff)
            .group_by(
                StoredConversationMetadataSaas.user_id,
                User.email,
                User.git_user_name,
            )
            .order_by(func.count(StoredConversationMetadata.conversation_id).desc())
        )
        result = await self.db_session.execute(team_query)
        team_rows = result.all()

        # Calculate percentages
        total_team_convs = sum(row.conv_count or 0 for row in team_rows)
        team_usage = []
        for row in team_rows:
            pct = (
                (row.conv_count / total_team_convs * 100) if total_team_convs > 0 else 0
            )
            team_usage.append(
                TeamUsageData(
                    user_id=str(row.user_id),
                    user_email=row.email,
                    user_name=row.git_user_name,
                    conversation_count=int(row.conv_count or 0),
                    total_tokens=int(row.token_count or 0),
                    percentage=round(pct, 1),
                )
            )

        # 6. Model usage (by model)
        model_query = (
            select(
                StoredConversationMetadata.llm_model,
                func.count(StoredConversationMetadata.conversation_id).label(
                    'conv_count'
                ),
                func.coalesce(
                    func.sum(
                        StoredConversationMetadata.prompt_tokens
                        + StoredConversationMetadata.completion_tokens
                    ),
                    0,
                ).label('token_count'),
                func.coalesce(
                    func.sum(StoredConversationMetadata.accumulated_cost), 0
                ).label('total_cost'),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.created_at >= cutoff)
            .group_by(StoredConversationMetadata.llm_model)
            .order_by(
                func.coalesce(
                    func.sum(StoredConversationMetadata.accumulated_cost), 0
                ).desc()
            )
        )
        result = await self.db_session.execute(model_query)
        model_rows = result.all()
        model_usage = []
        for row in model_rows:
            model_usage.append(
                ModelUsageData(
                    model_name=row.llm_model or 'Unknown',
                    conversation_count=int(row.conv_count or 0),
                    total_tokens=int(row.token_count or 0),
                    total_cost=float(row.total_cost or 0.0),
                )
            )

        agent_query = (
            select(
                StoredConversationMetadata.agent_kind,
                StoredConversationMetadata.llm_model,
                func.count(StoredConversationMetadata.conversation_id).label(
                    'conv_count'
                ),
                func.coalesce(
                    func.sum(StoredConversationMetadata.accumulated_cost), 0
                ).label('total_cost'),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.created_at >= cutoff)
            .group_by(
                StoredConversationMetadata.agent_kind,
                StoredConversationMetadata.llm_model,
            )
            .order_by(
                func.coalesce(
                    func.sum(StoredConversationMetadata.accumulated_cost), 0
                ).desc()
            )
        )
        result = await self.db_session.execute(agent_query)
        agent_rows = result.all()
        agent_counts: dict[str, int] = {}
        agent_costs: dict[str, float] = {}
        for row in agent_rows:
            label = _format_agent_label(row.agent_kind, row.llm_model)
            agent_counts[label] = agent_counts.get(label, 0) + int(row.conv_count or 0)
            agent_costs[label] = agent_costs.get(label, 0.0) + float(
                row.total_cost or 0.0
            )

        agent_usage = [
            AgentUsageData(
                agent_name=label,
                conversation_count=agent_counts[label],
                total_cost=agent_costs[label],
            )
            for label in agent_counts
        ]
        agent_usage.sort(key=lambda item: item.total_cost, reverse=True)

        return OrgUsageStats(
            active_users=int(active_users),
            agent_runs=int(agent_runs),
            total_tokens=int(total_prompt_tokens or 0)
            + int(total_completion_tokens or 0),
            estimated_spend=float(total_cost or 0),
            daily_usage=daily_usage,
            team_usage=team_usage,
            model_usage=model_usage,
            agent_usage=agent_usage,
        )

    async def get_user_usage_stats(
        self,
        org_id: UUID,
        limit: int = 500,
        offset: int = 0,
    ) -> OrgUserUsageStats:
        now = datetime.now(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        year_start = now.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )

        base_filter = [
            StoredConversationMetadata.conversation_version == 'V1',
            StoredConversationMetadataSaas.org_id == org_id,
        ]

        cost_events_subquery = (
            select(
                StoredConversationMetadataSaas.user_id.label('user_id'),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                StoredConversationCostEvent.occurred_at >= month_start,
                                StoredConversationCostEvent.cost_delta,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ).label('spend_mtd'),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                StoredConversationCostEvent.occurred_at >= year_start,
                                StoredConversationCostEvent.cost_delta,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ).label('spend_ytd'),
            )
            .select_from(StoredConversationCostEvent)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationCostEvent.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .join(
                StoredConversationMetadata,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .group_by(StoredConversationMetadataSaas.user_id)
        ).subquery()

        user_query = (
            select(
                StoredConversationMetadataSaas.user_id,
                User.email,
                User.git_user_name,
                User.first_login_at,
                User.last_login_at,
                func.count(StoredConversationMetadata.conversation_id).label(
                    'conversation_count'
                ),
                func.min(StoredConversationMetadata.created_at).label(
                    'first_conversation_at'
                ),
                func.max(StoredConversationMetadata.created_at).label(
                    'last_conversation_at'
                ),
                func.coalesce(
                    func.sum(StoredConversationMetadata.accumulated_cost), 0
                ).label('lifetime_spend'),
                func.coalesce(func.max(cost_events_subquery.c.spend_mtd), 0).label(
                    'spend_mtd'
                ),
                func.coalesce(func.max(cost_events_subquery.c.spend_ytd), 0).label(
                    'spend_ytd'
                ),
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .outerjoin(
                cost_events_subquery,
                cost_events_subquery.c.user_id
                == StoredConversationMetadataSaas.user_id,
            )
            .outerjoin(User, StoredConversationMetadataSaas.user_id == User.id)
            .where(*base_filter)
            .group_by(
                StoredConversationMetadataSaas.user_id,
                User.email,
                User.git_user_name,
                User.first_login_at,
                User.last_login_at,
            )
            .order_by(
                func.coalesce(
                    func.sum(StoredConversationMetadata.accumulated_cost), 0
                ).desc()
            )
        )

        fetch_limit = limit + 1
        result = await self.db_session.execute(
            user_query.limit(fetch_limit).offset(offset)
        )
        user_rows = result.all()
        has_more = False
        if len(user_rows) > limit:
            has_more = True
            user_rows = user_rows[:limit]

        settings_result = await self.db_session.execute(
            select(OrgBudgetSettings).where(OrgBudgetSettings.org_id == org_id)
        )
        budget_settings = settings_result.scalar_one_or_none()

        overrides_result = await self.db_session.execute(
            select(OrgUserBudgetOverride).where(OrgUserBudgetOverride.org_id == org_id)
        )
        override_map = {
            override.user_id: override for override in overrides_result.scalars().all()
        }

        pr_rows_result = await self.db_session.execute(
            select(
                StoredConversationMetadataSaas.user_id,
                StoredConversationMetadata.selected_repository,
                StoredConversationMetadata.git_provider,
                StoredConversationMetadata.pr_number,
            )
            .select_from(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(*base_filter)
            .where(StoredConversationMetadata.pr_number.is_not(None))
        )
        pr_rows = pr_rows_result.all()

        user_pr_keys: dict[UUID, set[tuple[str, str, int]]] = {}
        pr_keys: set[tuple[str, str, int]] = set()
        for row in pr_rows:
            if not row.selected_repository or not row.git_provider or not row.pr_number:
                continue
            for pr_number in row.pr_number:
                key = (row.git_provider, row.selected_repository, pr_number)
                pr_keys.add(key)
                user_pr_keys.setdefault(row.user_id, set()).add(key)

        pr_merge_map = await self._load_pr_merge_map(pr_keys)
        merged_counts: dict[UUID, int] = {}
        for user_id, keys in user_pr_keys.items():
            merged_counts[user_id] = sum(
                1 for key in keys if pr_merge_map.get(key) is True
            )

        items: list[OrgUserUsageRow] = []
        for row in user_rows:
            budget_monthly_limit = None
            budget_is_disabled = False
            if budget_settings and budget_settings.enabled:
                override = override_map.get(row.user_id)
                if override:
                    budget_is_disabled = override.is_disabled
                    if not override.is_disabled:
                        budget_monthly_limit = (
                            override.monthly_limit
                            if override.monthly_limit is not None
                            else budget_settings.default_user_monthly_limit
                        )
                else:
                    budget_monthly_limit = budget_settings.default_user_monthly_limit

            items.append(
                OrgUserUsageRow(
                    user_id=str(row.user_id),
                    user_email=row.email,
                    user_name=row.git_user_name,
                    conversation_count=int(row.conversation_count or 0),
                    first_conversation_at=row.first_conversation_at,
                    last_conversation_at=row.last_conversation_at,
                    first_login_at=row.first_login_at,
                    last_login_at=row.last_login_at,
                    spend_mtd=float(row.spend_mtd or 0.0),
                    spend_ytd=float(row.spend_ytd or 0.0),
                    spend_lifetime=float(row.lifetime_spend or 0.0),
                    budget_monthly_limit=budget_monthly_limit,
                    budget_is_disabled=budget_is_disabled,
                    prs_merged=(
                        merged_counts.get(row.user_id)
                        if row.user_id in user_pr_keys
                        else None
                    ),
                )
            )

        return OrgUserUsageStats(items=items, has_more=has_more)

    async def get_org_conversation(
        self,
        org_id: UUID,
        conversation_id: str,
    ) -> OrgConversationResponse | None:
        """Get a single conversation by ID.

        Args:
            org_id: The organization ID
            conversation_id: The conversation ID

        Returns:
            OrgConversationResponse if found, None otherwise
        """
        query = (
            select(StoredConversationMetadata, StoredConversationMetadataSaas, User)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .outerjoin(User, StoredConversationMetadataSaas.user_id == User.id)
            .where(StoredConversationMetadata.conversation_version == 'V1')
            .where(StoredConversationMetadataSaas.org_id == org_id)
            .where(StoredConversationMetadata.conversation_id == conversation_id)
        )

        result = await self.db_session.execute(query)
        row = result.one_or_none()

        if row is None:
            return None

        metadata, saas_metadata, user = row

        pr_keys: set[tuple[str, str, int]] = set()
        if (
            metadata.pr_number
            and metadata.selected_repository
            and metadata.git_provider
        ):
            for pr_number in metadata.pr_number:
                pr_keys.add(
                    (metadata.git_provider, metadata.selected_repository, pr_number)
                )
        pr_merge_map = await self._load_pr_merge_map(pr_keys)

        # Get sandbox info if available
        sandbox_info = None
        if metadata.sandbox_id and self.sandbox_service:
            try:
                sandbox_info = await self.sandbox_service.get_sandbox(
                    metadata.sandbox_id
                )
            except Exception as e:
                logger.warning(
                    'Failed to fetch sandbox info for conversation',
                    extra={'conversation_id': conversation_id, 'error': str(e)},
                )

        return self._build_conversation_response(
            metadata,
            saas_metadata,
            user,
            sandbox_info,
            pr_merged=self._resolve_pr_merged(metadata, pr_merge_map),
        )

    async def stop_conversation(
        self,
        org_id: UUID,
        conversation_id: str,
        user_id: str,
    ) -> dict | None:
        """Stop a running conversation and its runtime.

        Args:
            org_id: The organization ID
            conversation_id: The conversation ID
            user_id: The user performing the action

        Returns:
            Dict with success status and message
        """
        # First, verify the conversation exists and belongs to the org
        query = (
            select(StoredConversationMetadata)
            .join(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
            .where(StoredConversationMetadataSaas.org_id == org_id)
            .where(StoredConversationMetadata.conversation_id == conversation_id)
        )

        result = await self.db_session.execute(query)
        metadata = result.scalar_one_or_none()

        if metadata is None:
            return None

        # Check if there's a sandbox to stop
        if not metadata.sandbox_id:
            return {
                'success': True,
                'message': 'Conversation has no running sandbox',
                'conversation_id': conversation_id,
            }

        # Try to stop via sandbox service
        if self.sandbox_service:
            try:
                sandbox_info = await self.sandbox_service.get_sandbox(
                    metadata.sandbox_id
                )
                if sandbox_info is None:
                    return {
                        'success': True,
                        'message': 'Sandbox already stopped or not found',
                        'conversation_id': conversation_id,
                    }

                conversation_count = await self._count_conversations_by_sandbox_id(
                    metadata.sandbox_id
                )
                if conversation_count > 1:
                    return {
                        'success': False,
                        'error': 'Sandbox is shared by multiple conversations',
                        'error_code': 'sandbox_shared',
                        'conversation_id': conversation_id,
                        'sandbox_id': metadata.sandbox_id,
                    }

                # Update execution status to indicate stopping
                previous_status = metadata.execution_status
                metadata.execution_status = 'deleting'
                await self.db_session.commit()

                # Actually terminate the sandbox
                try:
                    deleted = await self.sandbox_service.delete_sandbox(
                        metadata.sandbox_id
                    )
                    if not deleted:
                        metadata.execution_status = previous_status
                        await self.db_session.commit()
                        return {
                            'success': False,
                            'error': 'Failed to stop sandbox',
                            'error_code': 'sandbox_stop_failed',
                            'conversation_id': conversation_id,
                            'sandbox_id': metadata.sandbox_id,
                        }
                except Exception:
                    # Rollback the status change so the row isn't left as 'deleting'
                    metadata.execution_status = previous_status
                    await self.db_session.commit()
                    raise

                logger.info(
                    'Stopping sandbox for org conversation',
                    extra={
                        'conversation_id': conversation_id,
                        'sandbox_id': metadata.sandbox_id,
                        'user_id': user_id,
                    },
                )

                return {
                    'success': True,
                    'message': 'Stop request sent to sandbox',
                    'conversation_id': conversation_id,
                    'sandbox_id': metadata.sandbox_id,
                }
            except Exception as e:
                logger.exception(
                    'Failed to stop sandbox',
                    extra={
                        'conversation_id': conversation_id,
                        'sandbox_id': metadata.sandbox_id,
                        'error': str(e),
                    },
                )
                return {
                    'success': False,
                    'error': 'Failed to stop sandbox',
                    'error_code': 'sandbox_stop_failed',
                    'conversation_id': conversation_id,
                }
        else:
            return {
                'success': False,
                'error': 'Sandbox service not available',
                'error_code': 'sandbox_unavailable',
                'conversation_id': conversation_id,
            }


class OrgConversationServiceInjector(Injector[OrgConversationService]):
    """Injector that composes db_session and sandbox_service for OrgConversationService."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[OrgConversationService, None]:
        # Local imports to avoid circular dependencies
        from openhands.app_server.config import get_db_session, get_sandbox_service

        async with get_db_session(state, request) as db_session:
            service = OrgConversationService(db_session=db_session)

            # Try to inject sandbox service if available
            try:
                async with get_sandbox_service(state, request) as sandbox_service:
                    service.set_sandbox_service(sandbox_service)
                    yield service
                    return
            except AssertionError as e:
                # Sandbox service not configured - log at warning level since
                # this is a SaaS-specific feature that requires it
                logger.warning(
                    'Sandbox service not available for OrgConversationService; '
                    'live sandbox status will not be available',
                    extra={'error': str(e)},
                )

            yield service
