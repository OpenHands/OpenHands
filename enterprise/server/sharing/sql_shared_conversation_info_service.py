"""SQL implementation of SharedConversationInfoService.

This implementation provides read-only access to shared conversations:
- Direct database access without user permission checks
- Filters only conversations marked as shared (currently public)
- Full async/await support using SQL async db_sessions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from server.sharing.shared_conversation_info_service import (
    SharedConversationInfoService,
    SharedConversationInfoServiceInjector,
)
from server.sharing.shared_conversation_models import (
    SharedConversation,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas

from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    StoredConversationMetadata,
)
from openhands.app_server.services.injector import InjectorState
from openhands.integrations.provider import ProviderType
from openhands.sdk.llm import MetricsSnapshot, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class SQLSharedConversationInfoService(SharedConversationInfoService):
    """SQL implementation of SharedConversationInfoService for shared conversations only."""

    db_session: AsyncSession

    async def get_shared_conversation_info(
        self, conversation_id: UUID
    ) -> SharedConversation | None:
        """Get a single public conversation info, returning None if missing or not shared."""
        query = self._public_select_with_saas_metadata().where(
            StoredConversationMetadata.conversation_id == str(conversation_id)
        )

        result = await self.db_session.execute(query)
        row = result.first()

        if row is None:
            return None

        stored, saas_metadata = row
        return self._to_shared_conversation(stored, saas_metadata=saas_metadata)

    def _public_select_with_saas_metadata(self):
        """Create a select query that returns public conversations with SAAS metadata.

        This joins with conversation_metadata_saas to retrieve the user_id needed
        for constructing the correct event storage path. Uses LEFT OUTER JOIN to
        support conversations that may not have SAAS metadata (e.g., in tests).
        """
        query = (
            select(StoredConversationMetadata, StoredConversationMetadataSaas)
            .outerjoin(
                StoredConversationMetadataSaas,
                StoredConversationMetadata.conversation_id
                == StoredConversationMetadataSaas.conversation_id,
            )
            .where(StoredConversationMetadata.conversation_version == 'V1')
            .where(StoredConversationMetadata.public == True)  # noqa: E712
        )
        return query

    def _to_shared_conversation(
        self,
        stored: StoredConversationMetadata,
        saas_metadata: StoredConversationMetadataSaas | None = None,
        sub_conversation_ids: list[UUID] | None = None,
    ) -> SharedConversation:
        """Convert StoredConversationMetadata to SharedConversation.

        Args:
            stored: The base conversation metadata from conversation_metadata table.
            saas_metadata: Optional SAAS metadata containing user_id and org_id.
            sub_conversation_ids: Optional list of sub-conversation IDs.
        """
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

        # Get user_id from SAAS metadata if available
        created_by_user_id = (
            str(saas_metadata.user_id)
            if saas_metadata and saas_metadata.user_id
            else None
        )

        return SharedConversation(
            id=UUID(stored.conversation_id),
            created_by_user_id=created_by_user_id,
            sandbox_id=stored.sandbox_id,
            selected_repository=stored.selected_repository,
            selected_branch=stored.selected_branch,
            git_provider=(
                ProviderType(stored.git_provider) if stored.git_provider else None
            ),
            title=stored.title,
            pr_number=stored.pr_number,
            llm_model=stored.llm_model,
            metrics=metrics,
            parent_conversation_id=(
                UUID(stored.parent_conversation_id)
                if stored.parent_conversation_id
                else None
            ),
            sub_conversation_ids=sub_conversation_ids or [],
            created_at=created_at,
            updated_at=updated_at,
        )

    def _fix_timezone(self, value: datetime) -> datetime:
        """Sqlite does not store timezones - and since we can't update the existing models
        we assume UTC if the timezone is missing."""
        if not value.tzinfo:
            value = value.replace(tzinfo=UTC)
        return value


class SQLSharedConversationInfoServiceInjector(SharedConversationInfoServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SharedConversationInfoService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import get_db_session

        async with get_db_session(state, request) as db_session:
            service = SQLSharedConversationInfoService(db_session=db_session)
            yield service
