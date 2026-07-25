import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationInfoPage,
    AppConversationSortOrder,
    has_managed_codex_credential,
)
from openhands.app_server.services.injector import Injector
from openhands.sdk import ConversationStats
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.sdk.utils.models import DiscriminatedUnionMixin


@dataclass(frozen=True)
class ManagedCredentialConversationRef:
    conversation_id: UUID
    created_by_user_id: str | None
    organization_id: UUID | None
    owner_resolved: bool = True


class AppConversationInfoService(ABC):
    """Service for accessing info on conversations without their current status."""

    @abstractmethod
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
        """Search for sandboxed conversations."""

    @abstractmethod
    async def count_app_conversation_info(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sandbox_id__eq: str | None = None,
    ) -> int:
        """Count sandboxed conversations."""

    @abstractmethod
    async def get_app_conversation_info(
        self, conversation_id: UUID
    ) -> AppConversationInfo | None:
        """Get a single conversation info, returning None if missing."""

    @abstractmethod
    async def is_app_conversation_id_available(self, conversation_id: UUID) -> bool:
        """Check whether a conversation ID is globally available."""

    async def try_reserve_app_conversation_id(
        self,
        conversation_id: UUID,
        created_by_user_id: str | None = None,
    ) -> bool:
        return await self.is_app_conversation_id_available(conversation_id)

    async def release_app_conversation_id_reservation(
        self, conversation_id: UUID
    ) -> None:
        return None

    async def renew_app_conversation_id_reservation(
        self, conversation_id: UUID
    ) -> bool:
        return True

    async def get_managed_credential_conversations_for_sandbox(
        self, sandbox_id: str
    ) -> list[ManagedCredentialConversationRef]:
        refs: list[ManagedCredentialConversationRef] = []
        page_id = None
        while True:
            page = await self.search_app_conversation_info(
                sandbox_id__eq=sandbox_id,
                page_id=page_id,
                limit=100,
                include_sub_conversations=True,
            )
            refs.extend(
                ManagedCredentialConversationRef(
                    conversation_id=info.id,
                    created_by_user_id=info.created_by_user_id,
                    organization_id=None,
                )
                for info in page.items
                if has_managed_codex_credential(info.tags)
            )
            if page.next_page_id is None:
                return refs
            page_id = page.next_page_id

    async def batch_get_app_conversation_info(
        self, conversation_ids: list[UUID]
    ) -> list[AppConversationInfo | None]:
        """Get a batch of conversation info, return None for any missing."""
        return await asyncio.gather(
            *[
                self.get_app_conversation_info(conversation_id)
                for conversation_id in conversation_ids
            ]
        )

    @abstractmethod
    async def delete_app_conversation_info(self, conversation_id: UUID) -> bool:
        """Delete a conversation info from the database.

        Args:
            conversation_id: The ID of the conversation to delete.

        Returns True if the conversation was deleted successfully, False otherwise.
        """

    @abstractmethod
    async def get_sub_conversation_ids(
        self, parent_conversation_id: UUID
    ) -> list[UUID]:
        """Get all sub-conversation IDs for a given parent conversation.

        Args:
            parent_conversation_id: The ID of the parent conversation

        Returns:
            List of sub-conversation IDs
        """

    @abstractmethod
    async def count_conversations_by_sandbox_id(self, sandbox_id: str) -> int:
        """Count V1 conversations that reference the given sandbox.

        Used to decide whether a sandbox can be safely deleted when a
        conversation is removed (only delete if count is 0).
        """

    # Mutators

    @abstractmethod
    async def save_app_conversation_info(
        self, info: AppConversationInfo
    ) -> AppConversationInfo:
        """Store the sandboxed conversation info object given.

        Return the stored info
        """

    @abstractmethod
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

    @abstractmethod
    async def update_conversation_statistics(
        self,
        conversation_id: UUID,
        stats: ConversationStats,
        event_timestamp: datetime | None = None,
    ) -> None:
        """Update persisted statistics from a ConversationStats snapshot."""

    @abstractmethod
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


class AppConversationInfoServiceInjector(
    DiscriminatedUnionMixin, Injector[AppConversationInfoService], ABC
):
    pass
