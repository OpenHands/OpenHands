import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTask,
    AppConversationStartTaskPage,
    AppConversationStartTaskSortOrder,
)
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class AppConversationStartTaskService(ABC):
    """Service for accessing start tasks for conversations."""

    @abstractmethod
    async def search_app_conversation_start_tasks(
        self,
        conversation_id__eq: UUID | None = None,
        created_at__gte: datetime | None = None,
        sort_order: AppConversationStartTaskSortOrder = AppConversationStartTaskSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 100,
    ) -> AppConversationStartTaskPage:
        """Search for conversation start tasks."""

    @abstractmethod
    async def count_app_conversation_start_tasks(
        self,
        conversation_id__eq: UUID | None = None,
        created_at__gte: datetime | None = None,
    ) -> int:
        """Count conversation start tasks."""

    @abstractmethod
    async def get_app_conversation_start_task(
        self, task_id: UUID
    ) -> AppConversationStartTask | None:
        """Get a single start task, returning None if missing."""

    async def batch_get_app_conversation_start_tasks(
        self, task_ids: list[UUID]
    ) -> list[AppConversationStartTask | None]:
        """Get a batch of start tasks, return None for any missing."""
        return await asyncio.gather(
            *[self.get_app_conversation_start_task(task_id) for task_id in task_ids]
        )

    # Mutators

    @abstractmethod
    async def save_app_conversation_start_task(
        self, info: AppConversationStartTask
    ) -> AppConversationStartTask:
        """Store the start task object given.

        Return the stored task
        """

    @abstractmethod
    async def delete_app_conversation_start_tasks(self, conversation_id: UUID) -> bool:
        """Delete all start tasks associated with a conversation.

        Args:
            conversation_id: The ID of the conversation to delete tasks for.

        Returns True if any tasks were deleted successfully, False otherwise.
        """


class AppConversationStartTaskServiceInjector(
    DiscriminatedUnionMixin, Injector[AppConversationStartTaskService], ABC
):
    pass
