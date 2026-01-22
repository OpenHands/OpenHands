import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
    AppConversationPage,
    AppConversationSortOrder,
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationUpdateRequest,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace


class AppConversationService(ABC):
    """Service for managing conversations running in sandboxes."""

    @abstractmethod
    async def search_app_conversations(
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
    ) -> AppConversationPage:
        """Search for sandboxed conversations."""

    @abstractmethod
    async def count_app_conversations(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
    ) -> int:
        """Count sandboxed conversations."""

    @abstractmethod
    async def get_app_conversation(
        self, conversation_id: UUID
    ) -> AppConversation | None:
        """Get a single sandboxed conversation info. Return None if missing."""

    async def batch_get_app_conversations(
        self, conversation_ids: list[UUID]
    ) -> list[AppConversation | None]:
        """Get a batch of sandboxed conversations, returning None for any missing."""
        return await asyncio.gather(
            *[
                self.get_app_conversation(conversation_id)
                for conversation_id in conversation_ids
            ]
        )

    @abstractmethod
    async def start_app_conversation(
        self, request: AppConversationStartRequest
    ) -> AsyncGenerator[AppConversationStartTask, None]:
        """Start a conversation, optionally specifying a sandbox in which to start.

        If no sandbox is specified a default may be used or started. This is a convenience
        method - the same effect should be achievable by creating / getting a sandbox
        id, starting a conversation, attaching a callback, and then running the
        conversation.

        Yields an instance of AppConversationStartTask as updates occur, which can be used to determine
        the progress of the task.
        """
        # This is an abstract method - concrete implementations should provide real values
        from openhands.app_server.app_conversation.app_conversation_models import (
            AppConversationStartRequest,
        )

        dummy_request = AppConversationStartRequest()
        yield AppConversationStartTask(
            created_by_user_id='dummy',
            request=dummy_request,
        )

    @abstractmethod
    async def run_setup_scripts(
        self,
        task: AppConversationStartTask,
        sandbox: SandboxInfo,
        workspace: AsyncRemoteWorkspace,
        agent_server_url: str,
    ) -> AsyncGenerator[AppConversationStartTask, None]:
        """Run the setup scripts for the project and yield status updates"""
        yield task

    @abstractmethod
    async def update_app_conversation(
        self, conversation_id: UUID, request: AppConversationUpdateRequest
    ) -> AppConversation | None:
        """Update an app conversation and return it. Return None if the conversation
        did not exist."""

    @abstractmethod
    async def delete_app_conversation(self, conversation_id: UUID) -> bool:
        """Delete a V1 conversation and all its associated data.

        Args:
            conversation_id: The UUID of the conversation to delete.

        This method should:
        1. Delete the conversation from the database
        2. Call the agent server to delete the conversation
        3. Clean up any related data

        Returns True if the conversation was deleted successfully, False otherwise.
        """

    @abstractmethod
    async def export_conversation(self, conversation_id: UUID) -> bytes:
        """Download a conversation trajectory as a zip file.

        Args:
            conversation_id: The UUID of the conversation to download.

        This method should:
        1. Get all events for the conversation
        2. Create a temporary directory
        3. Save each event as a JSON file
        4. Save conversation metadata as meta.json
        5. Create and return a zip file containing all the data

        Returns the zip file as bytes.
        """


class AppConversationServiceInjector(
    DiscriminatedUnionMixin, Injector[AppConversationService], ABC
):
    pass
