"""Enterprise injector for PendingMessageService with SAAS filtering."""

from typing import AsyncGenerator
from uuid import UUID

from fastapi import Request
from sqlalchemy import select
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user import User

from openhands.agent_server.models import ImageContent, TextContent
from openhands.app_server.errors import AuthError
from openhands.app_server.pending_messages.pending_message_models import (
    PendingMessageResponse,
)
from openhands.app_server.pending_messages.pending_message_service import (
    PendingMessageService,
    PendingMessageServiceInjector,
    SQLPendingMessageService,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN
from openhands.app_server.user.user_context import UserContext


class SaasSQLPendingMessageService(SQLPendingMessageService):
    """Extended SQLPendingMessageService with user and organization-based filtering.

    This enterprise version ensures that:
    - Users can only queue messages for conversations they own
    - Organization isolation is enforced for multi-tenant deployments
    """

    def __init__(self, db_session, user_context: UserContext):
        super().__init__(db_session=db_session)
        self.user_context = user_context

    async def _get_current_user(self) -> User | None:
        """Get the current user using the existing db_session.

        Returns:
            User object or None if no user_id is available
        """
        user_id_str = await self.user_context.get_user_id()
        if not user_id_str:
            return None

        user_id_uuid = UUID(user_id_str)
        result = await self.db_session.execute(
            select(User).where(User.id == user_id_uuid)
        )
        return result.scalars().first()

    async def _validate_conversation_ownership(self, conversation_id: str) -> None:
        """Validate that the current user owns the conversation.

        This ensures multi-tenant isolation by checking:
        - The conversation belongs to the current user
        - The conversation belongs to the user's current organization

        Args:
            conversation_id: The conversation ID to validate (can be task-id or UUID)

        Raises:
            AuthError: If user doesn't own the conversation or authentication fails
        """
        # For internal operations (e.g., processing pending messages during startup)
        # we need a mode that bypasses filtering. The ADMIN context enables this.
        if self.user_context == ADMIN:
            return

        user_id_str = await self.user_context.get_user_id()
        if not user_id_str:
            raise AuthError('User authentication required')

        user_id_uuid = UUID(user_id_str)

        # Check conversation ownership via SAAS metadata
        query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == conversation_id
        )
        result = await self.db_session.execute(query)
        saas_metadata = result.scalar_one_or_none()

        # If no SAAS metadata exists, the conversation might be a new task-id
        # that hasn't been linked to a conversation yet. Allow access in this case
        # as the message will be validated when the conversation is created.
        if saas_metadata is None:
            return

        # Verify user ownership
        if saas_metadata.user_id != user_id_uuid:
            raise AuthError('You do not have access to this conversation')

        # Verify organization ownership if applicable
        user = await self._get_current_user()
        if user and user.current_org_id is not None:
            if saas_metadata.org_id != user.current_org_id:
                raise AuthError('Conversation belongs to a different organization')

    async def add_message(
        self,
        conversation_id: str,
        content: list[TextContent | ImageContent],
        role: str = 'user',
    ) -> PendingMessageResponse:
        """Queue a message with ownership validation.

        Args:
            conversation_id: The conversation ID to queue the message for
            content: Message content
            role: Message role (default: 'user')

        Returns:
            PendingMessageResponse with the queued message info

        Raises:
            AuthError: If user doesn't own the conversation
        """
        await self._validate_conversation_ownership(conversation_id)
        return await super().add_message(conversation_id, content, role)

    async def get_pending_messages(self, conversation_id: str):
        """Get pending messages with ownership validation.

        Args:
            conversation_id: The conversation ID to get messages for

        Returns:
            List of pending messages

        Raises:
            AuthError: If user doesn't own the conversation
        """
        await self._validate_conversation_ownership(conversation_id)
        return await super().get_pending_messages(conversation_id)

    async def count_pending_messages(self, conversation_id: str) -> int:
        """Count pending messages with ownership validation.

        Args:
            conversation_id: The conversation ID to count messages for

        Returns:
            Number of pending messages

        Raises:
            AuthError: If user doesn't own the conversation
        """
        await self._validate_conversation_ownership(conversation_id)
        return await super().count_pending_messages(conversation_id)


class SaasPendingMessageServiceInjector(PendingMessageServiceInjector):
    """Enterprise injector for PendingMessageService with SAAS filtering."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[PendingMessageService, None]:
        from openhands.app_server.config import (
            get_db_session,
            get_user_context,
        )

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            service = SaasSQLPendingMessageService(
                db_session=db_session, user_context=user_context
            )
            yield service
