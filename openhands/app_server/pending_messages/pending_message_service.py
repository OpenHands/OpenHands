"""Service for managing pending messages in SQL database."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import Request
from pydantic import TypeAdapter
from sqlalchemy import JSON, Column, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.agent_server.models import ImageContent, TextContent
from openhands.app_server.pending_messages.pending_message_models import (
    PendingMessage,
    PendingMessageResponse,
)
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.utils.sql_utils import Base, UtcDateTime
from openhands.sdk.utils.models import DiscriminatedUnionMixin

# Type adapter for deserializing content from JSON
_content_type_adapter = TypeAdapter(list[TextContent | ImageContent])


class StoredPendingMessage(Base):  # type: ignore
    """SQLAlchemy model for pending messages."""

    __tablename__ = 'pending_messages'
    id = Column(String, primary_key=True)
    conversation_id = Column(String, nullable=False, index=True)
    role = Column(String(20), nullable=False, default='user')
    content = Column(JSON, nullable=False)
    created_at = Column(UtcDateTime, server_default=func.now(), index=True)


class PendingMessageService(ABC):
    """Abstract service for managing pending messages."""

    @abstractmethod
    async def add_message(
        self,
        conversation_id: str,
        content: list[TextContent | ImageContent],
        role: str = 'user',
    ) -> PendingMessageResponse:
        """Queue a message for delivery when conversation becomes ready."""

    @abstractmethod
    async def get_pending_messages(self, conversation_id: str) -> list[PendingMessage]:
        """Get all pending messages for a conversation, ordered by created_at."""

    @abstractmethod
    async def count_pending_messages(self, conversation_id: str) -> int:
        """Count pending messages for a conversation."""

    @abstractmethod
    async def delete_messages_for_conversation(self, conversation_id: str) -> int:
        """Delete all pending messages for a conversation, returning count deleted."""

    @abstractmethod
    async def update_conversation_id(
        self, old_conversation_id: str, new_conversation_id: str
    ) -> int:
        """Update conversation_id when task-id transitions to real conversation-id.

        Returns the number of messages updated.
        """


@dataclass
class SQLPendingMessageService(PendingMessageService):
    """SQL implementation of PendingMessageService."""

    db_session: AsyncSession

    async def add_message(
        self,
        conversation_id: str,
        content: list[TextContent | ImageContent],
        role: str = 'user',
    ) -> PendingMessageResponse:
        """Queue a message for delivery when conversation becomes ready."""
        # Create the pending message
        pending_message = PendingMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

        # Count existing pending messages for position
        count_stmt = select(func.count()).where(
            StoredPendingMessage.conversation_id == conversation_id
        )
        result = await self.db_session.execute(count_stmt)
        position = result.scalar() or 0

        # Serialize content to JSON-compatible format for storage
        content_json = [item.model_dump() for item in content]

        # Store in database
        stored_message = StoredPendingMessage(
            id=str(pending_message.id),
            conversation_id=conversation_id,
            role=role,
            content=content_json,
            created_at=pending_message.created_at,
        )
        self.db_session.add(stored_message)
        await self.db_session.commit()

        return PendingMessageResponse(
            id=pending_message.id,
            queued=True,
            position=position + 1,
        )

    async def get_pending_messages(self, conversation_id: str) -> list[PendingMessage]:
        """Get all pending messages for a conversation, ordered by created_at."""
        stmt = (
            select(StoredPendingMessage)
            .where(StoredPendingMessage.conversation_id == conversation_id)
            .order_by(StoredPendingMessage.created_at.asc())
        )
        result = await self.db_session.execute(stmt)
        stored_messages = result.scalars().all()

        return [
            PendingMessage(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,
                content=_content_type_adapter.validate_python(msg.content),
                created_at=msg.created_at,
            )
            for msg in stored_messages
        ]

    async def count_pending_messages(self, conversation_id: str) -> int:
        """Count pending messages for a conversation."""
        count_stmt = select(func.count()).where(
            StoredPendingMessage.conversation_id == conversation_id
        )
        result = await self.db_session.execute(count_stmt)
        return result.scalar() or 0

    async def delete_messages_for_conversation(self, conversation_id: str) -> int:
        """Delete all pending messages for a conversation, returning count deleted."""
        stmt = select(StoredPendingMessage).where(
            StoredPendingMessage.conversation_id == conversation_id
        )
        result = await self.db_session.execute(stmt)
        stored_messages = result.scalars().all()

        count = len(stored_messages)
        for msg in stored_messages:
            await self.db_session.delete(msg)

        if count > 0:
            await self.db_session.commit()

        return count

    async def update_conversation_id(
        self, old_conversation_id: str, new_conversation_id: str
    ) -> int:
        """Update conversation_id when task-id transitions to real conversation-id."""
        stmt = select(StoredPendingMessage).where(
            StoredPendingMessage.conversation_id == old_conversation_id
        )
        result = await self.db_session.execute(stmt)
        stored_messages = result.scalars().all()

        count = len(stored_messages)
        for msg in stored_messages:
            msg.conversation_id = new_conversation_id

        if count > 0:
            await self.db_session.commit()

        return count


class PendingMessageServiceInjector(
    DiscriminatedUnionMixin, Injector[PendingMessageService], ABC
):
    """Abstract injector for PendingMessageService."""

    pass


class SQLPendingMessageServiceInjector(PendingMessageServiceInjector):
    """SQL-based injector for PendingMessageService."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[PendingMessageService, None]:
        from openhands.app_server.config import get_db_session

        async with get_db_session(state) as db_session:
            yield SQLPendingMessageService(db_session=db_session)
