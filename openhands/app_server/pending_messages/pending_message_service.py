"""Service for managing pending messages in SQL database."""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

from fastapi import Request
from pydantic import TypeAdapter
from sqlalchemy import JSON, String, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from openhands.agent_server.models import ImageContent, TextContent, utc_now
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.sql_app_conversation_lock import (
    lock_app_conversation,
)
from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (
    StoredAppConversationStartTask,
)
from openhands.app_server.pending_messages.pending_message_models import (
    PendingMessage,
    PendingMessageResponse,
)
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.utils.sql_utils import Base, UtcDateTime
from openhands.sdk.utils.models import DiscriminatedUnionMixin

# Type adapter for deserializing content from JSON
_content_type_adapter = TypeAdapter(list[TextContent | ImageContent])
_MAX_PENDING_MESSAGES = 10


def _normalize_conversation_id(conversation_id: str) -> str:
    if conversation_id.startswith('task-'):
        try:
            return f'task-{UUID(conversation_id.removeprefix("task-")).hex}'
        except ValueError:
            return conversation_id
    try:
        return str(UUID(conversation_id))
    except ValueError:
        return conversation_id


def _conversation_id_aliases(conversation_id: str) -> set[str]:
    if conversation_id.startswith('task-'):
        try:
            task_id = UUID(conversation_id.removeprefix('task-'))
        except ValueError:
            return {conversation_id}
        return {
            f'task-{task_id.hex}',
            f'task-{task_id}',
            f'task-{task_id.hex.upper()}',
            f'task-{str(task_id).upper()}',
        }
    try:
        canonical_id = UUID(conversation_id)
    except ValueError:
        return {conversation_id}
    return {
        str(canonical_id),
        canonical_id.hex,
        str(canonical_id).upper(),
        canonical_id.hex.upper(),
    }


class PendingMessageLimitExceeded(Exception):
    pass


class PendingMessageUnavailable(Exception):
    pass


class StoredPendingMessage(Base):
    """SQLAlchemy model for pending messages."""

    __tablename__ = 'pending_messages'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default='user')
    content: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), index=True
    )


class PendingMessageService(ABC):
    """Abstract service for managing pending messages."""

    @abstractmethod
    async def add_message(
        self,
        conversation_id: str,
        content: list[TextContent | ImageContent],
        role: str = 'user',
        queue_if_ready: bool = False,
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
    async def delete_messages(self, message_ids: list[str]) -> int:
        """Delete pending messages by ID."""

    @abstractmethod
    async def update_conversation_id(
        self, old_conversation_id: str, new_conversation_id: str
    ) -> int:
        """Update conversation_id when task-id transitions to real conversation-id.

        Returns the number of messages updated.
        """

    @abstractmethod
    async def begin_message_delivery_cutover(
        self, task_id: UUID, conversation_id: UUID
    ) -> list[PendingMessage]:
        """Lock the task and return its pending messages."""

    @abstractmethod
    async def finish_message_delivery_cutover(
        self,
        task_id: UUID,
        processed_message_ids: list[str],
    ) -> tuple[int, bool]:
        """Delete the processed batch and publish READY when the queue is empty."""


@dataclass
class SQLPendingMessageService(PendingMessageService):
    """SQL implementation of PendingMessageService."""

    db_session: AsyncSession

    async def _get_start_task(
        self,
        conversation_id: str,
        *,
        for_update: bool,
        canonical_id: UUID | None = None,
    ) -> StoredAppConversationStartTask | None:
        if canonical_id is not None:
            stmt = (
                select(StoredAppConversationStartTask)
                .where(
                    StoredAppConversationStartTask.app_conversation_id == canonical_id
                )
                .order_by(
                    StoredAppConversationStartTask.created_at.desc(),
                    StoredAppConversationStartTask.id.desc(),
                )
                .limit(1)
            )
        elif conversation_id.startswith('task-'):
            try:
                task_id = UUID(conversation_id.removeprefix('task-'))
            except ValueError:
                return None
            stmt = select(StoredAppConversationStartTask).where(
                StoredAppConversationStartTask.id == task_id
            )
        else:
            try:
                app_conversation_id = UUID(conversation_id)
            except ValueError:
                return None
            stmt = (
                select(StoredAppConversationStartTask)
                .where(
                    StoredAppConversationStartTask.app_conversation_id
                    == app_conversation_id
                )
                .order_by(
                    StoredAppConversationStartTask.created_at.desc(),
                    StoredAppConversationStartTask.id.desc(),
                )
                .limit(1)
            )
        if for_update:
            stmt = stmt.with_for_update()
        result = await self.db_session.execute(
            stmt.execution_options(populate_existing=True)
        )
        return result.scalars().first()

    async def _lock_canonical_conversation(self, conversation_id: UUID) -> None:
        await lock_app_conversation(self.db_session, conversation_id)

    async def _lock_legacy_conversation(self, conversation_id: str) -> None:
        bind = self.db_session.get_bind()
        if bind.dialect.name == 'postgresql':
            await self.db_session.execute(
                select(
                    func.pg_advisory_xact_lock(
                        func.hashtext(f'pending-message:{conversation_id}')
                    )
                )
            )

    async def _rollback(self) -> None:
        rollback_task = asyncio.create_task(self.db_session.rollback())
        while not rollback_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(rollback_task)
        with contextlib.suppress(Exception):
            await rollback_task

    async def _conversation_aliases(
        self,
        conversation_id: UUID,
        extra_conversation_id: str | None = None,
    ) -> set[str]:
        task_ids_stmt = select(StoredAppConversationStartTask.id).where(
            StoredAppConversationStartTask.app_conversation_id == conversation_id
        )
        task_ids = (await self.db_session.execute(task_ids_stmt)).scalars().all()
        aliases = _conversation_id_aliases(str(conversation_id))
        for task_id in task_ids:
            aliases.update(_conversation_id_aliases(f'task-{task_id.hex}'))
        if extra_conversation_id is not None:
            aliases.update(_conversation_id_aliases(extra_conversation_id))
        return aliases

    async def add_message(
        self,
        conversation_id: str,
        content: list[TextContent | ImageContent],
        role: str = 'user',
        queue_if_ready: bool = False,
    ) -> PendingMessageResponse:
        """Queue a message for delivery when conversation becomes ready."""
        conversation_aliases = _conversation_id_aliases(conversation_id)
        conversation_id = _normalize_conversation_id(conversation_id)
        try:
            task_hint = await self._get_start_task(
                conversation_id,
                for_update=False,
            )
            canonical_id = (
                task_hint.app_conversation_id if task_hint is not None else None
            )
            if canonical_id is None and not conversation_id.startswith('task-'):
                try:
                    canonical_id = UUID(conversation_id)
                except ValueError:
                    pass
            if canonical_id is not None:
                await self._lock_canonical_conversation(canonical_id)
            task = await self._get_start_task(
                conversation_id,
                for_update=True,
                canonical_id=canonical_id,
            )
            if (
                canonical_id is None
                and task is not None
                and task.app_conversation_id is not None
            ):
                await self.db_session.rollback()
                return await self.add_message(
                    conversation_id,
                    content,
                    role,
                    queue_if_ready,
                )
            if task is None and canonical_id is None:
                await self._lock_legacy_conversation(conversation_id)
            elif task is None:
                raise PendingMessageUnavailable
            elif task.status == AppConversationStartTaskStatus.ERROR:
                raise PendingMessageUnavailable
            elif (
                task.status == AppConversationStartTaskStatus.READY
                and task.app_conversation_id is not None
                and not queue_if_ready
            ):
                await self.db_session.commit()
                return PendingMessageResponse(
                    id=str(uuid4()),
                    queued=False,
                    position=0,
                    conversation_id=str(task.app_conversation_id),
                )

            conversation_ids = (
                await self._conversation_aliases(canonical_id, conversation_id)
                if canonical_id is not None
                else conversation_aliases
            )
            conversation_ids.update(conversation_aliases)
            count_stmt = select(func.count()).where(
                StoredPendingMessage.conversation_id.in_(conversation_ids)
            )
            position = (await self.db_session.execute(count_stmt)).scalar() or 0
            if position >= _MAX_PENDING_MESSAGES:
                raise PendingMessageLimitExceeded

            latest_created_at_stmt = (
                select(StoredPendingMessage.created_at)
                .where(StoredPendingMessage.conversation_id.in_(conversation_ids))
                .order_by(StoredPendingMessage.created_at.desc())
                .limit(1)
            )
            latest_created_at = (
                await self.db_session.execute(latest_created_at_stmt)
            ).scalar_one_or_none()
            created_at = utc_now()
            if latest_created_at is not None and latest_created_at.tzinfo is None:
                latest_created_at = latest_created_at.replace(tzinfo=UTC)
            if latest_created_at is not None and created_at <= latest_created_at:
                created_at = latest_created_at + timedelta(microseconds=1)
            pending_message = PendingMessage(
                conversation_id=conversation_id,
                role=role,
                content=content,
                created_at=created_at,
            )
            self.db_session.add(
                StoredPendingMessage(
                    id=pending_message.id,
                    conversation_id=conversation_id,
                    role=role,
                    content=[item.model_dump() for item in content],
                    created_at=pending_message.created_at,
                )
            )
            await self.db_session.commit()
            return PendingMessageResponse(
                id=pending_message.id,
                queued=True,
                position=position + 1,
            )
        except BaseException:
            await self._rollback()
            raise

    async def get_pending_messages(self, conversation_id: str) -> list[PendingMessage]:
        """Get all pending messages for a conversation, ordered by created_at."""
        conversation_ids = _conversation_id_aliases(conversation_id)
        stmt = (
            select(StoredPendingMessage)
            .where(StoredPendingMessage.conversation_id.in_(conversation_ids))
            .order_by(
                StoredPendingMessage.created_at.asc(),
                StoredPendingMessage.id.asc(),
            )
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
        conversation_ids = _conversation_id_aliases(conversation_id)
        count_stmt = select(func.count()).where(
            StoredPendingMessage.conversation_id.in_(conversation_ids)
        )
        result = await self.db_session.execute(count_stmt)
        return result.scalar() or 0

    async def delete_messages_for_conversation(self, conversation_id: str) -> int:
        """Delete all pending messages for a conversation, returning count deleted."""
        conversation_ids = _conversation_id_aliases(conversation_id)
        stmt = select(StoredPendingMessage).where(
            StoredPendingMessage.conversation_id.in_(conversation_ids)
        )
        result = await self.db_session.execute(stmt)
        stored_messages = result.scalars().all()

        count = len(stored_messages)
        for msg in stored_messages:
            await self.db_session.delete(msg)

        if count > 0:
            await self.db_session.commit()

        return count

    async def delete_messages(self, message_ids: list[str]) -> int:
        if not message_ids:
            return 0
        stmt = select(StoredPendingMessage).where(
            StoredPendingMessage.id.in_(message_ids)
        )
        stored_messages = (await self.db_session.execute(stmt)).scalars().all()
        for message in stored_messages:
            await self.db_session.delete(message)
        if stored_messages:
            await self.db_session.commit()
        return len(stored_messages)

    async def update_conversation_id(
        self, old_conversation_id: str, new_conversation_id: str
    ) -> int:
        """Update conversation_id when task-id transitions to real conversation-id."""
        old_conversation_ids = _conversation_id_aliases(old_conversation_id)
        new_conversation_id = _normalize_conversation_id(new_conversation_id)
        stmt = select(StoredPendingMessage).where(
            StoredPendingMessage.conversation_id.in_(old_conversation_ids)
        )
        result = await self.db_session.execute(stmt)
        stored_messages = result.scalars().all()

        count = len(stored_messages)
        for msg in stored_messages:
            msg.conversation_id = new_conversation_id

        if count > 0:
            await self.db_session.commit()

        return count

    async def begin_message_delivery_cutover(
        self, task_id: UUID, conversation_id: UUID
    ) -> list[PendingMessage]:
        try:
            await self._lock_canonical_conversation(conversation_id)
            task_stmt = (
                select(StoredAppConversationStartTask)
                .where(
                    or_(
                        StoredAppConversationStartTask.id == task_id,
                        StoredAppConversationStartTask.app_conversation_id
                        == conversation_id,
                    )
                )
                .order_by(
                    StoredAppConversationStartTask.created_at.desc(),
                    StoredAppConversationStartTask.id.desc(),
                )
                .limit(1)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            task = (await self.db_session.execute(task_stmt)).scalar_one_or_none()
            if task is None or task.id != task_id:
                raise PendingMessageUnavailable
            if task.status != AppConversationStartTaskStatus.STARTING_CONVERSATION:
                raise PendingMessageUnavailable
            if task.app_conversation_id not in (None, conversation_id):
                raise PendingMessageUnavailable

            canonical_id = str(conversation_id)
            task.app_conversation_id = conversation_id
            task_aliases = await self._conversation_aliases(
                conversation_id,
                f'task-{task_id.hex}',
            )
            task_aliases.add(f'task-{task_id}')
            task_aliases.discard(canonical_id)
            await self.db_session.execute(
                update(StoredPendingMessage)
                .where(StoredPendingMessage.conversation_id.in_(task_aliases))
                .values(conversation_id=canonical_id)
            )
            await self.db_session.flush()

            messages_stmt = (
                select(StoredPendingMessage)
                .where(StoredPendingMessage.conversation_id == canonical_id)
                .order_by(
                    StoredPendingMessage.created_at.asc(),
                    StoredPendingMessage.id.asc(),
                )
            )
            stored_messages = (
                (await self.db_session.execute(messages_stmt)).scalars().all()
            )
            messages = [
                PendingMessage(
                    id=message.id,
                    conversation_id=message.conversation_id,
                    role=message.role,
                    content=_content_type_adapter.validate_python(message.content),
                    created_at=message.created_at,
                )
                for message in stored_messages
            ]
            await self.db_session.commit()
            return messages
        except BaseException:
            await self._rollback()
            raise

    async def finish_message_delivery_cutover(
        self,
        task_id: UUID,
        processed_message_ids: list[str],
    ) -> tuple[int, bool]:
        try:
            task_stmt = select(StoredAppConversationStartTask).where(
                StoredAppConversationStartTask.id == task_id
            )
            task = (await self.db_session.execute(task_stmt)).scalar_one_or_none()
            if task is None:
                raise PendingMessageUnavailable
            if task.app_conversation_id is None:
                raise PendingMessageUnavailable
            await self._lock_canonical_conversation(task.app_conversation_id)
            current_task = await self._get_start_task(
                str(task.app_conversation_id),
                for_update=True,
                canonical_id=task.app_conversation_id,
            )
            if current_task is None or current_task.id != task_id:
                raise PendingMessageUnavailable
            pending_messages = list(
                (
                    await self.db_session.execute(
                        select(StoredPendingMessage)
                        .where(
                            StoredPendingMessage.conversation_id
                            == str(task.app_conversation_id)
                        )
                        .order_by(
                            StoredPendingMessage.created_at.asc(),
                            StoredPendingMessage.id.asc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            processed_messages = pending_messages[: len(processed_message_ids)]
            if [message.id for message in processed_messages] != processed_message_ids:
                raise PendingMessageUnavailable
            for message in processed_messages:
                await self.db_session.delete(message)
            await self.db_session.flush()
            remaining_count = (
                await self.db_session.execute(
                    select(func.count()).where(
                        StoredPendingMessage.conversation_id
                        == str(task.app_conversation_id)
                    )
                )
            ).scalar_one()
            ready = remaining_count == 0
            if ready:
                task.status = AppConversationStartTaskStatus.READY
                task.updated_at = utc_now()
            await self.db_session.commit()
            return len(processed_messages), ready
        except BaseException:
            await self._rollback()
            raise


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
