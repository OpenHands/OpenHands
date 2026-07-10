# pyright: reportArgumentType=false
"""SQL implementation of EventCallbackService."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID, uuid4

from fastapi import Request
from sqlalchemy import Enum, Index, String, and_, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from openhands.agent_server.utils import utc_now
from openhands.app_server.event_callback.event_callback_models import (
    CreateEventCallbackRequest,
    EventCallback,
    EventCallbackPage,
    EventCallbackProcessor,
    EventCallbackStatus,
    EventKind,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)
from openhands.app_server.event_callback.event_callback_service import (
    EventCallbackService,
    EventCallbackServiceInjector,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.sql_utils import (
    Base,
    UtcDateTime,
    create_json_type_decorator,
    row2dict,
)
from openhands.sdk import Event

_logger = logging.getLogger(__name__)

# TODO: Add user level filtering to this class


class StoredEventCallback(Base):
    __tablename__ = 'event_callback'
    __table_args__ = (
        Index(
            'ix_event_callback_conversation_id_status_event_kind',
            'conversation_id',
            'status',
            'event_kind',
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID | None] = mapped_column(nullable=True)
    status: Mapped[EventCallbackStatus] = mapped_column(
        Enum(EventCallbackStatus), nullable=False, default=EventCallbackStatus.ACTIVE
    )
    processor: Mapped[EventCallbackProcessor] = mapped_column(
        create_json_type_decorator(EventCallbackProcessor)
    )
    event_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), index=True
    )


class StoredEventCallbackResult(Base):
    __tablename__ = 'event_callback_result'

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    status: Mapped[EventCallbackResultStatus | None] = mapped_column(
        Enum(EventCallbackResultStatus), nullable=True
    )
    event_callback_id: Mapped[UUID] = mapped_column(index=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    conversation_id: Mapped[UUID] = mapped_column(index=True)
    detail: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, server_default=func.now(), index=True
    )


@dataclass
class SQLEventCallbackService(EventCallbackService):
    """SQL implementation of EventCallbackService.

    The service does **not** hold a long-lived ``AsyncSession``. Each public
    method opens a short-lived session via ``self.async_session_maker`` for the
    duration of its own DB work. This means:

    * The pool connection is only checked out while the method runs an
      actual SQL statement — it is released as soon as the method returns,
      so callers never accidentally pin a connection across slow logic
      (e.g. webhook deliveries, ``asyncio.gather`` of callback processors).
    * Methods are independent: a failure in one (e.g. a flush error) cannot
      leave another with a session in an unknown transactional state.
    * The service itself is no longer an async-context-manager. There is
      no per-service ``__aenter__``/``__aexit__`` lifecycle to get wrong.
    """

    async_session_maker: async_sessionmaker

    async def create_event_callback(
        self, request: CreateEventCallbackRequest
    ) -> EventCallback:
        """Create a new event callback."""
        event_callback = EventCallback(
            conversation_id=request.conversation_id,
            processor=request.processor,
            event_kind=request.event_kind,
        )
        async with self.async_session_maker() as db_session:
            stored_callback = StoredEventCallback(**event_callback.model_dump())
            db_session.add(stored_callback)
            await db_session.commit()
            await db_session.refresh(stored_callback)
            return EventCallback.model_validate(row2dict(stored_callback))

    async def get_event_callback(self, id: UUID) -> EventCallback | None:
        """Get a single event callback, returning None if not found."""
        async with self.async_session_maker() as db_session:
            stmt = select(StoredEventCallback).where(StoredEventCallback.id == id)
            result = await db_session.execute(stmt)
            stored_callback = result.scalar_one_or_none()
            if stored_callback:
                return EventCallback.model_validate(row2dict(stored_callback))
            return None

    async def delete_event_callback(self, id: UUID) -> bool:
        """Delete an event callback, returning True if deleted, False if not found."""
        async with self.async_session_maker() as db_session:
            stmt = select(StoredEventCallback).where(StoredEventCallback.id == id)
            result = await db_session.execute(stmt)
            stored_callback = result.scalar_one_or_none()
            if stored_callback is None:
                return False
            await db_session.delete(stored_callback)
            await db_session.commit()
            return True

    async def search_event_callbacks(
        self,
        conversation_id__eq: UUID | None = None,
        event_kind__eq: EventKind | None = None,
        event_id__eq: UUID | None = None,
        page_id: str | None = None,
        limit: int = 100,
    ) -> EventCallbackPage:
        """Search for event callbacks, optionally filtered by parameters."""
        conditions = []
        if conversation_id__eq is not None:
            conditions.append(
                StoredEventCallback.conversation_id == conversation_id__eq
            )
        if event_kind__eq is not None:
            conditions.append(StoredEventCallback.event_kind == event_kind__eq)
        # Note: event_id__eq is not stored in the event_callbacks table; kept
        # in the signature for ABI compatibility with the abstract service.

        stmt = select(StoredEventCallback)
        if conditions:
            stmt = stmt.where(and_(*conditions))

        if page_id is not None:
            try:
                offset = int(page_id)
                stmt = stmt.offset(offset)
            except ValueError:
                offset = 0
        else:
            offset = 0
        stmt = stmt.limit(limit + 1).order_by(StoredEventCallback.created_at.desc())

        async with self.async_session_maker() as db_session:
            result = await db_session.execute(stmt)
            stored_callbacks = result.scalars().all()

        has_more = len(stored_callbacks) > limit
        if has_more:
            stored_callbacks = stored_callbacks[:limit]
        next_page_id = str(offset + limit) if has_more else None
        callbacks = [
            EventCallback.model_validate(row2dict(cb)) for cb in stored_callbacks
        ]
        return EventCallbackPage(items=callbacks, next_page_id=next_page_id)

    async def save_event_callback(self, event_callback: EventCallback) -> EventCallback:
        event_callback.updated_at = utc_now()
        async with self.async_session_maker() as db_session:
            stored_callback = StoredEventCallback(**event_callback.model_dump())
            await db_session.merge(stored_callback)
            await db_session.commit()
        return event_callback

    async def get_active_callbacks(
        self, conversation_id: UUID, event: Event
    ) -> list[EventCallback]:
        """Return the active callbacks registered for this conversation+event kind.

        Each returned ``EventCallback`` is detached from the session, so the
        caller can run the (potentially slow) callback processors without
        holding a pool connection. Use :meth:`persist_callback_results`
        afterwards to save any status changes the callbacks made to themselves
        plus the ``EventCallbackResult`` rows they produced.
        """
        query = (
            select(StoredEventCallback)
            .where(StoredEventCallback.status == EventCallbackStatus.ACTIVE)
            .where(StoredEventCallback.event_kind == event.kind)
            .where(StoredEventCallback.conversation_id == conversation_id)
        )
        async with self.async_session_maker() as db_session:
            result = await db_session.execute(query)
            stored_callbacks = result.scalars().all()
        return [EventCallback.model_validate(row2dict(cb)) for cb in stored_callbacks]

    async def persist_callback_results(
        self,
        callbacks: list[EventCallback],
        results: list[StoredEventCallbackResult | None],
    ) -> None:
        """Persist callback status changes and their results.

        Pairs up ``callbacks`` and ``results`` index-wise. ``results[i]`` may be
        ``None`` if the callback returned ``None``. Opens its own short-lived
        session so the pool connection is only held for the duration of the
        COMMIT, not the entire callback execution.
        """
        async with self.async_session_maker() as db_session:
            for callback, stored_result in zip(callbacks, results, strict=False):
                callback.updated_at = utc_now()
                stored_callback = StoredEventCallback(**callback.model_dump())
                await db_session.merge(stored_callback)
                if stored_result is not None:
                    db_session.add(stored_result)
            if any(r is not None for r in results):
                await db_session.commit()

    async def execute_callbacks(self, conversation_id: UUID, event: Event) -> None:
        """Run all active callbacks for the event and persist their results.

        Each step (``get_active_callbacks``, ``invoke_callback``,
        ``persist_callback_results``) opens and closes its own session, so the
        pool connection is never pinned while the (potentially slow) callback
        processors run. Failed callbacks are turned into ERROR result rows so a
        single failure doesn't abort the rest.
        """
        callbacks = await self.get_active_callbacks(conversation_id, event)
        if not callbacks:
            return
        outcomes = await asyncio.gather(
            *[invoke_callback(cb, conversation_id, event) for cb in callbacks],
            return_exceptions=True,
        )
        normalised: list[StoredEventCallbackResult | None] = []
        for callback, outcome in zip(callbacks, outcomes, strict=False):
            if isinstance(outcome, BaseException):
                _logger.exception(
                    f'Exception in callback {callback.id}', stack_info=True
                )
                normalised.append(
                    StoredEventCallbackResult(
                        status=EventCallbackResultStatus.ERROR,
                        event_callback_id=callback.id,
                        event_id=event.id,
                        conversation_id=conversation_id,
                        detail=str(outcome),
                    )
                )
            else:
                normalised.append(outcome)
        await self.persist_callback_results(callbacks, normalised)


async def invoke_callback(
    callback: EventCallback, conversation_id: UUID, event: Event
) -> StoredEventCallbackResult | None:
    """Run a single callback processor and convert its outcome into a stored
    result row. No DB session is required to call the processor — the row is
    only added to a session later by ``persist_callback_results``. This lets
    the caller release the pool connection while the (potentially slow)
    callbacks run, avoiding pool exhaustion when several webhooks fire in
    burst."""
    try:
        result = await callback.processor(conversation_id, callback, event)
    except Exception as exc:
        _logger.exception(f'Exception in callback {callback.id}', stack_info=True)
        return StoredEventCallbackResult(
            status=EventCallbackResultStatus.ERROR,
            event_callback_id=callback.id,
            event_id=event.id,
            conversation_id=conversation_id,
            detail=str(exc),
        )
    if result is None:
        return None
    return StoredEventCallbackResult(**result.model_dump())


class SQLEventCallbackServiceInjector(EventCallbackServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[EventCallbackService, None]:
        # No async-context-manager needed here: the service does not hold a
        # session, so there is nothing to release on __aexit__. We just hand
        # it the cached ``async_sessionmaker`` (which does not open a
        # connection on its own) and let the service open short-lived
        # sessions per method call.
        from openhands.app_server.config import get_global_config

        async_session_maker = (
            await get_global_config().db_session.get_async_session_maker()
        )
        yield SQLEventCallbackService(async_session_maker=async_session_maker)
