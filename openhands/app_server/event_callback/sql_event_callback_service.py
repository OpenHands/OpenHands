# pyright: reportArgumentType=false
"""SQL implementation of EventCallbackService."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator
from uuid import UUID, uuid4

from fastapi import Request
from sqlalchemy import UUID as SQLUUID
from sqlalchemy import Column, Enum, String, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

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


class StoredEventCallback(Base):  # type: ignore
    __tablename__ = 'event_callback'
    id = Column(SQLUUID, primary_key=True)
    conversation_id = Column(SQLUUID, nullable=True)
    status = Column(
        Enum(EventCallbackStatus), nullable=False, default=EventCallbackStatus.ACTIVE
    )
    processor = Column(create_json_type_decorator(EventCallbackProcessor))
    event_kind = Column(String, nullable=True)
    created_at = Column(UtcDateTime, server_default=func.now(), index=True)
    updated_at = Column(UtcDateTime, server_default=func.now(), index=True)


class StoredEventCallbackResult(Base):  # type: ignore
    __tablename__ = 'event_callback_result'
    id = Column(SQLUUID, primary_key=True, default=uuid4)
    status = Column(Enum(EventCallbackResultStatus), nullable=True)
    event_callback_id = Column(SQLUUID, index=True)
    event_id = Column(String, index=True)
    conversation_id = Column(SQLUUID, index=True)
    detail = Column(String, nullable=True)
    created_at = Column(UtcDateTime, server_default=func.now(), index=True)


@dataclass
class SQLEventCallbackService(EventCallbackService):
    """SQL implementation of EventCallbackService."""

    db_session: AsyncSession

    async def create_event_callback(
        self, request: CreateEventCallbackRequest
    ) -> EventCallback:
        """Create a new event callback."""
        # Create EventCallback from request
        event_callback = EventCallback(
            conversation_id=request.conversation_id,
            processor=request.processor,
            event_kind=request.event_kind,
        )

        # Create stored version and add to db_session
        stored_callback = StoredEventCallback(**event_callback.model_dump())
        self.db_session.add(stored_callback)
        await self.db_session.commit()
        await self.db_session.refresh(stored_callback)
        return EventCallback.model_validate(row2dict(stored_callback))

    async def get_event_callback(self, id: UUID) -> EventCallback | None:
        """Get a single event callback, returning None if not found."""
        stmt = select(StoredEventCallback).where(StoredEventCallback.id == id)
        result = await self.db_session.execute(stmt)
        stored_callback = result.scalar_one_or_none()
        if stored_callback:
            return EventCallback.model_validate(row2dict(stored_callback))
        return None

    async def delete_event_callback(self, id: UUID) -> bool:
        """Delete an event callback, returning True if deleted, False if not found."""
        stmt = select(StoredEventCallback).where(StoredEventCallback.id == id)
        result = await self.db_session.execute(stmt)
        stored_callback = result.scalar_one_or_none()

        if stored_callback is None:
            return False

        await self.db_session.delete(stored_callback)
        await self.db_session.commit()
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
        # Build the query with filters
        conditions = []

        if conversation_id__eq is not None:
            conditions.append(
                StoredEventCallback.conversation_id == conversation_id__eq
            )

        if event_kind__eq is not None:
            conditions.append(StoredEventCallback.event_kind == event_kind__eq)

        # Note: event_id__eq is not stored in the event_callbacks table
        # This parameter might be used for filtering results after retrieval
        # or might be intended for a different use case

        # Build the base query
        stmt = select(StoredEventCallback)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Handle pagination
        if page_id is not None:
            # Parse page_id to get offset or cursor
            try:
                offset = int(page_id)
                stmt = stmt.offset(offset)
            except ValueError:
                # If page_id is not a valid integer, start from beginning
                offset = 0
        else:
            offset = 0

        # Apply limit and get one extra to check if there are more results
        stmt = stmt.limit(limit + 1).order_by(StoredEventCallback.created_at.desc())

        result = await self.db_session.execute(stmt)
        stored_callbacks = result.scalars().all()

        # Check if there are more results
        has_more = len(stored_callbacks) > limit
        if has_more:
            stored_callbacks = stored_callbacks[:limit]

        # Calculate next page ID
        next_page_id = None
        if has_more:
            next_page_id = str(offset + limit)

        # Convert stored callbacks to domain models
        callbacks = [
            EventCallback.model_validate(row2dict(cb)) for cb in stored_callbacks
        ]
        return EventCallbackPage(items=callbacks, next_page_id=next_page_id)

    async def save_event_callback(self, event_callback: EventCallback) -> EventCallback:
        event_callback.updated_at = utc_now()
        stored_callback = StoredEventCallback(**event_callback.model_dump())
        await self.db_session.merge(stored_callback)
        return event_callback

    async def execute_callbacks(self, conversation_id: UUID, event: Event) -> None:
        query = (
            select(StoredEventCallback)
            .where(StoredEventCallback.status == EventCallbackStatus.ACTIVE)
            .where(
                or_(
                    StoredEventCallback.event_kind == event.kind,
                    StoredEventCallback.event_kind.is_(None),
                )
            )
            .where(
                or_(
                    StoredEventCallback.conversation_id == conversation_id,
                    StoredEventCallback.conversation_id.is_(None),
                )
            )
        )
        result = await self.db_session.execute(query)
        stored_callbacks = result.scalars().all()
        if stored_callbacks:
            callbacks = [
                EventCallback.model_validate(row2dict(cb)) for cb in stored_callbacks
            ]
            await asyncio.gather(
                *[
                    self.execute_callback(conversation_id, callback, event)
                    for callback in callbacks
                ]
            )

            # Persist any new changes callbacks may have made to itself
            for callback in callbacks:
                await self.save_event_callback(callback)
            await self.db_session.commit()

    async def execute_callback(
        self, conversation_id: UUID, callback: EventCallback, event: Event
    ):
        try:
            result = await callback.processor(conversation_id, callback, event)
            if result is None:
                return
            stored_result = StoredEventCallbackResult(**result.model_dump())
        except Exception as exc:
            _logger.exception(f'Exception in callback {callback.id}', stack_info=True)
            stored_result = StoredEventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=str(exc),
            )
        self.db_session.add(stored_result)

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Stop using this event callback service."""
        pass


class SQLEventCallbackServiceInjector(EventCallbackServiceInjector):
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[EventCallbackService, None]:
        from openhands.app_server.config import get_db_session

        async with get_db_session(state) as db_session:
            yield SQLEventCallbackService(db_session=db_session)
