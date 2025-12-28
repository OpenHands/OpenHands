# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from pydantic import Field

from openhands.agent_server.utils import OpenHandsUUID, utc_now
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.sdk import Event
from openhands.sdk.utils.models import (
    DiscriminatedUnionMixin,
    OpenHandsModel,
    get_known_concrete_subclasses,
)

_logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    EventKind = str
else:
    EventKind = Literal[tuple(c.__name__ for c in get_known_concrete_subclasses(Event))]


class EventCallbackStatus(Enum):
    ACTIVE = 'ACTIVE'
    DISABLED = 'DISABLED'
    COMPLETED = 'COMPLETED'
    ERROR = 'ERROR'


class EventCallbackProcessor(DiscriminatedUnionMixin, ABC):
    @abstractmethod
    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        """Process an event."""


class LoggingCallbackProcessor(EventCallbackProcessor):
    """Example implementation which logs callbacks."""

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult:
        _logger.info(f'Callback {callback.id} Invoked for event {event}')
        return EventCallbackResult(
            status=EventCallbackResultStatus.SUCCESS,
            event_callback_id=callback.id,
            event_id=event.id,
            conversation_id=conversation_id,
        )


class CreateEventCallbackRequest(OpenHandsModel):
    conversation_id: OpenHandsUUID | None = Field(
        default=None,
        description=(
            'Optional filter on the conversation to which this callback applies'
        ),
    )
    processor: EventCallbackProcessor
    event_kind: EventKind | None = Field(
        default=None,
        description=(
            'Optional filter on the type of events to which this callback applies'
        ),
    )


class EventCallback(CreateEventCallbackRequest):
    id: OpenHandsUUID = Field(default_factory=uuid4)
    status: EventCallbackStatus = Field(default=EventCallbackStatus.ACTIVE)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EventCallbackPage(OpenHandsModel):
    items: list[EventCallback]
    next_page_id: str | None = None
