from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from uuid import UUID
from openhands.sdk.event.types import EventID


class EventCallbackResultStatus(Enum):
    SUCCESS = 'SUCCESS'
    ERROR = 'ERROR'


class EventCallbackResultSortOrder(Enum):
    CREATED_AT = 'CREATED_AT'
    CREATED_AT_DESC = 'CREATED_AT_DESC'


class EventCallbackResult(BaseModel):
    """Object representing the result of an event callback."""

    id: UUID = Field(default_factory=uuid4)
    status: EventCallbackResultStatus
    event_callback_id: UUID
    event_id: EventID
    conversation_id: UUID
    detail: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventCallbackResultPage(BaseModel):
    items: list[EventCallbackResult]
    next_page_id: str | None = None
