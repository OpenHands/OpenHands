from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

# Type alias for UUID and utc_now function
from datetime import datetime, UTC
from uuid import UUID

OpenHandsUUID = UUID

def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(UTC)

# Temporarily comment out SDK import
# from openhands.sdk.event.types import EventID
EventID = str


class EventCallbackResultStatus(Enum):
    SUCCESS = 'SUCCESS'
    ERROR = 'ERROR'


class EventCallbackResultSortOrder(Enum):
    CREATED_AT = 'CREATED_AT'
    CREATED_AT_DESC = 'CREATED_AT_DESC'


class EventCallbackResult(BaseModel):
    """Object representing the result of an event callback."""

    id: OpenHandsUUID = Field(default_factory=uuid4)
    status: EventCallbackResultStatus
    event_callback_id: OpenHandsUUID
    event_id: EventID
    conversation_id: OpenHandsUUID
    detail: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class EventCallbackResultPage(BaseModel):
    items: list[EventCallbackResult]
    next_page_id: str | None = None
