from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.utils import OpenHandsUUID, utc_now
from openhands.sdk.event import EventID


class EventCallbackResultStatus(Enum):
    SUCCESS = 'SUCCESS'
    ERROR = 'ERROR'


class EventCallbackResult(BaseModel):
    """Object representing the result of an event callback."""

    id: OpenHandsUUID = Field(default_factory=uuid4)
    status: EventCallbackResultStatus
    event_callback_id: OpenHandsUUID
    event_id: EventID
    conversation_id: OpenHandsUUID
    detail: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
