"""Models for pending message queue functionality."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.models import ImageContent, TextContent
from openhands.agent_server.utils import utc_now


class PendingMessage(BaseModel):
    """Represent a message queued for delivery."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str  # Can be task-{uuid} or real conversation UUID
    role: str = 'user'
    content: list[TextContent | ImageContent]
    created_at: datetime = Field(default_factory=utc_now)


class PendingMessageResponse(BaseModel):
    """Response when queueing a pending message."""

    id: str
    queued: bool
    position: int = Field(description='Position in the queue (1-based)')
    conversation_id: str | None = None
