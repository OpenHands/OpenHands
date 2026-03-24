"""Models for pending message queue functionality."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.models import ImageContent, TextContent
from openhands.agent_server.utils import utc_now


class PendingMessage(BaseModel):
    """A message queued for delivery when conversation becomes ready.

    Pending messages are stored in the database and delivered to the agent_server
    when the conversation transitions to READY status. Messages are deleted after
    processing, regardless of success or failure.
    """

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
