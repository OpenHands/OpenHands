"""Sub-agent event router for OpenHands App Server.

Provides: GET /conversation/{conversation_id}/subagents/{tool_call_id}/events
"""

from uuid import UUID

from fastapi import APIRouter

from openhands.app_server.config import depends_event_service
from openhands.app_server.event.event_service import EventService
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.sdk import Event

subagent_router = APIRouter(
    prefix='/conversation/{conversation_id}',
    tags=['Events'],
    dependencies=get_dependencies(),
)
event_service_dependency = depends_event_service()


@subagent_router.get('/subagents/{tool_call_id}/events')
async def search_subagent_events(
    conversation_id: str,
    tool_call_id: str,
    event_service: EventService = event_service_dependency,
) -> list[Event]:
    """Return all events for a sub-agent task identified by tool_call_id."""
    return await event_service.search_subagent_events(
        UUID(conversation_id), tool_call_id
    )
