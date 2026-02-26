"""Event router for OpenHands App Server."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.config import depends_event_service
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.sdk import Event
from openhands.server.dependencies import get_dependencies

# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(
    prefix='/conversation/{conversation_id}/events',
    tags=['Events'],
    dependencies=get_dependencies(),
)
event_service_dependency = depends_event_service()


# Read methods


@router.get('/search')
async def search_events(
    conversation_id: str,
    kind__eq: Annotated[
        EventKind | None,
        Query(title='Optional filter by event kind'),
    ] = None,
    timestamp__gte: Annotated[
        datetime | None,
        Query(title='Optional filter by timestamp greater than or equal to'),
    ] = None,
    timestamp__lt: Annotated[
        datetime | None,
        Query(title='Optional filter by timestamp less than'),
    ] = None,
    sort_order: Annotated[
        EventSortOrder,
        Query(title='Sort order for results'),
    ] = EventSortOrder.TIMESTAMP,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, lte=100),
    ] = 100,
    event_service: EventService = event_service_dependency,
) -> EventPage:
    """Search / List events."""
    assert limit > 0
    assert limit <= 100
    return await event_service.search_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
    )


@router.get('/count')
async def count_events(
    conversation_id: str,
    kind__eq: Annotated[
        EventKind | None,
        Query(title='Optional filter by event kind'),
    ] = None,
    timestamp__gte: Annotated[
        datetime | None,
        Query(title='Optional filter by timestamp greater than or equal to'),
    ] = None,
    timestamp__lt: Annotated[
        datetime | None,
        Query(title='Optional filter by timestamp less than'),
    ] = None,
    event_service: EventService = event_service_dependency,
) -> int:
    """Count events matching the given filters."""
    return await event_service.count_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
    )


@router.get('')
async def batch_get_events(
    conversation_id: str,
    id: Annotated[list[str], Query()],
    event_service: EventService = event_service_dependency,
) -> list[Event | None]:
    """Get a batch of events given their ids, returning null for any missing event."""
    event_ids = [UUID(id_) for id_ in id]
    assert len(id) <= 100
    events = await event_service.batch_get_events(UUID(conversation_id), event_ids)
    return events
