"""Public Event router for OpenHands Server."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.config import depends_public_event_service
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.sharing.public_event_service import PublicEventService
from openhands.sdk import Event

router = APIRouter(prefix='/public-events', tags=['Public Events'])

public_event_service_dependency = depends_public_event_service()

# Attach dependency to router for testing
router.public_event_service_dependency = public_event_service_dependency


# Read methods


@router.get('/search')
async def search_public_events(
    conversation_id: Annotated[
        str,
        Query(title='Conversation ID to search events for'),
    ],
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
    public_event_service: PublicEventService = public_event_service_dependency,
) -> EventPage:
    """Search / List events for a public conversation."""
    assert limit > 0
    assert limit <= 100
    return await public_event_service.search_public_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
    )


@router.get('/count')
async def count_public_events(
    conversation_id: Annotated[
        UUID,
        Query(title='Conversation ID to count events for'),
    ],
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
    public_event_service: PublicEventService = public_event_service_dependency,
) -> int:
    """Count events for a public conversation matching the given filters."""
    return await public_event_service.count_public_events(
        conversation_id=conversation_id,
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
    )


@router.get('')
async def batch_get_public_events(
    conversation_id: Annotated[
        UUID,
        Query(title='Conversation ID to get events for'),
    ],
    id: Annotated[list[str], Query()],
    public_event_service: PublicEventService = public_event_service_dependency,
) -> list[Event | None]:
    """Get a batch of events for a public conversation given their ids, returning null for any missing event."""
    assert len(id) <= 100
    events = await public_event_service.batch_get_public_events(conversation_id, id)
    return events


@router.get('/{conversation_id}/{event_id}')
async def get_public_event(
    conversation_id: UUID,
    event_id: str,
    public_event_service: PublicEventService = public_event_service_dependency,
) -> Event | None:
    """Get a single event from a public conversation by conversation_id and event_id."""
    return await public_event_service.get_public_event(conversation_id, event_id)
