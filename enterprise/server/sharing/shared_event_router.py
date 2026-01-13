"""Shared Event router for OpenHands Server."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from server.sharing.google_cloud_shared_event_service import (
    GoogleCloudSharedEventServiceInjector,
)
from server.sharing.shared_event_service import SharedEventService

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.sdk import Event

router = APIRouter(prefix='/api/shared-events', tags=['Sharing'])
shared_event_service_dependency = Depends(
    GoogleCloudSharedEventServiceInjector().depends
)


# Read methods


@router.get('/search')
async def search_shared_events(
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
    shared_event_service: SharedEventService = shared_event_service_dependency,
) -> EventPage:
    """Search / List events for a shared conversation."""
    assert limit > 0
    assert limit <= 100
    return await shared_event_service.search_shared_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
    )


@router.get('/count')
async def count_shared_events(
    conversation_id: Annotated[
        str,
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
    shared_event_service: SharedEventService = shared_event_service_dependency,
) -> int:
    """Count events for a shared conversation matching the given filters."""
    return await shared_event_service.count_shared_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
    )


@router.get('')
async def batch_get_shared_events(
    conversation_id: Annotated[
        str,
        Query(title='Conversation ID to get events for'),
    ],
    id: Annotated[list[str], Query()],
    shared_event_service: SharedEventService = shared_event_service_dependency,
) -> list[Event | None]:
    """Get a batch of events for a shared conversation given their ids, returning null for any missing event."""
    assert len(id) <= 100
    event_ids = [UUID(id_) for id_ in id]
    events = await shared_event_service.batch_get_shared_events(
        UUID(conversation_id), event_ids
    )
    return events


@router.get('/{conversation_id}/{event_id}')
async def get_shared_event(
    conversation_id: str,
    event_id: str,
    shared_event_service: SharedEventService = shared_event_service_dependency,
) -> Event | None:
    """Get a single event from a shared conversation by conversation_id and event_id."""
    return await shared_event_service.get_shared_event(
        UUID(conversation_id), UUID(event_id)
    )
