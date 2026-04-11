"""Shared Event router for OpenHands Server.

All endpoints in this router are unauthenticated — shared conversations are
public.  To avoid returning internal system state that the viewer does not
need, ``ConversationStateUpdateEvent`` instances are filtered out before the
response is sent.  The shared-conversation frontend only renders messages,
actions, observations, errors, and hook-execution events; state snapshots
are consumed exclusively by the authenticated WebSocket path.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from server.sharing.shared_event_service import (
    SharedEventService,
    SharedEventServiceInjector,
)

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.sdk import Event
from openhands.sdk.event.conversation_state import ConversationStateUpdateEvent
from openhands.utils.environment import StorageProvider, get_storage_provider


def _is_viewable(event: Event) -> bool:
    """Return True if *event* should be included in public shared responses."""
    return not isinstance(event, ConversationStateUpdateEvent)


def get_shared_event_service_injector() -> SharedEventServiceInjector:
    """Get the appropriate SharedEventServiceInjector based on configuration.

    Uses get_storage_provider() to determine the storage backend.
    See openhands.utils.environment for supported environment variables.

    Note: Shared events only support AWS and GCP storage. Filesystem storage
    falls back to GCP for shared events.
    """
    provider = get_storage_provider()

    if provider == StorageProvider.AWS:
        from server.sharing.aws_shared_event_service import (
            AwsSharedEventServiceInjector,
        )

        return AwsSharedEventServiceInjector()
    elif provider == StorageProvider.FILESYSTEM:
        from server.sharing.filesystem_shared_event_service import (
            FilesystemSharedEventServiceInjector,
        )

        return FilesystemSharedEventServiceInjector()
    else:
        # GCP is the default for shared events (including filesystem fallback)
        from server.sharing.google_cloud_shared_event_service import (
            GoogleCloudSharedEventServiceInjector,
        )

        return GoogleCloudSharedEventServiceInjector()


router = APIRouter(prefix='/api/shared-events', tags=['Sharing'])
shared_event_service_dependency = Depends(get_shared_event_service_injector().depends)


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
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 100,
    shared_event_service: SharedEventService = shared_event_service_dependency,
) -> EventPage:
    """Search / List events for a shared conversation."""
    page = await shared_event_service.search_shared_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
    )
    return EventPage(
        items=[e for e in page.items if _is_viewable(e)],
        next_page_id=page.next_page_id,
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
    if len(id) > 100:
        raise HTTPException(
            status_code=400,
            detail=f'Cannot request more than 100 events at once, got {len(id)}',
        )
    event_ids = [UUID(id_) for id_ in id]
    events = await shared_event_service.batch_get_shared_events(
        UUID(conversation_id), event_ids
    )
    return [e if e is not None and _is_viewable(e) else None for e in events]


@router.get('/{conversation_id}/{event_id}')
async def get_shared_event(
    conversation_id: str,
    event_id: str,
    shared_event_service: SharedEventService = shared_event_service_dependency,
) -> Event | None:
    """Get a single event from a shared conversation by conversation_id and event_id."""
    event = await shared_event_service.get_shared_event(
        UUID(conversation_id), UUID(event_id)
    )
    if event is not None and not _is_viewable(event):
        return None
    return event
