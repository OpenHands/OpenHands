"""Public Conversation router for OpenHands Server."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.config import (
    depends_public_conversation_info_service,
    depends_public_event_service,
)
from openhands.app_server.public_conversations.public_conversation_info_service import (
    PublicConversationInfoService,
)
from openhands.app_server.public_conversations.public_conversation_models import (
    PublicConversationInfo,
    PublicConversationInfoPage,
    PublicConversationSortOrder,
)
from openhands.app_server.public_conversations.public_event_service import (
    PublicEventService,
)
from openhands.sdk import Event

router = APIRouter(prefix='/public/conversations', tags=['Public Conversations'])
public_conversation_info_service_dependency = depends_public_conversation_info_service()
public_event_service_dependency = depends_public_event_service()


# Read methods for conversations


@router.get('/search')
async def search_public_conversations(
    title__contains: Annotated[
        str | None,
        Query(title='Optional filter by title containing text'),
    ] = None,
    created_at__gte: Annotated[
        datetime | None,
        Query(title='Optional filter by created_at greater than or equal to'),
    ] = None,
    created_at__lt: Annotated[
        datetime | None,
        Query(title='Optional filter by created_at less than'),
    ] = None,
    updated_at__gte: Annotated[
        datetime | None,
        Query(title='Optional filter by updated_at greater than or equal to'),
    ] = None,
    updated_at__lt: Annotated[
        datetime | None,
        Query(title='Optional filter by updated_at less than'),
    ] = None,
    sort_order: Annotated[
        PublicConversationSortOrder,
        Query(title='Sort order for results'),
    ] = PublicConversationSortOrder.CREATED_AT_DESC,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, lte=100),
    ] = 100,
    include_sub_conversations: Annotated[
        bool,
        Query(title='Whether to include sub-conversations'),
    ] = False,
    public_conversation_info_service: PublicConversationInfoService = public_conversation_info_service_dependency,
) -> PublicConversationInfoPage:
    """Search / List public conversations."""
    assert limit > 0
    assert limit <= 100
    return await public_conversation_info_service.search_public_conversation_info(
        title__contains=title__contains,
        created_at__gte=created_at__gte,
        created_at__lt=created_at__lt,
        updated_at__gte=updated_at__gte,
        updated_at__lt=updated_at__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
        include_sub_conversations=include_sub_conversations,
    )


@router.get('/count')
async def count_public_conversations(
    title__contains: Annotated[
        str | None,
        Query(title='Optional filter by title containing text'),
    ] = None,
    created_at__gte: Annotated[
        datetime | None,
        Query(title='Optional filter by created_at greater than or equal to'),
    ] = None,
    created_at__lt: Annotated[
        datetime | None,
        Query(title='Optional filter by created_at less than'),
    ] = None,
    updated_at__gte: Annotated[
        datetime | None,
        Query(title='Optional filter by updated_at greater than or equal to'),
    ] = None,
    updated_at__lt: Annotated[
        datetime | None,
        Query(title='Optional filter by updated_at less than'),
    ] = None,
    public_conversation_info_service: PublicConversationInfoService = public_conversation_info_service_dependency,
) -> int:
    """Count public conversations matching the given filters."""
    return await public_conversation_info_service.count_public_conversation_info(
        title__contains=title__contains,
        created_at__gte=created_at__gte,
        created_at__lt=created_at__lt,
        updated_at__gte=updated_at__gte,
        updated_at__lt=updated_at__lt,
    )


@router.get('/{conversation_id}')
async def get_public_conversation(
    conversation_id: UUID,
    public_conversation_info_service: PublicConversationInfoService = public_conversation_info_service_dependency,
) -> PublicConversationInfo | None:
    """Get a single public conversation by ID."""
    return await public_conversation_info_service.get_public_conversation_info(
        conversation_id
    )


@router.get('/token/{token}')
async def get_public_conversation_by_token(
    token: str,
    public_conversation_info_service: PublicConversationInfoService = public_conversation_info_service_dependency,
) -> PublicConversationInfo | None:
    """Get a single public conversation by share token."""
    return await public_conversation_info_service.get_public_conversation_info_by_token(
        token
    )


# Read methods for events


@router.get('/{conversation_id}/events/search')
async def search_public_conversation_events(
    conversation_id: UUID,
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
    """Search / List events from a public conversation."""
    assert limit > 0
    assert limit <= 100
    return await public_event_service.search_public_events(
        conversation_id__eq=conversation_id,
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
    )


@router.get('/{conversation_id}/events/count')
async def count_public_conversation_events(
    conversation_id: UUID,
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
    """Count events from a public conversation matching the given filters."""
    return await public_event_service.count_public_events(
        conversation_id__eq=conversation_id,
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
    )


@router.get('/{conversation_id}/events')
async def batch_get_public_conversation_events(
    conversation_id: UUID,
    id: Annotated[list[str], Query()],
    public_event_service: PublicEventService = public_event_service_dependency,
) -> list[Event | None]:
    """Get a batch of events from a public conversation given their ids, returning null for any missing event."""
    assert len(id) <= 100
    events = await public_event_service.batch_get_public_events(id)
    return events


@router.get('/token/{token}/events/search')
async def search_public_conversation_events_by_token(
    token: str,
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
    """Search / List events from a public conversation by share token."""
    assert limit > 0
    assert limit <= 100
    return await public_event_service.search_public_events_by_token(
        token=token,
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
    )