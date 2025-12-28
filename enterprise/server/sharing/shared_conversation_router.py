"""Shared Conversation router for OpenHands Server."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from server.sharing.shared_conversation_info_service import (
    SharedConversationInfoService,
)
from server.sharing.shared_conversation_models import (
    SharedConversation,
    SharedConversationPage,
    SharedConversationSortOrder,
)
from server.sharing.sql_shared_conversation_info_service import (
    SQLSharedConversationInfoServiceInjector,
)

router = APIRouter(prefix='/api/shared-conversations', tags=['Sharing'])
shared_conversation_info_service_dependency = Depends(
    SQLSharedConversationInfoServiceInjector().depends
)

# Read methods


@router.get('/search')
async def search_shared_conversations(
    title__contains: Annotated[
        str | None,
        Query(title='Filter by title containing this string'),
    ] = None,
    created_at__gte: Annotated[
        datetime | None,
        Query(title='Filter by created_at greater than or equal to this datetime'),
    ] = None,
    created_at__lt: Annotated[
        datetime | None,
        Query(title='Filter by created_at less than this datetime'),
    ] = None,
    updated_at__gte: Annotated[
        datetime | None,
        Query(title='Filter by updated_at greater than or equal to this datetime'),
    ] = None,
    updated_at__lt: Annotated[
        datetime | None,
        Query(title='Filter by updated_at less than this datetime'),
    ] = None,
    sort_order: Annotated[
        SharedConversationSortOrder,
        Query(title='Sort order for results'),
    ] = SharedConversationSortOrder.CREATED_AT_DESC,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(
            title='The max number of results in the page',
            gt=0,
            lte=100,
        ),
    ] = 100,
    include_sub_conversations: Annotated[
        bool,
        Query(
            title='If True, include sub-conversations in the results. If False (default), exclude all sub-conversations.'
        ),
    ] = False,
    shared_conversation_service: SharedConversationInfoService = shared_conversation_info_service_dependency,
) -> SharedConversationPage:
    """Search / List shared conversations."""
    assert limit > 0
    assert limit <= 100
    return await shared_conversation_service.search_shared_conversation_info(
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
async def count_shared_conversations(
    title__contains: Annotated[
        str | None,
        Query(title='Filter by title containing this string'),
    ] = None,
    created_at__gte: Annotated[
        datetime | None,
        Query(title='Filter by created_at greater than or equal to this datetime'),
    ] = None,
    created_at__lt: Annotated[
        datetime | None,
        Query(title='Filter by created_at less than this datetime'),
    ] = None,
    updated_at__gte: Annotated[
        datetime | None,
        Query(title='Filter by updated_at greater than or equal to this datetime'),
    ] = None,
    updated_at__lt: Annotated[
        datetime | None,
        Query(title='Filter by updated_at less than this datetime'),
    ] = None,
    shared_conversation_service: SharedConversationInfoService = shared_conversation_info_service_dependency,
) -> int:
    """Count shared conversations matching the given filters."""
    return await shared_conversation_service.count_shared_conversation_info(
        title__contains=title__contains,
        created_at__gte=created_at__gte,
        created_at__lt=created_at__lt,
        updated_at__gte=updated_at__gte,
        updated_at__lt=updated_at__lt,
    )


@router.get('')
async def batch_get_shared_conversations(
    ids: Annotated[list[str], Query()],
    shared_conversation_service: SharedConversationInfoService = shared_conversation_info_service_dependency,
) -> list[SharedConversation | None]:
    """Get a batch of shared conversations given their ids. Return None for any missing or non-shared."""
    assert len(ids) <= 100
    uuids = [UUID(id_) for id_ in ids]
    shared_conversation_info = (
        await shared_conversation_service.batch_get_shared_conversation_info(uuids)
    )
    return shared_conversation_info
