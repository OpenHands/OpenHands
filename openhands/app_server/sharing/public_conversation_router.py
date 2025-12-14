"""Public Conversation router for OpenHands Server."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from openhands.app_server.config import depends_public_conversation_info_service
from openhands.app_server.sharing.public_conversation_info_service import (
    PublicConversationInfoService,
)
from openhands.app_server.sharing.public_conversation_models import (
    PublicConversation,
    PublicConversationPage,
    PublicConversationSortOrder,
)

router = APIRouter(prefix='/public-conversations', tags=['Public Conversations'])

public_conversation_service_dependency = depends_public_conversation_info_service()

# Attach dependency to router for testing
router.public_conversation_service_dependency = public_conversation_service_dependency


# Read methods


@router.get('/search')
async def search_public_conversations(
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
        PublicConversationSortOrder,
        Query(title='Sort order for results'),
    ] = PublicConversationSortOrder.CREATED_AT_DESC,
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
    public_conversation_service: PublicConversationInfoService = public_conversation_service_dependency,
) -> PublicConversationPage:
    """Search / List public conversations."""
    assert limit > 0
    assert limit <= 100
    return await public_conversation_service.search_public_conversation_info(
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
    public_conversation_service: PublicConversationInfoService = public_conversation_service_dependency,
) -> int:
    """Count public conversations matching the given filters."""
    return await public_conversation_service.count_public_conversation_info(
        title__contains=title__contains,
        created_at__gte=created_at__gte,
        created_at__lt=created_at__lt,
        updated_at__gte=updated_at__gte,
        updated_at__lt=updated_at__lt,
    )


@router.get('')
async def batch_get_public_conversations(
    ids: Annotated[list[str], Query()],
    public_conversation_service: PublicConversationInfoService = public_conversation_service_dependency,
) -> list[PublicConversation | None]:
    """Get a batch of public conversations given their ids. Return None for any missing or non-public."""
    assert len(ids) <= 100
    uuids = [UUID(id_) for id_ in ids]
    public_conversations = (
        await public_conversation_service.batch_get_public_conversation_info(uuids)
    )
    return public_conversations
