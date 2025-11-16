"""Sandboxed Conversation router for OpenHands Server."""

import asyncio
import sys
from datetime import datetime
from typing import Annotated, AsyncGenerator
from uuid import UUID

import httpx

from openhands.app_server.services.db_session_injector import set_db_session_keep_open
from openhands.app_server.services.httpx_client_injector import (
    set_httpx_client_keep_open,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.app_server.user.user_context import UserContext

# Handle anext compatibility for Python < 3.10
if sys.version_info >= (3, 10):
    from builtins import anext
else:

    async def anext(async_iterator):
        """Compatibility function for anext in Python < 3.10"""
        return await async_iterator.__anext__()


from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
    AppConversationPage,
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskPage,
    AppConversationStartTaskSortOrder,
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
)
from openhands.app_server.app_conversation.app_conversation_start_task_service import (
    AppConversationStartTaskService,
)
from openhands.app_server.config import (
    depends_app_conversation_service,
    depends_app_conversation_start_task_service,
    depends_db_session,
    depends_httpx_client,
    depends_user_context,
    get_app_conversation_service,
)

router = APIRouter(prefix='/app-conversations', tags=['Conversations'])
app_conversation_service_dependency = depends_app_conversation_service()
app_conversation_start_task_service_dependency = (
    depends_app_conversation_start_task_service()
)
user_context_dependency = depends_user_context()
db_session_dependency = depends_db_session()
httpx_client_dependency = depends_httpx_client()

# Read methods


@router.get('/search')
async def search_app_conversations(
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
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
) -> AppConversationPage:
    """Search / List sandboxed conversations."""
    assert limit > 0
    assert limit <= 100
    return await app_conversation_service.search_app_conversations(
        title__contains=title__contains,
        created_at__gte=created_at__gte,
        created_at__lt=created_at__lt,
        updated_at__gte=updated_at__gte,
        updated_at__lt=updated_at__lt,
        page_id=page_id,
        limit=limit,
        include_sub_conversations=include_sub_conversations,
    )


@router.get('/count')
async def count_app_conversations(
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
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
) -> int:
    """Count sandboxed conversations matching the given filters."""
    return await app_conversation_service.count_app_conversations(
        title__contains=title__contains,
        created_at__gte=created_at__gte,
        created_at__lt=created_at__lt,
        updated_at__gte=updated_at__gte,
        updated_at__lt=updated_at__lt,
    )


@router.get('')
async def batch_get_app_conversations(
    ids: Annotated[list[UUID], Query()],
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
) -> list[AppConversation | None]:
    """Get a batch of sandboxed conversations given their ids. Return None for any missing."""
    assert len(ids) < 100
    app_conversations = await app_conversation_service.batch_get_app_conversations(ids)
    return app_conversations


@router.post('')
async def start_app_conversation(
    request: Request,
    start_request: AppConversationStartRequest,
    db_session: AsyncSession = db_session_dependency,
    httpx_client: httpx.AsyncClient = httpx_client_dependency,
    app_conversation_service: AppConversationService = (
        app_conversation_service_dependency
    ),
) -> AppConversationStartTask:
    # Because we are processing after the request finishes, keep the db connection open
    set_db_session_keep_open(request.state, True)
    set_httpx_client_keep_open(request.state, True)

    """Start an app conversation start task and return it."""
    async_iter = app_conversation_service.start_app_conversation(start_request)
    result = await anext(async_iter)
    asyncio.create_task(_consume_remaining(async_iter, db_session, httpx_client))
    return result


@router.post('/stream-start')
async def stream_app_conversation_start(
    request: AppConversationStartRequest,
    user_context: UserContext = user_context_dependency,
) -> list[AppConversationStartTask]:
    """Start an app conversation start task and stream updates from it.
    Leaves the connection open until either the conversation starts or there was an error
    """
    response = StreamingResponse(
        _stream_app_conversation_start(request, user_context),
        media_type='application/json',
    )
    return response


@router.get('/start-tasks/search')
async def search_app_conversation_start_tasks(
    conversation_id__eq: Annotated[
        UUID | None,
        Query(title='Filter by conversation ID equal to this value'),
    ] = None,
    created_at__gte: Annotated[
        datetime | None,
        Query(title='Filter by created_at greater than or equal to this datetime'),
    ] = None,
    sort_order: Annotated[
        AppConversationStartTaskSortOrder,
        Query(title='Sort order for the results'),
    ] = AppConversationStartTaskSortOrder.CREATED_AT_DESC,
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
    app_conversation_start_task_service: AppConversationStartTaskService = (
        app_conversation_start_task_service_dependency
    ),
) -> AppConversationStartTaskPage:
    """Search / List conversation start tasks."""
    assert limit > 0
    assert limit <= 100
    return (
        await app_conversation_start_task_service.search_app_conversation_start_tasks(
            conversation_id__eq=conversation_id__eq,
            created_at__gte=created_at__gte,
            sort_order=sort_order,
            page_id=page_id,
            limit=limit,
        )
    )


@router.get('/start-tasks/count')
async def count_app_conversation_start_tasks(
    conversation_id__eq: Annotated[
        UUID | None,
        Query(title='Filter by conversation ID equal to this value'),
    ] = None,
    created_at__gte: Annotated[
        datetime | None,
        Query(title='Filter by created_at greater than or equal to this datetime'),
    ] = None,
    app_conversation_start_task_service: AppConversationStartTaskService = (
        app_conversation_start_task_service_dependency
    ),
) -> int:
    """Count conversation start tasks matching the given filters."""
    return await app_conversation_start_task_service.count_app_conversation_start_tasks(
        conversation_id__eq=conversation_id__eq,
        created_at__gte=created_at__gte,
    )


@router.get('/start-tasks')
async def batch_get_app_conversation_start_tasks(
    ids: Annotated[list[UUID], Query()],
    app_conversation_start_task_service: AppConversationStartTaskService = (
        app_conversation_start_task_service_dependency
    ),
) -> list[AppConversationStartTask | None]:
    """Get a batch of start app conversation tasks given their ids. Return None for any missing."""
    assert len(ids) < 100
    start_tasks = await app_conversation_start_task_service.batch_get_app_conversation_start_tasks(
        ids
    )
    return start_tasks


async def _consume_remaining(
    async_iter, db_session: AsyncSession, httpx_client: httpx.AsyncClient
):
    """Consume the remaining items from an async iterator"""
    try:
        while True:
            await anext(async_iter)
    except StopAsyncIteration:
        return
    finally:
        await db_session.close()
        await httpx_client.aclose()


async def _stream_app_conversation_start(
    request: AppConversationStartRequest,
    user_context: UserContext,
) -> AsyncGenerator[str, None]:
    """Stream a json list, item by item."""

    # Because the original dependencies are closed after the method returns, we need
    # a new dependency context which will continue intil the stream finishes.
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, user_context)
    async with get_app_conversation_service(state) as app_conversation_service:
        yield '[\n'
        comma = False
        async for task in app_conversation_service.start_app_conversation(request):
            chunk = task.model_dump_json()
            if comma:
                chunk = ',\n' + chunk
            comma = True
            yield chunk
        yield ']'
