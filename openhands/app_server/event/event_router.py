"""Event router for OpenHands App Server with agent-server proxy fallback."""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.config import (
    depends_app_conversation_service,
    depends_event_service,
    depends_httpx_client,
    depends_sandbox_service,
    depends_sandbox_spec_service,
)
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk import Event

# Lazy import of _get_agent_server_context to avoid circular import
_agent_server_context_fn = None


def _get_agent_server_context_fn():
    global _agent_server_context_fn
    if _agent_server_context_fn is None:
        from openhands.app_server.app_conversation.app_conversation_router import (  # noqa: E402
            _get_agent_server_context,
        )

        _agent_server_context_fn = _get_agent_server_context
    return _agent_server_context_fn


async def _proxy_request(
    request: Request,
    http_client: httpx.AsyncClient,
    agent_server_url: str,
    session_api_key: str | None,
    path_suffix: str,
) -> Response | None:
    """Forward the raw HTTP request to the agent-server inside the sandbox."""
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "connection", "transfer-encoding")
    }
    if session_api_key:
        headers["X-Session-API-Key"] = session_api_key

    url = f"{agent_server_url}/api/conversations/{path_suffix}"

    try:
        proxy_request = http_client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            params=request.query_params,
            content=request.stream(),
        )
        resp = await http_client.send(proxy_request, stream=False)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception as exc:
        logger.warning("Event proxy failed for %s: %s", path_suffix, exc)
        return None


# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(
    prefix='/conversation/{conversation_id}/events',
    tags=['Events'],
    dependencies=get_dependencies(),
)
event_service_dependency = depends_event_service()


# Read methods


@router.get('/search', response_model=None)
async def search_events(
    request: Request,
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
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 100,
    event_service: EventService = event_service_dependency,
    app_conversation_service: Any = depends_app_conversation_service(),
    sandbox_service: Any = depends_sandbox_service(),
    sandbox_spec_service: Any = depends_sandbox_spec_service(),
    httpx_client: httpx.AsyncClient = depends_httpx_client(),
) -> Response | EventPage:
    """Search / List events. Falls back to agent-server proxy if local store is empty."""
    local_result = await event_service.search_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
        sort_order=sort_order,
        page_id=page_id,
        limit=limit,
    )
    if local_result.items:
        return local_result

    logger.info(
        'No events in local store for conversation %s, proxying search to agent-server',
        conversation_id,
    )

    ctx = await _get_agent_server_context_fn()(
        UUID(conversation_id),
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    if ctx is None:
        return local_result

    return await _proxy_request(
        request=request,
        http_client=httpx_client,
        agent_server_url=ctx.agent_server_url,
        session_api_key=ctx.session_api_key,
        path_suffix=f'{conversation_id}/events/search',
    ) or local_result


@router.get('/count', response_model=None)
async def count_events(
    request: Request,
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
    app_conversation_service: Any = depends_app_conversation_service(),
    sandbox_service: Any = depends_sandbox_service(),
    sandbox_spec_service: Any = depends_sandbox_spec_service(),
    httpx_client: httpx.AsyncClient = depends_httpx_client(),
) -> Response | int:
    """Count events matching the given filters. Falls back to agent-server proxy if local store is empty."""
    local_count = await event_service.count_events(
        conversation_id=UUID(conversation_id),
        kind__eq=kind__eq,
        timestamp__gte=timestamp__gte,
        timestamp__lt=timestamp__lt,
    )
    if local_count > 0:
        return local_count

    logger.info(
        'No events in local store for conversation %s, proxying count to agent-server',
        conversation_id,
    )

    ctx = await _get_agent_server_context_fn()(
        UUID(conversation_id),
        app_conversation_service,
        sandbox_service,
        sandbox_spec_service,
    )
    if ctx is None:
        return local_count

    return await _proxy_request(
        request=request,
        http_client=httpx_client,
        agent_server_url=ctx.agent_server_url,
        session_api_key=ctx.session_api_key,
        path_suffix=f'{conversation_id}/events/count',
    ) or Response(content=str(local_count).encode(), status_code=200)


@router.get('')
async def batch_get_events(
    conversation_id: str,
    id: Annotated[list[str], Query()],
    event_service: EventService = event_service_dependency,
) -> list[Event | None]:
    """Get a batch of events given their ids, returning null for any missing event."""
    if len(id) > 100:
        raise HTTPException(
            status_code=400,
            detail=f'Cannot request more than 100 events at once, got {len(id)}',
        )
    event_ids = [UUID(id_) for id_ in id]
    events = await event_service.batch_get_events(UUID(conversation_id), event_ids)
    return events
