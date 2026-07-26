import asyncio
import logging
from typing import ClassVar
from uuid import UUID

import httpx

from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
    EventCallbackStatus,
    EventKind,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)
from openhands.sdk import Event, MessageEvent
from openhands.sdk.utils.redact import redact_text_secrets

_logger = logging.getLogger(__name__)

# Delay between attempts to poll title
_POLL_DELAY_S = 3
# Number of attempts to poll title
_NUM_POLL_ATTEMPTS = 4
# Avoid starting one slow title poll per webhook event. This set is scoped to a
# worker process; with multiple workers, at most one poll per worker and
# conversation can run at a time.
_CONVERSATIONS_BEING_POLLED: set[UUID] = set()


async def _poll_for_title(
    httpx_client: httpx.AsyncClient,
    url: str,
    session_api_key: str | None,
) -> str | None:
    """Poll the agent server for the conversation title.

    Args:
        httpx_client: The HTTP client to use for requests.
        url: The conversation URL to poll.
        session_api_key: The session API key for authentication.

    Returns:
        The title if available, None otherwise.
    """
    for _ in range(_NUM_POLL_ATTEMPTS):
        await asyncio.sleep(_POLL_DELAY_S)
        try:
            headers = (
                {
                    'X-Session-API-Key': session_api_key,
                }
                if session_api_key
                else {}
            )
            response = await httpx_client.get(
                url,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Transient agent-server failures are acceptable; retry later.
            _logger.warning(
                'Title poll failed for conversation %s: %s',
                url,
                exc,
            )
        else:
            title = response.json().get('title')
            if title:
                return title

    return None


class SetTitleCallbackProcessor(EventCallbackProcessor):
    """Callback processor which sets conversation titles."""

    event_kind: ClassVar[EventKind] = 'MessageEvent'

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        if not isinstance(event, MessageEvent):
            return None

        if conversation_id in _CONVERSATIONS_BEING_POLLED:
            _logger.debug(
                'Skipping duplicate title poll for conversation %s',
                conversation_id,
            )
            return None

        _CONVERSATIONS_BEING_POLLED.add(conversation_id)
        try:
            return await self._process_callback(
                conversation_id,
                callback,
                event,
            )
        finally:
            _CONVERSATIONS_BEING_POLLED.discard(conversation_id)

    async def _process_callback(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: MessageEvent,
    ) -> EventCallbackResult | None:
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_app_conversation_service,
            get_event_callback_service,
            get_httpx_client,
        )

        _logger.info(
            'Callback %s Invoked for event %s',
            callback.id,
            redact_text_secrets(str(event)),
        )

        read_state = InjectorState()
        setattr(read_state, USER_CONTEXT_ATTR, ADMIN)
        async with get_app_conversation_service(read_state) as app_conversation_service:
            app_conversation = await app_conversation_service.get_app_conversation(
                conversation_id
            )
            assert app_conversation is not None
            app_conversation_url = app_conversation.conversation_url
            assert app_conversation_url is not None
            app_conversation_url = replace_localhost_hostname_for_docker(
                app_conversation_url
            )
            session_api_key = app_conversation.session_api_key

        http_state = InjectorState()
        setattr(http_state, USER_CONTEXT_ATTR, ADMIN)
        async with get_httpx_client(http_state) as httpx_client:
            title = await _poll_for_title(
                httpx_client,
                app_conversation_url,
                session_api_key,
            )

        if not title:
            # Keep the callback active so later message events can retry.
            _logger.info(
                f'Conversation {conversation_id} title not available yet; '
                'will retry on a future message event.'
            )
            return None

        info_state = InjectorState()
        setattr(info_state, USER_CONTEXT_ATTR, ADMIN)
        async with get_app_conversation_info_service(
            info_state
        ) as app_conversation_info_service:
            info = await app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
            assert info is not None
            info.title = title
            await app_conversation_info_service.save_app_conversation_info(info)

        callback_state = InjectorState()
        setattr(callback_state, USER_CONTEXT_ATTR, ADMIN)
        async with get_event_callback_service(callback_state) as event_callback_service:
            callback.status = EventCallbackStatus.DISABLED
            await event_callback_service.save_event_callback(callback)

        return EventCallbackResult(
            status=EventCallbackResultStatus.SUCCESS,
            event_callback_id=callback.id,
            event_id=event.id,
            conversation_id=conversation_id,
        )
