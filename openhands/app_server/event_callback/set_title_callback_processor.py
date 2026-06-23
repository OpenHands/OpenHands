import asyncio
import logging
from typing import ClassVar
from uuid import UUID

import httpx

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
    AppConversationInfo,
)
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
from openhands.app_server.sandbox.sandbox_service import SandboxService
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


async def _resolve_internal_conversation_url(
    sandbox_service: SandboxService,
    app_conversation: AppConversation,
    conversation_id: UUID,
) -> str:
    """Resolve the agent-server conversation URL for server-side (app -> agent) use.

    ``app_conversation.conversation_url`` is the *browser-facing* URL. When the
    runtime proxy is enabled it points at the public ``/runtime/...`` proxy path,
    which the app server must not call back into. The sandbox always knows the
    direct internal agent-server URL, so prefer that; fall back to the stored URL
    (with ``localhost`` rewritten for Docker) for non-proxy deployments or if the
    sandbox can no longer be resolved.
    """
    agent_server_url = await sandbox_service.get_agent_server_url_by_id(
        app_conversation.sandbox_id
    )
    if agent_server_url:
        return f'{agent_server_url.rstrip("/")}/api/conversations/{conversation_id.hex}'
    # Fallback for non-proxy deployments; the caller guarantees a URL is present.
    assert app_conversation.conversation_url is not None
    return replace_localhost_hostname_for_docker(app_conversation.conversation_url)


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
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_app_conversation_service,
            get_event_callback_service,
            get_httpx_client,
            get_sandbox_service,
        )

        _logger.info(
            'Callback %s Invoked for event %s',
            callback.id,
            redact_text_secrets(str(event)),
        )

        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)
        async with (
            get_event_callback_service(state) as event_callback_service,
            get_app_conversation_service(state) as app_conversation_service,
            get_app_conversation_info_service(state) as app_conversation_info_service,
            get_httpx_client(state) as httpx_client,
            get_sandbox_service(state) as sandbox_service,
        ):
            app_conversation = await app_conversation_service.get_app_conversation(
                conversation_id
            )
            assert app_conversation is not None
            # Ensures the conversation is running (i.e. has an exposed URL).
            assert app_conversation.conversation_url is not None
            app_conversation_url = await _resolve_internal_conversation_url(
                sandbox_service, app_conversation, conversation_id
            )

            title = await _poll_for_title(
                httpx_client,
                app_conversation_url,
                app_conversation.session_api_key,
            )

            if not title:
                # Keep the callback active so later message events can retry.
                _logger.info(
                    f'Conversation {conversation_id} title not available yet; '
                    'will retry on a future message event.'
                )
                return None

            # Save the conversation info
            info = AppConversationInfo(
                **{
                    name: getattr(app_conversation, name)
                    for name in AppConversationInfo.model_fields
                }
            )
            info.title = title
            await app_conversation_info_service.save_app_conversation_info(info)

            # Disable callback - we have already set the status
            callback.status = EventCallbackStatus.DISABLED
            await event_callback_service.save_event_callback(callback)

        return EventCallbackResult(
            status=EventCallbackResultStatus.SUCCESS,
            event_callback_id=callback.id,
            event_id=event.id,
            conversation_id=conversation_id,
        )
