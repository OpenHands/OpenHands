import logging
from uuid import UUID

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
    EventCallbackStatus,
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

_logger = logging.getLogger(__name__)


class SetTitleCallbackProcessor(EventCallbackProcessor):
    """Callback processor which sets conversation titles."""

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
        )

        _logger.info(f'Callback {callback.id} Invoked for event {event}')

        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)
        async with (
            get_event_callback_service(state) as event_callback_service,
            get_app_conversation_service(state) as app_conversation_service,
            get_app_conversation_info_service(state) as app_conversation_info_service,
            get_httpx_client(state) as httpx_client,
        ):
            # Generate a title for the conversation
            app_conversation = await app_conversation_service.get_app_conversation(
                conversation_id
            )
            assert app_conversation is not None
            app_conversation_url = app_conversation.conversation_url
            assert app_conversation_url is not None
            app_conversation_url = replace_localhost_hostname_for_docker(
                app_conversation_url
            )
            response = await httpx_client.post(
                f'{app_conversation_url}/generate_title',
                headers={
                    'X-Session-API-Key': app_conversation.session_api_key,
                },
                content='{}',
            )
            response.raise_for_status()
            title = response.json()['title']

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
