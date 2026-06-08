import logging
from collections.abc import Sequence
from typing import ClassVar
from uuid import UUID

from integrations.utils import CONVERSATION_URL
from integrations.v1_utils import handle_callback_error
from pydantic import Field
from slack_sdk import WebClient
from storage.slack_team_store import SlackTeamStore

from openhands.agent_server.models import EventSortOrder
from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
    EventKind,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.sdk import Event, MessageEvent, TextContent
from openhands.sdk.event import ConversationStateUpdateEvent

_logger = logging.getLogger(__name__)

_FINAL_MESSAGE_SEARCH_LIMIT = 50


class SlackV1CallbackProcessor(EventCallbackProcessor):
    """Callback processor for Slack V1 integrations."""

    event_kind: ClassVar[EventKind] = 'ConversationStateUpdateEvent'

    slack_view_data: dict[str, str | None] = Field(default_factory=dict)

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        """Process events for Slack V1 integration."""
        # Only handle ConversationStateUpdateEvent for execution_status
        if not isinstance(event, ConversationStateUpdateEvent):
            return None

        if event.key != 'execution_status':
            return None

        # Log ALL terminal states for monitoring (finished, error, stuck)
        _logger.info('[Slack V1] Callback agent state was %s', event)

        # Only post the final assistant message when execution has finished successfully
        if event.value != 'finished':
            return None

        try:
            message = await self._get_final_assistant_message(conversation_id)
            await self._post_message_to_slack(message)

            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=message,
            )
        except Exception as e:
            await handle_callback_error(
                error=e,
                conversation_id=conversation_id,
                service_name='Slack',
                service_logger=_logger,
                can_post_error=True,  # Slack always attempts to post errors
                post_error_func=self._post_message_to_slack,
            )

            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=str(e),
            )

    # -------------------------------------------------------------------------
    # Slack helpers
    # -------------------------------------------------------------------------

    async def _get_bot_access_token(self) -> str | None:
        team_id = self.slack_view_data.get('team_id')
        if team_id is None:
            return None
        slack_team_store = SlackTeamStore.get_instance()
        bot_access_token = await slack_team_store.get_team_bot_token(team_id)

        return bot_access_token

    async def _post_message_to_slack(self, message: str) -> None:
        """Post a message to the configured Slack channel."""
        bot_access_token = await self._get_bot_access_token()
        if not bot_access_token:
            raise RuntimeError('Missing Slack bot access token')

        channel_id = self.slack_view_data['channel_id']
        thread_ts = self.slack_view_data.get('thread_ts') or self.slack_view_data.get(
            'message_ts'
        )

        client = WebClient(token=bot_access_token)

        try:
            # Post the message as a threaded reply
            # Use markdown_text instead of text to properly render standard Markdown
            # (e.g., **bold**, [link](url)) which is used throughout the codebase
            response = client.chat_postMessage(
                channel=channel_id,
                markdown_text=message,
                thread_ts=thread_ts,
                unfurl_links=False,
                unfurl_media=False,
            )

            if not response['ok']:
                raise RuntimeError(
                    f"Slack API error: {response.get('error', 'Unknown error')}"
                )

            _logger.info(
                '[Slack V1] Successfully posted message to channel %s', channel_id
            )

        except Exception as e:
            _logger.error('[Slack V1] Failed to post message to Slack: %s', e)
            raise

    # -------------------------------------------------------------------------
    # Final message lookup
    # -------------------------------------------------------------------------

    async def _get_final_assistant_message(self, conversation_id: UUID) -> str:
        """Return the latest assistant message for a completed Slack run."""
        # Import services within the method to avoid circular imports
        from openhands.app_server.config import get_event_service
        from openhands.app_server.services.injector import InjectorState
        from openhands.app_server.user.specifiy_user_context import (
            ADMIN,
            USER_CONTEXT_ATTR,
        )

        # Create injector state for dependency injection
        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)

        async with get_event_service(state) as event_service:
            # Finished callbacks are emitted immediately after a run completes, so the
            # final agent reply should be among the newest MessageEvents. Keep this
            # bounded to avoid an expensive callback on very long conversations.
            page = await event_service.search_events(
                conversation_id,
                kind__eq='MessageEvent',
                sort_order=EventSortOrder.TIMESTAMP_DESC,
                limit=_FINAL_MESSAGE_SEARCH_LIMIT,
            )

        # EventPage.items is a materialized list, so it is safe to select from it
        # after the event service context manager has closed.
        for event in page.items:
            if not self._is_agent_message_event(event):
                continue
            message = self._extract_message_text(event)
            if message:
                return message

        return self._get_no_final_message_fallback(conversation_id)

    def _is_agent_message_event(self, event: Event) -> bool:
        if not isinstance(event, MessageEvent):
            return False

        return event.source == 'agent'

    def _extract_message_text(self, event: MessageEvent) -> str:
        llm_message = getattr(event, 'llm_message', None)
        content = getattr(llm_message, 'content', None)
        if isinstance(content, str) or not isinstance(content, Sequence):
            _logger.debug(
                '[Slack V1] Skipping message event %s because content is not a '
                'sequence of content blocks',
                event.id,
            )
            return ''

        text_parts = []
        for content_part in content:
            if not isinstance(content_part, TextContent):
                continue
            text = content_part.text.strip()
            if text:
                text_parts.append(text)

        return '\n\n'.join(text_parts)

    def _get_no_final_message_fallback(self, conversation_id: UUID) -> str:
        return (
            'OpenHands finished the run, but no final assistant message was found. '
            f'[See the conversation]({CONVERSATION_URL.format(conversation_id)}) '
            'for more information.'
        )
