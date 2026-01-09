import logging
from uuid import UUID

import httpx
from integrations.utils import CONVERSATION_URL, get_summary_instruction
from pydantic import Field
from slack_sdk import WebClient
from storage.slack_team_store import SlackTeamStore

from openhands.agent_server.models import AskAgentRequest, AskAgentResponse
from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.app_server.event_callback.util import (
    ensure_conversation_found,
    ensure_running_sandbox,
    get_agent_server_url_from_sandbox,
)
from openhands.sdk import Event
from openhands.sdk.event import ConversationStateUpdateEvent

_logger = logging.getLogger(__name__)


class SlackV1CallbackProcessor(EventCallbackProcessor):
    """Callback processor for Slack V1 integrations."""

    slack_view_data: dict[str, str | None] = Field(default_factory=dict)

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        """Process events for Slack V1 integration."""

        # Only handle ConversationStateUpdateEvent
        if not isinstance(event, ConversationStateUpdateEvent):
            return None

        # Only act when execution has finished
        if not (event.key == 'execution_status' and event.value == 'finished'):
            return None

        _logger.info('[Slack V1] Callback agent state was %s', event)

        try:
            summary = await self._request_summary(conversation_id)
            await self._post_summary_to_slack(summary)

            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=summary,
            )
        except Exception as e:
            _logger.exception('[Slack V1] Error processing callback: %s', e)

            # Only try to post error to Slack if we have basic requirements
            try:
                await self._post_summary_to_slack(
                    f'OpenHands encountered an error: **{str(e)}**.\n\n'
                    f'[See the conversation]({CONVERSATION_URL.format(conversation_id)})'
                    'for more information.'
                )
            except Exception as post_error:
                _logger.warning(
                    '[Slack V1] Failed to post error message to Slack: %s', post_error
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

    def _get_bot_access_token(self):
        slack_team_store = SlackTeamStore.get_instance()
        bot_access_token = slack_team_store.get_team_bot_token(
            self.slack_view_data['team_id']
        )

        return bot_access_token

    async def _post_summary_to_slack(self, summary: str) -> None:
        """Post a summary message to the configured Slack channel."""
        bot_access_token = self._get_bot_access_token()
        if not bot_access_token:
            raise RuntimeError('Missing Slack bot access token')

        channel_id = self.slack_view_data['channel_id']
        thread_ts = self.slack_view_data.get('thread_ts') or self.slack_view_data.get(
            'message_ts'
        )

        client = WebClient(token=bot_access_token)

        try:
            # Post the summary as a threaded reply
            response = client.chat_postMessage(
                channel=channel_id,
                text=summary,
                thread_ts=thread_ts,
                unfurl_links=False,
                unfurl_media=False,
            )

            if not response['ok']:
                raise RuntimeError(
                    f"Slack API error: {response.get('error', 'Unknown error')}"
                )

            _logger.info(
                '[Slack V1] Successfully posted summary to channel %s', channel_id
            )

        except Exception as e:
            _logger.error('[Slack V1] Failed to post message to Slack: %s', e)
            raise

    # -------------------------------------------------------------------------
    # Agent / sandbox helpers
    # -------------------------------------------------------------------------

    async def _ask_question(
        self,
        httpx_client: httpx.AsyncClient,
        agent_server_url: str,
        conversation_id: UUID,
        session_api_key: str,
        message_content: str,
    ) -> str:
        """Send a message to the agent server via the V1 API and return response text."""
        send_message_request = AskAgentRequest(question=message_content)

        url = (
            f'{agent_server_url.rstrip("/")}'
            f'/api/conversations/{conversation_id}/ask_agent'
        )
        headers = {'X-Session-API-Key': session_api_key}
        payload = send_message_request.model_dump()

        try:
            response = await httpx_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            agent_response = AskAgentResponse.model_validate(response.json())
            return agent_response.response

        except httpx.HTTPStatusError as e:
            error_detail = f'HTTP {e.response.status_code} error'
            try:
                error_body = e.response.text
                if error_body:
                    error_detail += f': {error_body}'
            except Exception:  # noqa: BLE001
                pass

            _logger.error(
                '[Slack V1] HTTP error sending message to %s: %s. '
                'Request payload: %s. Response headers: %s',
                url,
                error_detail,
                payload,
                dict(e.response.headers),
                exc_info=True,
            )
            raise Exception(f'Failed to send message to agent server: {error_detail}')

        except httpx.TimeoutException:
            error_detail = f'Request timeout after 30 seconds to {url}'
            _logger.error(
                '[Slack V1] %s. Request payload: %s',
                error_detail,
                payload,
                exc_info=True,
            )
            raise Exception(error_detail)

        except httpx.RequestError as e:
            error_detail = f'Request error to {url}: {str(e)}'
            _logger.error(
                '[Slack V1] %s. Request payload: %s',
                error_detail,
                payload,
                exc_info=True,
            )
            raise Exception(error_detail)

    # -------------------------------------------------------------------------
    # Summary orchestration
    # -------------------------------------------------------------------------

    async def _request_summary(self, conversation_id: UUID) -> str:
        """
        Ask the agent to produce a summary of its work and return the agent response.

        NOTE: This method now returns a string (the agent server's response text)
        and raises exceptions on errors. The wrapping into EventCallbackResult
        is handled by __call__.
        """
        # Import services within the method to avoid circular imports
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_httpx_client,
            get_sandbox_service,
        )
        from openhands.app_server.services.injector import InjectorState
        from openhands.app_server.user.specifiy_user_context import (
            ADMIN,
            USER_CONTEXT_ATTR,
        )

        # Create injector state for dependency injection
        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)

        async with (
            get_app_conversation_info_service(state) as app_conversation_info_service,
            get_sandbox_service(state) as sandbox_service,
            get_httpx_client(state) as httpx_client,
        ):
            # 1. Conversation lookup
            app_conversation_info = ensure_conversation_found(
                await app_conversation_info_service.get_app_conversation_info(
                    conversation_id
                ),
                conversation_id,
            )

            # 2. Sandbox lookup + validation
            sandbox = ensure_running_sandbox(
                await sandbox_service.get_sandbox(app_conversation_info.sandbox_id),
                app_conversation_info.sandbox_id,
            )

            assert (
                sandbox.session_api_key is not None
            ), f'No session API key for sandbox: {sandbox.id}'

            # 3. URL + instruction
            agent_server_url = get_agent_server_url_from_sandbox(sandbox)

            # Prepare message based on agent state
            message_content = get_summary_instruction()

            # Ask the agent and return the response text
            return await self._ask_question(
                httpx_client=httpx_client,
                agent_server_url=agent_server_url,
                conversation_id=conversation_id,
                session_api_key=sandbox.session_api_key,
                message_content=message_content,
            )
