import logging
from typing import ClassVar
from uuid import UUID

import httpx
from integrations.v1_utils import handle_callback_error
from pydantic import Field
from slack_sdk import WebClient
from storage.slack_team_store import SlackTeamStore

from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
    EventKind,
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

        # Only post the final response when execution has finished successfully
        if event.value != 'finished':
            return None

        try:
            _logger.info(f'[Slack V1] Fetching final response {conversation_id}')
            final_response = await self._request_final_response(conversation_id)
            _logger.info(
                f'[Slack V1] Posting final response {conversation_id}',
                extra={'final_response': final_response},
            )
            await self._post_summary_to_slack(final_response)

            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=final_response,
            )
        except Exception as e:
            await handle_callback_error(
                error=e,
                conversation_id=conversation_id,
                service_name='Slack',
                service_logger=_logger,
                can_post_error=True,  # Slack always attempts to post errors
                post_error_func=self._post_summary_to_slack,
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

    async def _post_summary_to_slack(self, summary: str) -> None:
        """Post a resolver response message to the configured Slack channel."""
        bot_access_token = await self._get_bot_access_token()
        if not bot_access_token:
            raise RuntimeError('Missing Slack bot access token')

        channel_id = self.slack_view_data['channel_id']
        thread_ts = self.slack_view_data.get('thread_ts') or self.slack_view_data.get(
            'message_ts'
        )

        client = WebClient(token=bot_access_token)

        try:
            # Post the response as a threaded reply.
            # Use markdown_text instead of text to properly render standard Markdown
            # (e.g., **bold**, [link](url)) which is used throughout the codebase.
            response = client.chat_postMessage(
                channel=channel_id,
                markdown_text=summary,
                thread_ts=thread_ts,
                unfurl_links=False,
                unfurl_media=False,
            )

            if not response['ok']:
                raise RuntimeError(
                    f'Slack API error: {response.get("error", "Unknown error")}'
                )

            _logger.info(
                '[Slack V1] Successfully posted final response to channel %s',
                channel_id,
            )

        except Exception as e:
            _logger.error('[Slack V1] Failed to post message to Slack: %s', e)
            raise

    # -------------------------------------------------------------------------
    # Agent / sandbox helpers
    # -------------------------------------------------------------------------

    async def _get_final_response(
        self,
        httpx_client: httpx.AsyncClient,
        agent_server_url: str,
        conversation_id: UUID,
        session_api_key: str,
    ) -> str:
        """Fetch the agent's final response from the V1 API."""
        url = (
            f'{agent_server_url.rstrip("/")}'
            f'/api/conversations/{conversation_id}/agent_final_response'
        )
        headers = {'X-Session-API-Key': session_api_key}

        try:
            response = await httpx_client.get(
                url,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError('Invalid agent final response payload')

            final_response = str(payload.get('response') or '').strip()
            if not final_response:
                raise RuntimeError('Agent final response was empty')
            return final_response

        except httpx.HTTPStatusError as e:
            error_detail = f'HTTP {e.response.status_code} error'
            try:
                error_body = e.response.text
                if error_body:
                    error_detail += f': {error_body}'
            except Exception:  # noqa: BLE001
                pass

            _logger.error(
                '[Slack V1] HTTP error fetching final response from %s: %s. '
                'Response headers: %s',
                url,
                error_detail,
                dict(e.response.headers),
                exc_info=True,
            )
            raise Exception(
                f'Failed to fetch final response from agent server: {error_detail}'
            )

        except httpx.TimeoutException:
            error_detail = f'Request timeout after 30 seconds to {url}'
            _logger.error('[Slack V1] %s', error_detail, exc_info=True)
            raise Exception(error_detail)

        except httpx.RequestError as e:
            error_detail = f'Request error to {url}: {str(e)}'
            _logger.error('[Slack V1] %s', error_detail, exc_info=True)
            raise Exception(error_detail)

    # -------------------------------------------------------------------------
    # Final response orchestration
    # -------------------------------------------------------------------------

    async def _request_final_response(self, conversation_id: UUID) -> str:
        """Return the agent's final response without asking the LLM to summarize."""
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

            assert sandbox.session_api_key is not None, (
                f'No session API key for sandbox: {sandbox.id}'
            )

            # 3. URL
            agent_server_url = get_agent_server_url_from_sandbox(sandbox)

            return await self._get_final_response(
                httpx_client=httpx_client,
                agent_server_url=agent_server_url,
                conversation_id=conversation_id,
                session_api_key=sandbox.session_api_key,
            )
