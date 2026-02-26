import logging
from typing import Any
from uuid import UUID

import httpx
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
from pydantic import Field

from integrations.utils import CONVERSATION_URL, get_summary_instruction
from storage.linear_integration_store import LinearIntegrationStore

_logger = logging.getLogger(__name__)


class LinearV1CallbackProcessor(EventCallbackProcessor):
    """Callback processor for Linear V1 integrations.

    Handles ``ConversationStateUpdateEvent`` events to post summaries
    back to Linear issues when conversations finish.
    """

    linear_view_data: dict[str, Any] = Field(default_factory=dict)
    should_request_summary: bool = Field(default=True)

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        """Process events for Linear V1 integration."""
        if not isinstance(event, ConversationStateUpdateEvent):
            return None

        if not (event.key == 'execution_status' and event.value == 'finished'):
            return None

        _logger.info('[Linear V1] Callback agent state was %s', event)
        _logger.info(
            '[Linear V1] Should request summary: %s', self.should_request_summary
        )

        if not self.should_request_summary:
            return None

        self.should_request_summary = False

        try:
            _logger.info(f'[Linear V1] Requesting summary {conversation_id}')
            summary = await self._request_summary(conversation_id)

            _logger.info(
                f'[Linear V1] Posting summary {conversation_id}',
                extra={'summary': summary},
            )
            await self._post_summary_to_linear(summary)

            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=str(event.id),
                conversation_id=conversation_id,
                detail=summary,
            )
        except Exception as e:
            _logger.exception('[Linear V1] Error processing callback: %s', e)

            try:
                await self._post_summary_to_linear(
                    f'OpenHands encountered an error: **{str(e)}**.\n\n'
                    f'[See the conversation]({CONVERSATION_URL.format(conversation_id)})'
                    ' for more information.'
                )
            except Exception as post_error:
                _logger.warning(
                    '[Linear V1] Failed to post error message to Linear: %s',
                    post_error,
                )

            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=str(event.id),
                conversation_id=conversation_id,
                detail=str(e),
            )

    # -------------------------------------------------------------------------
    # Linear helpers
    # -------------------------------------------------------------------------

    async def _post_summary_to_linear(self, summary: str) -> None:
        """Post a summary comment to the configured Linear issue."""
        from server.auth.token_manager import TokenManager

        issue_id = self.linear_view_data.get('issue_id')
        issue_key = self.linear_view_data.get('issue_key')
        workspace_name = self.linear_view_data.get('workspace_name')

        if not all([issue_id, workspace_name]):
            raise RuntimeError(
                'Missing required Linear view data '
                f'(issue_id={issue_id}, workspace_name={workspace_name})'
            )

        integration_store = LinearIntegrationStore.get_instance()
        workspace = await integration_store.get_workspace_by_name(workspace_name)
        if not workspace:
            raise RuntimeError(f'Workspace {workspace_name} not found')

        if workspace.status != 'active':
            raise RuntimeError(f'Workspace {workspace_name} is not active')

        token_manager = TokenManager()
        api_key = token_manager.decrypt_text(workspace.svc_acc_api_key)

        # Use Linear GraphQL API to create a comment
        graphql_url = 'https://api.linear.app/graphql'
        headers = {
            'Authorization': api_key,
            'Content-Type': 'application/json',
        }
        mutation = {
            'query': '''
                mutation CommentCreate($input: CommentCreateInput!) {
                    commentCreate(input: $input) {
                        success
                        comment { id }
                    }
                }
            ''',
            'variables': {
                'input': {
                    'issueId': issue_id,
                    'body': summary,
                }
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                graphql_url, headers=headers, json=mutation
            )
            response.raise_for_status()

        _logger.info(
            '[Linear V1] Successfully posted summary to issue %s', issue_key
        )

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
                '[Linear V1] HTTP error sending message to %s: %s. '
                'Request payload: %s. Response headers: %s',
                url,
                error_detail,
                payload,
                dict(e.response.headers),
                exc_info=True,
            )
            raise Exception(
                f'Failed to send message to agent server: {error_detail}'
            )

        except httpx.TimeoutException:
            error_detail = f'Request timeout after 30 seconds to {url}'
            _logger.error(
                '[Linear V1] %s. Request payload: %s',
                error_detail,
                payload,
                exc_info=True,
            )
            raise Exception(error_detail)

        except httpx.RequestError as e:
            error_detail = f'Request error to {url}: {str(e)}'
            _logger.error(
                '[Linear V1] %s. Request payload: %s',
                error_detail,
                payload,
                exc_info=True,
            )
            raise Exception(error_detail)

    # -------------------------------------------------------------------------
    # Summary orchestration
    # -------------------------------------------------------------------------

    async def _request_summary(self, conversation_id: UUID) -> str:
        """Ask the agent to produce a summary of its work and return the agent response."""
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

        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)

        async with (
            get_app_conversation_info_service(state) as app_conversation_info_service,
            get_sandbox_service(state) as sandbox_service,
            get_httpx_client(state) as httpx_client,
        ):
            app_conversation_info = ensure_conversation_found(
                await app_conversation_info_service.get_app_conversation_info(
                    conversation_id
                ),
                conversation_id,
            )

            sandbox = ensure_running_sandbox(
                await sandbox_service.get_sandbox(
                    app_conversation_info.sandbox_id
                ),
                app_conversation_info.sandbox_id,
            )

            assert sandbox.session_api_key is not None, (
                f'No session API key for sandbox: {sandbox.id}'
            )

            agent_server_url = get_agent_server_url_from_sandbox(sandbox)
            message_content = get_summary_instruction()

            return await self._ask_question(
                httpx_client=httpx_client,
                agent_server_url=agent_server_url,
                conversation_id=conversation_id,
                session_api_key=sandbox.session_api_key,
                message_content=message_content,
            )
