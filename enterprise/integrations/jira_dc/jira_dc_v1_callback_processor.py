"""Jira Data Center V1 callback processor.

This processor handles events from V1 conversations and posts
summaries back to Jira DC issues when the agent finishes work.
"""

import logging
from uuid import UUID

import httpx
from integrations.utils import get_summary_instruction, markdown_to_jira_markup
from pydantic import Field

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
from openhands.app_server.utils.http_session import httpx_verify_option
from openhands.sdk import Event
from openhands.sdk.event import ConversationStateUpdateEvent

_logger = logging.getLogger(__name__)


class JiraDcV1CallbackProcessor(EventCallbackProcessor):
    """Callback processor for Jira Data Center V1 integrations."""

    should_request_summary: bool = Field(default=True)
    issue_key: str
    workspace_name: str
    base_api_url: str
    svc_acc_api_key: str  # Decrypted API key

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        """Process events for Jira DC V1 integration."""
        # Only handle ConversationStateUpdateEvent for execution_status
        if not isinstance(event, ConversationStateUpdateEvent):
            return None

        if event.key != 'execution_status':
            return None

        _logger.info('[Jira DC] Callback agent state was %s', event)

        # Only request summary when execution has finished successfully
        if event.value != 'finished':
            return None

        _logger.info(
            '[Jira DC] Should request summary: %s', self.should_request_summary
        )

        if not self.should_request_summary:
            return None

        self.should_request_summary = False

        try:
            _logger.info(f'[Jira DC] Requesting summary {conversation_id}')
            summary = await self._request_summary(conversation_id)
            _logger.info(
                f'[Jira DC] Posting summary {conversation_id}',
                extra={'summary': summary},
            )
            await self._post_summary_to_jira_dc(summary)

            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=summary,
            )
        except Exception as e:
            _logger.exception(f'[Jira DC] Failed to post summary: {e}', stack_info=True)
            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=str(e),
            )

    async def _request_summary(self, conversation_id: UUID) -> str:
        """Ask the agent to produce a summary of its work and return the agent response."""
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
            f"{agent_server_url.rstrip('/')}"
            f"/api/conversations/{conversation_id}/ask_agent"
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
            except Exception:
                pass

            _logger.exception(
                '[Jira DC] HTTP error sending message to %s: %s. '
                'Request payload: %s. Response headers: %s',
                url,
                error_detail,
                payload,
                dict(e.response.headers),
                stack_info=True,
            )
            raise Exception(f'Failed to send message to agent server: {error_detail}')

        except httpx.TimeoutException:
            error_detail = f'Request timeout after 30 seconds to {url}'
            _logger.exception(
                '[Jira DC] Timeout error: %s. Request payload: %s',
                error_detail,
                payload,
                stack_info=True,
            )
            raise Exception(f'Failed to send message to agent server: {error_detail}')

    async def _post_summary_to_jira_dc(self, summary: str):
        """Post the summary back to the Jira DC issue."""
        if not all(
            [
                self.svc_acc_api_key,
                self.issue_key,
                self.base_api_url,
            ]
        ):
            _logger.warning('[Jira DC] Missing required data for posting summary')
            return

        # Add a comment to the Jira DC issue with the summary
        comment_url = f'{self.base_api_url}/rest/api/2/issue/{self.issue_key}/comment'

        message = f'OpenHands resolved this issue:\n\n{summary}'
        # Convert standard Markdown to Jira Wiki Markup for proper rendering
        comment_body = {'body': markdown_to_jira_markup(message)}

        headers = {'Authorization': f'Bearer {self.svc_acc_api_key}'}

        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(
                comment_url,
                headers=headers,
                json=comment_body,
            )
            response.raise_for_status()
            _logger.info(f'[Jira DC] Posted summary to {self.issue_key}')
