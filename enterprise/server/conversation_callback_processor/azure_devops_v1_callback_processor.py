"""Azure DevOps V1 callback processor for handling conversation events."""

import logging
from typing import Any
from uuid import UUID

import httpx
from integrations.utils import CONVERSATION_URL, get_summary_instruction
from pydantic import Field, SecretStr
from server.auth.token_manager import TokenManager

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
from openhands.integrations.azure_devops.azure_devops_service import AzureDevOpsService
from openhands.sdk import Event
from openhands.sdk.event import ConversationStateUpdateEvent

_logger = logging.getLogger(__name__)

# Module-level token manager instance
_token_manager = TokenManager()


class AzureDevOpsV1CallbackProcessor(EventCallbackProcessor):
    """Callback processor for Azure DevOps V1 integrations."""

    azure_devops_view_data: dict[str, Any] = Field(default_factory=dict)
    should_request_summary: bool = Field(default=True)
    is_pr_comment: bool = Field(default=False)
    thread_id: int | None = Field(default=None)

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        """Process events for Azure DevOps V1 integration."""

        # Only handle ConversationStateUpdateEvent
        if not isinstance(event, ConversationStateUpdateEvent):
            return None

        # Only act when execution has finished
        if not (event.key == 'execution_status' and event.value == 'finished'):
            return None

        _logger.info('[Azure DevOps V1] Callback agent state was %s', event)
        _logger.info(
            '[Azure DevOps V1] Should request summary: %s', self.should_request_summary
        )

        if not self.should_request_summary:
            return None

        self.should_request_summary = False

        try:
            _logger.info(f'[Azure DevOps V1] Requesting summary {conversation_id}')
            summary = await self._request_summary(conversation_id)
            _logger.info(
                f'[Azure DevOps V1] Posting summary {conversation_id}',
                extra={'summary': summary},
            )
            await self._post_summary_to_azure_devops(summary)

            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=summary,
            )
        except Exception as e:
            _logger.exception('[Azure DevOps V1] Error processing callback: %s', e)

            # Try to post error to Azure DevOps
            try:
                await self._post_summary_to_azure_devops(
                    f'OpenHands encountered an error: **{str(e)}**.\n\n'
                    f'[See the conversation]({CONVERSATION_URL.format(conversation_id)}) '
                    'for more information.'
                )
            except Exception as post_error:
                _logger.warning(
                    '[Azure DevOps V1] Failed to post error message to Azure DevOps: %s',
                    post_error,
                )

            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=str(e),
            )

    # -------------------------------------------------------------------------
    # Azure DevOps helpers
    # -------------------------------------------------------------------------

    async def _get_service_principal_token(self) -> str:
        """Get the Azure DevOps service principal token."""
        try:
            token = await _token_manager.get_azure_devops_service_principal_token()
            if not token:
                raise ValueError('Azure DevOps service principal token not configured')
            return token
        except Exception as e:
            _logger.error(
                '[Azure DevOps V1] Failed to get service principal token: %s', e
            )
            raise

    async def _post_summary_to_azure_devops(self, summary: str) -> None:
        """Post a summary comment to the configured Azure DevOps work item or PR."""
        sp_token = await self._get_service_principal_token()

        full_repo_name = self.azure_devops_view_data.get('full_repo_name')
        if not full_repo_name:
            raise ValueError('Missing full_repo_name in azure_devops_view_data')

        service = AzureDevOpsService(token=SecretStr(sp_token))

        if self.is_pr_comment:
            # Post to PR
            pr_id = self.azure_devops_view_data.get('pr_id')
            if not pr_id:
                raise ValueError('Missing pr_id in azure_devops_view_data')

            if self.thread_id:
                # Reply in the original thread
                await service.add_pr_comment_to_thread(
                    full_repo_name, pr_id, self.thread_id, summary
                )
            else:
                # Create new thread for general discussion
                await service.add_pr_thread(full_repo_name, pr_id, summary)

            _logger.info(
                f'[Azure DevOps V1] Posted summary to PR #{pr_id} in {full_repo_name}'
            )
        else:
            # Post to work item
            work_item_id = self.azure_devops_view_data.get('work_item_id')
            if not work_item_id:
                raise ValueError('Missing work_item_id in azure_devops_view_data')

            await service.add_work_item_comment(full_repo_name, work_item_id, summary)

            _logger.info(
                f'[Azure DevOps V1] Posted summary to work item #{work_item_id} in {full_repo_name}'
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
                '[Azure DevOps V1] HTTP error sending message to %s: %s. '
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
                '[Azure DevOps V1] %s. Request payload: %s',
                error_detail,
                payload,
                exc_info=True,
            )
            raise Exception(error_detail)

        except httpx.RequestError as e:
            error_detail = f'Request error to {url}: {str(e)}'
            _logger.error(
                '[Azure DevOps V1] %s. Request payload: %s',
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

        NOTE: This method returns a string (the agent server's response text)
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
