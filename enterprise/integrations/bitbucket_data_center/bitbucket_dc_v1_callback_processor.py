import logging
from typing import Any
from uuid import UUID

import httpx
from integrations.utils import get_summary_instruction
from integrations.v1_utils import handle_callback_error
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
from openhands.sdk import Event
from openhands.sdk.event import ConversationStateUpdateEvent

_logger = logging.getLogger(__name__)


class BitbucketDCV1CallbackProcessor(EventCallbackProcessor):
    """V1 callback processor for the Bitbucket Data Center resolver.

    Posts a summary comment back to the originating PR when the agent
    finishes successfully. Mirrors :class:`BitbucketV1CallbackProcessor`.
    """

    bitbucket_dc_view_data: dict[str, Any] = Field(default_factory=dict)
    should_request_summary: bool = Field(default=True)

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        if not isinstance(event, ConversationStateUpdateEvent):
            return None
        if event.key != 'execution_status':
            return None

        _logger.info('[Bitbucket DC V1] Callback agent state was %s', event)

        if event.value != 'finished':
            return None
        if not self.should_request_summary:
            return None

        self.should_request_summary = False
        try:
            summary = await self._request_summary(conversation_id)
            await self._post_summary_to_bitbucket_dc(summary)
            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=summary,
            )
        except Exception as e:
            can_post_error = bool(self.bitbucket_dc_view_data.get('keycloak_user_id'))
            await handle_callback_error(
                error=e,
                conversation_id=conversation_id,
                service_name='Bitbucket DC',
                service_logger=_logger,
                can_post_error=can_post_error,
                post_error_func=self._post_summary_to_bitbucket_dc,
            )
            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=str(e),
            )

    async def _post_summary_to_bitbucket_dc(self, summary: str) -> None:
        from integrations.bitbucket_data_center.bitbucket_dc_service import (
            SaaSBitbucketDCService,
        )

        keycloak_user_id = self.bitbucket_dc_view_data.get('keycloak_user_id')
        if not keycloak_user_id:
            raise RuntimeError('Missing keycloak user ID for Bitbucket DC')

        bitbucket_service = SaaSBitbucketDCService(external_auth_id=keycloak_user_id)
        await bitbucket_service.reply_to_pr_comment(
            owner=self.bitbucket_dc_view_data['project_key'],
            repo_slug=self.bitbucket_dc_view_data['repo_slug'],
            pr_id=self.bitbucket_dc_view_data['pr_id'],
            body=summary,
            parent_comment_id=self.bitbucket_dc_view_data.get('parent_comment_id'),
        )

    async def _ask_question(
        self,
        httpx_client: httpx.AsyncClient,
        agent_server_url: str,
        conversation_id: UUID,
        session_api_key: str,
        message_content: str,
    ) -> str:
        send_message_request = AskAgentRequest(question=message_content)
        url = (
            f"{agent_server_url.rstrip('/')}"
            f'/api/conversations/{conversation_id}/ask_agent'
        )
        headers = {'X-Session-API-Key': session_api_key}
        payload = send_message_request.model_dump()

        try:
            response = await httpx_client.post(
                url, json=payload, headers=headers, timeout=30.0
            )
            response.raise_for_status()
            return AskAgentResponse.model_validate(response.json()).response
        except httpx.HTTPStatusError as e:
            error_detail = f'HTTP {e.response.status_code} error'
            try:
                if e.response.text:
                    error_detail += f': {e.response.text}'
            except Exception:  # noqa: BLE001
                pass
            _logger.error(
                '[Bitbucket DC V1] HTTP error: %s', error_detail, exc_info=True
            )
            raise Exception(f'Failed to send message to agent server: {error_detail}')
        except httpx.TimeoutException:
            raise Exception(f'Request timeout after 30 seconds to {url}')
        except httpx.RequestError as e:
            raise Exception(f'Request error to {url}: {e}')

    async def _request_summary(self, conversation_id: UUID) -> str:
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
            get_app_conversation_info_service(state) as info_service,
            get_sandbox_service(state) as sandbox_service,
            get_httpx_client(state) as httpx_client,
        ):
            app_info = ensure_conversation_found(
                await info_service.get_app_conversation_info(conversation_id),
                conversation_id,
            )
            sandbox = ensure_running_sandbox(
                await sandbox_service.get_sandbox(app_info.sandbox_id),
                app_info.sandbox_id,
            )
            assert sandbox.session_api_key is not None
            agent_server_url = get_agent_server_url_from_sandbox(sandbox)
            return await self._ask_question(
                httpx_client=httpx_client,
                agent_server_url=agent_server_url,
                conversation_id=conversation_id,
                session_api_key=sandbox.session_api_key,
                message_content=get_summary_instruction(),
            )
