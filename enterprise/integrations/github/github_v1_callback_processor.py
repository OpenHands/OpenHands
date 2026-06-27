import logging
from typing import Any, ClassVar
from uuid import UUID

import httpx
from github import Auth, Github, GithubException, GithubIntegration
from integrations.v1_utils import handle_callback_error
from pydantic import Field
from server.auth.constants import GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY

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


class GithubV1CallbackProcessor(EventCallbackProcessor):
    """Callback processor for GitHub V1 integrations."""

    event_kind: ClassVar[EventKind] = 'ConversationStateUpdateEvent'

    github_view_data: dict[str, Any] = Field(default_factory=dict)
    should_request_summary: bool = Field(default=True)
    inline_pr_comment: bool = Field(default=False)

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        """Process events for GitHub V1 integration."""
        # Only handle ConversationStateUpdateEvent for execution_status
        if not isinstance(event, ConversationStateUpdateEvent):
            return None

        if event.key != 'execution_status':
            return None

        # Log ALL terminal states for monitoring (finished, error, stuck)
        _logger.info('[GitHub V1] Callback agent state was %s', event)

        # Only post the final response when execution has finished successfully
        if event.value != 'finished':
            return None

        _logger.info(
            '[GitHub V1] Should post final response: %s', self.should_request_summary
        )

        if not self.should_request_summary:
            return None

        self.should_request_summary = False

        try:
            _logger.info(f'[GitHub V1] Fetching final response {conversation_id}')
            final_response = await self._request_final_response(conversation_id)
            _logger.info(
                f'[GitHub V1] Posting final response {conversation_id}',
                extra={'final_response': final_response},
            )
            await self._post_summary_to_github(final_response)

            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=final_response,
            )
        except Exception as e:
            # Check if we have installation ID and credentials before posting
            can_post_error = bool(
                self.github_view_data.get('installation_id')
                and GITHUB_APP_CLIENT_ID
                and GITHUB_APP_PRIVATE_KEY
            )
            await handle_callback_error(
                error=e,
                conversation_id=conversation_id,
                service_name='GitHub',
                service_logger=_logger,
                can_post_error=can_post_error,
                post_error_func=self._post_summary_to_github,
            )

            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=str(e),
            )

    # -------------------------------------------------------------------------
    # GitHub helpers
    # -------------------------------------------------------------------------

    def _get_installation_access_token(self) -> str:
        installation_id = self.github_view_data.get('installation_id')

        if not installation_id:
            raise ValueError(
                f'Missing installation ID for GitHub payload: {self.github_view_data}'
            )

        if not GITHUB_APP_CLIENT_ID or not GITHUB_APP_PRIVATE_KEY:
            raise ValueError('GitHub App credentials are not configured')

        github_integration = GithubIntegration(
            auth=Auth.AppAuth(GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY),
        )
        token_data = github_integration.get_access_token(installation_id)
        return token_data.token

    async def _post_summary_to_github(self, summary: str) -> None:
        """Post a resolver response comment to the configured GitHub issue."""
        installation_token = self._get_installation_access_token()

        if not installation_token:
            raise RuntimeError('Missing GitHub credentials')

        full_repo_name = self.github_view_data['full_repo_name']
        issue_number = self.github_view_data['issue_number']

        try:
            if self.inline_pr_comment:
                with Github(auth=Auth.Token(installation_token)) as github_client:
                    repo = github_client.get_repo(full_repo_name)
                    pr = repo.get_pull(issue_number)
                    pr.create_review_comment_reply(
                        comment_id=self.github_view_data.get('comment_id', ''),
                        body=summary,
                    )
                return

            with Github(auth=Auth.Token(installation_token)) as github_client:
                repo = github_client.get_repo(full_repo_name)
                issue = repo.get_issue(number=issue_number)
                issue.create_comment(summary)
        except GithubException as e:
            if e.status == 410:
                _logger.info(
                    '[GitHub V1] Issue/PR %s#%s was deleted, skipping summary post',
                    full_repo_name,
                    issue_number,
                )
            else:
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
                '[GitHub V1] HTTP error fetching final response from %s: %s. '
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
            _logger.error('[GitHub V1] %s', error_detail, exc_info=True)
            raise Exception(error_detail)

        except httpx.RequestError as e:
            error_detail = f'Request error to {url}: {str(e)}'
            _logger.error('[GitHub V1] %s', error_detail, exc_info=True)
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

            return await self._get_final_response(
                httpx_client=httpx_client,
                agent_server_url=get_agent_server_url_from_sandbox(sandbox),
                conversation_id=conversation_id,
                session_api_key=sandbox.session_api_key,
            )
