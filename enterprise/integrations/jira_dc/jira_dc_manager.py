import hashlib
import hmac
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
from fastapi import Request
from integrations.jira_dc.jira_dc_types import (
    JiraDcViewInterface,
)
from integrations.jira_dc.jira_dc_view import (
    JiraDcExistingConversationView,
    JiraDcFactory,
    JiraDcNewConversationView,
)
from integrations.manager import Manager
from integrations.models import JobContext, Message
from integrations.utils import (
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
    filter_potential_repos_by_user_msg,
)
from jinja2 import Environment, FileSystemLoader
from server.auth.saas_user_auth import get_user_auth_from_keycloak_id
from server.auth.token_manager import TokenManager
from server.utils.conversation_callback_utils import register_callback_processor
from storage.jira_dc_integration_store import JiraDcIntegrationStore
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import Repository
from openhands.server.shared import server_config
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.server.user_auth.user_auth import UserAuth
from openhands.utils.http_session import httpx_verify_option


class JiraDcManager(Manager):
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.integration_store = JiraDcIntegrationStore.get_instance()
        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'jira_dc')
        )

    async def authenticate_user(
        self, user_email: str, jira_dc_user_id: str, workspace_id: int
    ) -> tuple[JiraDcUser | None, UserAuth | None]:
        """Authenticate Jira DC user and get their OpenHands user auth."""

        if not jira_dc_user_id or jira_dc_user_id == 'none':
            # Get Keycloak user ID from email
            keycloak_user_id = await self.token_manager.get_user_id_from_user_email(
                user_email
            )
            if not keycloak_user_id:
                logger.warning(
                    f'[Jira DC] No Keycloak user found for email: {user_email}'
                )
                return None, None

            # Find active Jira DC user by Keycloak user ID and organization
            jira_dc_user = await self.integration_store.get_active_user_by_keycloak_id_and_workspace(
                keycloak_user_id, workspace_id
            )
        else:
            jira_dc_user = await self.integration_store.get_active_user(
                jira_dc_user_id, workspace_id
            )

        if not jira_dc_user:
            logger.warning(
                f'[Jira DC] No active Jira DC user found for {user_email} in workspace {workspace_id}'
            )
            return None, None

        saas_user_auth = await get_user_auth_from_keycloak_id(
            jira_dc_user.keycloak_user_id
        )
        return jira_dc_user, saas_user_auth

    async def _get_repositories(self, user_auth: UserAuth) -> list[Repository]:
        """Get repositories that the user has access to."""
        provider_tokens = await user_auth.get_provider_tokens()
        if provider_tokens is None:
            return []
        access_token = await user_auth.get_access_token()
        user_id = await user_auth.get_user_id()
        client = ProviderHandler(
            provider_tokens=provider_tokens,
            external_auth_token=access_token,
            external_auth_id=user_id,
        )
        repos: list[Repository] = await client.get_repositories(
            'pushed', server_config.app_mode, None, None, None, None
        )
        return repos

    async def validate_request(
        self, request: Request
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Verify Jira DC webhook signature."""
        signature_header = request.headers.get('x-hub-signature')
        signature = signature_header.split('=')[1] if signature_header else None
        body = await request.body()
        payload = await request.json()
        workspace_name = ''

        if payload.get('webhookEvent') == 'comment_created':
            selfUrl = payload.get('comment', {}).get('author', {}).get('self')
        elif payload.get('webhookEvent') == 'jira:issue_updated':
            selfUrl = payload.get('user', {}).get('self')
        else:
            workspace_name = ''

        parsedUrl = urlparse(selfUrl)
        if parsedUrl.hostname:
            workspace_name = parsedUrl.hostname

        if not workspace_name:
            logger.warning('[Jira DC] No workspace name found in webhook payload')
            return False, None, None

        if not signature:
            logger.warning('[Jira DC] No signature found in webhook headers')
            return False, None, None

        workspace = await self.integration_store.get_workspace_by_name(workspace_name)

        if not workspace:
            logger.warning('[Jira DC] Could not identify workspace for webhook')
            return False, None, None

        if workspace.status != 'active':
            logger.warning(f'[Jira DC] Workspace {workspace.id} is not active')
            return False, None, None

        webhook_secret = self.token_manager.decrypt_text(workspace.webhook_secret)
        digest = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()

        if hmac.compare_digest(signature, digest):
            logger.info('[Jira DC] Webhook signature verified successfully')
            return True, signature, payload

        return False, None, None

    def parse_webhook(self, payload: Dict) -> JobContext | None:
        event_type = payload.get('webhookEvent')

        if event_type == 'comment_created':
            comment_data = payload.get('comment', {})
            comment = comment_data.get('body', '')

            if '@openhands' not in comment:
                return None

            issue_data = payload.get('issue', {})
            issue_id = issue_data.get('id')
            issue_key = issue_data.get('key')
            base_api_url = issue_data.get('self', '').split('/rest/')[0]

            user_data = comment_data.get('author', {})
            user_email = user_data.get('emailAddress')
            display_name = user_data.get('displayName')
            user_key = user_data.get('key')
        elif event_type == 'jira:issue_updated':
            changelog = payload.get('changelog', {})
            items = changelog.get('items', [])
            labels = [
                item.get('toString', '')
                for item in items
                if item.get('field') == 'labels' and 'toString' in item
            ]

            if 'openhands' not in labels:
                return None

            issue_data = payload.get('issue', {})
            issue_id = issue_data.get('id')
            issue_key = issue_data.get('key')
            base_api_url = issue_data.get('self', '').split('/rest/')[0]

            user_data = payload.get('user', {})
            user_email = user_data.get('emailAddress')
            display_name = user_data.get('displayName')
            user_key = user_data.get('key')
            comment = ''
        else:
            return None

        workspace_name = ''

        parsedUrl = urlparse(base_api_url)
        if parsedUrl.hostname:
            workspace_name = parsedUrl.hostname

        if not all(
            [
                issue_id,
                issue_key,
                user_email,
                display_name,
                user_key,
                workspace_name,
                base_api_url,
            ]
        ):
            return None

        return JobContext(
            issue_id=issue_id,
            issue_key=issue_key,
            user_msg=comment,
            user_email=user_email,
            display_name=display_name,
            platform_user_id=user_key,
            workspace_name=workspace_name,
            base_api_url=base_api_url,
        )

    async def receive_message(self, message: Message):
        """Process incoming Jira DC webhook message."""

        payload = message.message.get('payload', {})
        job_context = self.parse_webhook(payload)

        if not job_context:
            logger.info('[Jira DC] Webhook does not match trigger conditions')
            return

        workspace = await self.integration_store.get_workspace_by_name(
            job_context.workspace_name
        )
        if not workspace:
            logger.warning(
                f'[Jira DC] No workspace found for email domain: {job_context.user_email}'
            )
            await self._send_error_comment(
                job_context,
                'Your workspace is not configured with Jira DC integration.',
                None,
            )
            return

        # Prevent any recursive triggers from the service account
        if job_context.user_email == workspace.svc_acc_email:
            return

        if workspace.status != 'active':
            logger.warning(f'[Jira DC] Workspace {workspace.id} is not active')
            await self._send_error_comment(
                job_context,
                'Jira DC integration is not active for your workspace.',
                workspace,
            )
            return

        # Authenticate user
        jira_dc_user, saas_user_auth = await self.authenticate_user(
            job_context.user_email, job_context.platform_user_id, workspace.id
        )
        if not jira_dc_user or not saas_user_auth:
            logger.warning(
                f'[Jira DC] User authentication failed for {job_context.user_email}'
            )
            await self._send_error_comment(
                job_context,
                f'User {job_context.user_email} is not authenticated or active in the Jira DC integration.',
                workspace,
            )
            return

        # Get issue details
        try:
            api_key = self.token_manager.decrypt_text(workspace.svc_acc_api_key)
            issue_title, issue_description = await self.get_issue_details(
                job_context, api_key
            )
            job_context.issue_title = issue_title
            job_context.issue_description = issue_description
        except Exception as e:
            logger.error(f'[Jira DC] Failed to get issue context: {str(e)}')
            await self._send_error_comment(
                job_context,
                'Failed to retrieve issue details. Please check the issue key and try again.',
                workspace,
            )
            return

        try:
            # Create Jira DC view
            jira_dc_view = await JiraDcFactory.create_jira_dc_view_from_payload(
                job_context,
                saas_user_auth,
                jira_dc_user,
                workspace,
            )
        except Exception as e:
            logger.error(
                f'[Jira DC] Failed to create jira dc view: {str(e)}', exc_info=True
            )
            await self._send_error_comment(
                job_context,
                'Failed to initialize conversation. Please try again.',
                workspace,
            )
            return

        if not await self.is_job_requested(message, jira_dc_view):
            return

        await self.start_job(jira_dc_view)

    async def is_job_requested(
        self, message: Message, jira_dc_view: JiraDcViewInterface
    ) -> bool:
        """
        Check if a job is requested and handle repository selection.
        """

        if isinstance(jira_dc_view, JiraDcExistingConversationView):
            return True

        try:
            # Get user repositories
            user_repos: list[Repository] = await self._get_repositories(
                jira_dc_view.saas_user_auth
            )

            target_str = f'{jira_dc_view.job_context.issue_description}\n{jira_dc_view.job_context.user_msg}'

            # Try to infer repository from issue description
            match, repos = filter_potential_repos_by_user_msg(target_str, user_repos)

            if match:
                # Found exact repository match
                jira_dc_view.selected_repo = repos[0].full_name
                logger.info(f'[Jira DC] Inferred repository: {repos[0].full_name}')
                return True
            else:
                # No clear match - send repository selection comment
                await self._send_repo_selection_comment(jira_dc_view)
                return False

        except Exception as e:
            logger.error(f'[Jira DC] Error in is_job_requested: {str(e)}')
            return False

    async def start_job(self, jira_dc_view: JiraDcViewInterface):
        """Start a Jira DC job/conversation."""
        # Import here to prevent circular import
        from server.conversation_callback_processor.jira_dc_callback_processor import (
            JiraDcCallbackProcessor,
        )

        try:
            user_info: JiraDcUser = jira_dc_view.jira_dc_user
            logger.info(
                f'[Jira DC] Starting job for user {user_info.keycloak_user_id} '
                f'issue {jira_dc_view.job_context.issue_key}',
            )

            # Create conversation
            conversation_id = await jira_dc_view.create_or_update_conversation(
                self.jinja_env
            )

            logger.info(
                f'[Jira DC] Created/Updated conversation {conversation_id} for issue {jira_dc_view.job_context.issue_key}'
            )

            if isinstance(jira_dc_view, JiraDcNewConversationView):
                # Register callback processor for updates
                processor = JiraDcCallbackProcessor(
                    issue_key=jira_dc_view.job_context.issue_key,
                    workspace_name=jira_dc_view.jira_dc_workspace.name,
                    base_api_url=jira_dc_view.job_context.base_api_url,
                )

                # Register the callback processor
                register_callback_processor(conversation_id, processor)

                logger.info(
                    f'[Jira DC] Created callback processor for conversation {conversation_id}'
                )

            # Send initial response
            msg_info = jira_dc_view.get_response_msg()

        except MissingSettingsError as e:
            logger.warning(f'[Jira DC] Missing settings error: {str(e)}')
            msg_info = f'Please re-login into [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except LLMAuthenticationError as e:
            logger.warning(f'[Jira DC] LLM authentication error: {str(e)}')
            msg_info = f'Please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except Exception as e:
            logger.error(
                f'[Jira DC] Unexpected error starting job: {str(e)}', exc_info=True
            )
            msg_info = 'Sorry, there was an unexpected error starting the job. Please try again.'

        # Send response comment
        try:
            api_key = self.token_manager.decrypt_text(
                jira_dc_view.jira_dc_workspace.svc_acc_api_key
            )
            await self.send_message(
                self.create_outgoing_message(msg=msg_info),
                issue_key=jira_dc_view.job_context.issue_key,
                base_api_url=jira_dc_view.job_context.base_api_url,
                svc_acc_api_key=api_key,
            )
        except Exception as e:
            logger.error(f'[Jira] Failed to send response message: {str(e)}')

    async def get_issue_details(
        self, job_context: JobContext, svc_acc_api_key: str
    ) -> Tuple[str, str]:
        """Get issue details from Jira DC API."""
        url = f'{job_context.base_api_url}/rest/api/2/issue/{job_context.issue_key}'
        headers = {'Authorization': f'Bearer {svc_acc_api_key}'}
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            issue_payload = response.json()

        if not issue_payload:
            raise ValueError(f'Issue with key {job_context.issue_key} not found.')

        title = issue_payload.get('fields', {}).get('summary', '')
        description = issue_payload.get('fields', {}).get('description', '')

        if not title:
            raise ValueError(
                f'Issue with key {job_context.issue_key} does not have a title.'
            )

        if not description:
            raise ValueError(
                f'Issue with key {job_context.issue_key} does not have a description.'
            )

        return title, description

    async def send_message(
        self, message: Message, issue_key: str, base_api_url: str, svc_acc_api_key: str
    ):
        """Send message/comment to Jira DC issue."""
        url = f'{base_api_url}/rest/api/2/issue/{issue_key}/comment'
        headers = {'Authorization': f'Bearer {svc_acc_api_key}'}
        data = {'body': message.message}
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()

    async def _send_error_comment(
        self,
        job_context: JobContext,
        error_msg: str,
        workspace: JiraDcWorkspace | None,
    ):
        """Send error comment to Jira DC issue."""
        if not workspace:
            logger.error('[Jira DC] Cannot send error comment - no workspace available')
            return

        try:
            api_key = self.token_manager.decrypt_text(workspace.svc_acc_api_key)
            await self.send_message(
                self.create_outgoing_message(msg=error_msg),
                issue_key=job_context.issue_key,
                base_api_url=job_context.base_api_url,
                svc_acc_api_key=api_key,
            )
        except Exception as e:
            logger.error(f'[Jira DC] Failed to send error comment: {str(e)}')

    async def _send_repo_selection_comment(self, jira_dc_view: JiraDcViewInterface):
        """Send a comment with repository options for the user to choose."""
        try:
            comment_msg = (
                'I need to know which repository to work with. '
                'Please add it to your issue description or send a followup comment.'
            )

            api_key = self.token_manager.decrypt_text(
                jira_dc_view.jira_dc_workspace.svc_acc_api_key
            )

            await self.send_message(
                self.create_outgoing_message(msg=comment_msg),
                issue_key=jira_dc_view.job_context.issue_key,
                base_api_url=jira_dc_view.job_context.base_api_url,
                svc_acc_api_key=api_key,
            )

            logger.info(
                f'[Jira] Sent repository selection comment for issue {jira_dc_view.job_context.issue_key}'
            )

        except Exception as e:
            logger.error(
                f'[Jira] Failed to send repository selection comment: {str(e)}'
            )
