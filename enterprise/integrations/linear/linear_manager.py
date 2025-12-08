import hashlib
import hmac
from typing import Dict, Optional, Tuple

import httpx
from fastapi import Request
from integrations.linear.linear_types import LinearViewInterface
from integrations.linear.linear_view import (
    LinearExistingConversationView,
    LinearFactory,
    LinearNewConversationView,
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
from storage.linear_integration_store import LinearIntegrationStore
from storage.linear_user import LinearUser
from storage.linear_workspace import LinearWorkspace

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import Repository
from openhands.server.shared import server_config
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.server.user_auth.user_auth import UserAuth
from openhands.utils.http_session import httpx_verify_option


class LinearManager(Manager):
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.integration_store = LinearIntegrationStore.get_instance()
        self.api_url = 'https://api.linear.app/graphql'
        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'linear')
        )

    async def authenticate_user(
        self, linear_user_id: str, workspace_id: int
    ) -> tuple[LinearUser | None, UserAuth | None]:
        """Authenticate Linear user and get their OpenHands user auth."""

        # Find active Linear user by Linear user ID and workspace ID
        linear_user = await self.integration_store.get_active_user(
            linear_user_id, workspace_id
        )

        if not linear_user:
            logger.warning(
                f'[Linear] No active Linear user found for {linear_user_id} in workspace {workspace_id}'
            )
            return None, None

        saas_user_auth = await get_user_auth_from_keycloak_id(
            linear_user.keycloak_user_id
        )
        return linear_user, saas_user_auth

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
        """Verify Linear webhook signature."""
        signature = request.headers.get('linear-signature')
        body = await request.body()
        payload = await request.json()
        actor_url = payload.get('actor', {}).get('url', '')
        workspace_name = ''

        # Extract workspace name from actor URL
        # Format: https://linear.app/{workspace}/profiles/{user}
        if actor_url.startswith('https://linear.app/'):
            url_parts = actor_url.split('/')
            if len(url_parts) >= 4:
                workspace_name = url_parts[3]  # Extract workspace name
            else:
                logger.warning(f'[Linear] Invalid actor URL format: {actor_url}')
                return False, None, None
        else:
            logger.warning(
                f'[Linear] Actor URL does not match expected format: {actor_url}'
            )
            return False, None, None

        if not workspace_name:
            logger.warning('[Linear] No workspace name found in webhook payload')
            return False, None, None

        if not signature:
            logger.warning('[Linear] No signature found in webhook headers')
            return False, None, None

        workspace = await self.integration_store.get_workspace_by_name(workspace_name)

        if not workspace:
            logger.warning('[Linear] Could not identify workspace for webhook')
            return False, None, None

        if workspace.status != 'active':
            logger.warning(f'[Linear] Workspace {workspace.id} is not active')
            return False, None, None

        webhook_secret = self.token_manager.decrypt_text(workspace.webhook_secret)
        digest = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()

        if hmac.compare_digest(signature, digest):
            logger.info('[Linear] Webhook signature verified successfully')
            return True, signature, payload

        return False, None, None

    def parse_webhook(self, payload: Dict) -> JobContext | None:
        action = payload.get('action')
        type = payload.get('type')

        if action == 'create' and type == 'Comment':
            data = payload.get('data', {})
            comment = data.get('body', '')

            if '@openhands' not in comment:
                return None

            issue_data = data.get('issue', {})
            issue_id = issue_data.get('id', '')
            issue_key = issue_data.get('identifier', '')
        elif action == 'update' and type == 'Issue':
            data = payload.get('data', {})
            labels = data.get('labels', [])

            has_openhands_label = False
            label_id = ''
            for label in labels:
                if label.get('name') == 'openhands':
                    label_id = label.get('id', '')
                    has_openhands_label = True
                    break

            if not has_openhands_label and not label_id:
                return None

            labelIdChanges = data.get('updatedFrom', {}).get('labelIds', [])

            if labelIdChanges and label_id in labelIdChanges:
                return None  # Label was added previously, ignore this webhook

            issue_id = data.get('id', '')
            issue_key = data.get('identifier', '')
            comment = ''

        else:
            return None

        actor = payload.get('actor', {})
        display_name = actor.get('name', '')
        user_email = actor.get('email', '')
        actor_url = actor.get('url', '')
        actor_id = actor.get('id', '')
        workspace_name = ''

        if actor_url.startswith('https://linear.app/'):
            url_parts = actor_url.split('/')
            if len(url_parts) >= 4:
                workspace_name = url_parts[3]  # Extract workspace name
            else:
                logger.warning(f'[Linear] Invalid actor URL format: {actor_url}')
                return None
        else:
            logger.warning(
                f'[Linear] Actor URL does not match expected format: {actor_url}'
            )
            return None

        if not all(
            [issue_id, issue_key, display_name, user_email, actor_id, workspace_name]
        ):
            logger.warning('[Linear] Missing required fields in webhook payload')
            return None

        return JobContext(
            issue_id=issue_id,
            issue_key=issue_key,
            user_msg=comment,
            user_email=user_email,
            platform_user_id=actor_id,
            workspace_name=workspace_name,
            display_name=display_name,
        )

    async def receive_message(self, message: Message):
        """Process incoming Linear webhook message."""
        payload = message.message.get('payload', {})
        job_context = self.parse_webhook(payload)

        if not job_context:
            logger.info('[Linear] Webhook does not match trigger conditions')
            return

        # Get workspace by user email domain
        workspace = await self.integration_store.get_workspace_by_name(
            job_context.workspace_name
        )
        if not workspace:
            logger.warning(
                f'[Linear] No workspace found for email domain: {job_context.workspace_name}'
            )
            await self._send_error_comment(
                job_context.issue_id,
                'Your workspace is not configured with Linear integration.',
                None,
            )
            return

        # Prevent any recursive triggers from the service account
        if job_context.user_email == workspace.svc_acc_email:
            return

        if workspace.status != 'active':
            logger.warning(f'[Linear] Workspace {workspace.id} is not active')
            await self._send_error_comment(
                job_context.issue_id,
                'Linear integration is not active for your workspace.',
                workspace,
            )
            return

        # Authenticate user
        linear_user, saas_user_auth = await self.authenticate_user(
            job_context.platform_user_id, workspace.id
        )
        if not linear_user or not saas_user_auth:
            logger.warning(
                f'[Linear] User authentication failed for {job_context.user_email}'
            )
            await self._send_error_comment(
                job_context.issue_id,
                f'User {job_context.user_email} is not authenticated or active in the Linear integration.',
                workspace,
            )
            return

        # Get issue details
        try:
            api_key = self.token_manager.decrypt_text(workspace.svc_acc_api_key)
            issue_title, issue_description = await self.get_issue_details(
                job_context.issue_id, api_key
            )
            job_context.issue_title = issue_title
            job_context.issue_description = issue_description
        except Exception as e:
            logger.error(f'[Linear] Failed to get issue context: {str(e)}')
            await self._send_error_comment(
                job_context.issue_id,
                'Failed to retrieve issue details. Please check the issue ID and try again.',
                workspace,
            )
            return

        try:
            # Create Linear view
            linear_view = await LinearFactory.create_linear_view_from_payload(
                job_context,
                saas_user_auth,
                linear_user,
                workspace,
            )
        except Exception as e:
            logger.error(
                f'[Linear] Failed to create linear view: {str(e)}', exc_info=True
            )
            await self._send_error_comment(
                job_context.issue_id,
                'Failed to initialize conversation. Please try again.',
                workspace,
            )
            return

        if not await self.is_job_requested(message, linear_view):
            return

        await self.start_job(linear_view)

    async def is_job_requested(
        self, message: Message, linear_view: LinearViewInterface
    ) -> bool:
        """
        Check if a job is requested and handle repository selection.
        """

        if isinstance(linear_view, LinearExistingConversationView):
            return True

        try:
            # Get user repositories
            user_repos: list[Repository] = await self._get_repositories(
                linear_view.saas_user_auth
            )

            target_str = f'{linear_view.job_context.issue_description}\n{linear_view.job_context.user_msg}'

            # Try to infer repository from issue description
            match, repos = filter_potential_repos_by_user_msg(target_str, user_repos)

            if match:
                # Found exact repository match
                linear_view.selected_repo = repos[0].full_name
                logger.info(f'[Linear] Inferred repository: {repos[0].full_name}')
                return True
            else:
                # No clear match - send repository selection comment
                await self._send_repo_selection_comment(linear_view)
                return False

        except Exception as e:
            logger.error(f'[Linear] Error in is_job_requested: {str(e)}')
            return False

    async def start_job(self, linear_view: LinearViewInterface):
        """Start a Linear job/conversation."""
        # Import here to prevent circular import
        from server.conversation_callback_processor.linear_callback_processor import (
            LinearCallbackProcessor,
        )

        try:
            user_info: LinearUser = linear_view.linear_user
            logger.info(
                f'[Linear] Starting job for user {user_info.keycloak_user_id} '
                f'issue {linear_view.job_context.issue_key}',
            )

            # Create conversation
            conversation_id = await linear_view.create_or_update_conversation(
                self.jinja_env
            )

            logger.info(
                f'[Linear] Created/Updated conversation {conversation_id} for issue {linear_view.job_context.issue_key}'
            )

            if isinstance(linear_view, LinearNewConversationView):
                # Register callback processor for updates
                processor = LinearCallbackProcessor(
                    issue_id=linear_view.job_context.issue_id,
                    issue_key=linear_view.job_context.issue_key,
                    workspace_name=linear_view.linear_workspace.name,
                )

                # Register the callback processor
                register_callback_processor(conversation_id, processor)

                logger.info(
                    f'[Linear] Created callback processor for conversation {conversation_id}'
                )

            # Send initial response
            msg_info = linear_view.get_response_msg()

        except MissingSettingsError as e:
            logger.warning(f'[Linear] Missing settings error: {str(e)}')
            msg_info = f'Please re-login into [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except LLMAuthenticationError as e:
            logger.warning(f'[Linear] LLM authentication error: {str(e)}')
            msg_info = f'Please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except Exception as e:
            logger.error(
                f'[Linear] Unexpected error starting job: {str(e)}', exc_info=True
            )
            msg_info = 'Sorry, there was an unexpected error starting the job. Please try again.'

        # Send response comment
        try:
            api_key = self.token_manager.decrypt_text(
                linear_view.linear_workspace.svc_acc_api_key
            )
            await self.send_message(
                self.create_outgoing_message(msg=msg_info),
                linear_view.job_context.issue_id,
                api_key,
            )
        except Exception as e:
            logger.error(f'[Linear] Failed to send response message: {str(e)}')

    async def _query_api(self, query: str, variables: Dict, api_key: str) -> Dict:
        """Query Linear GraphQL API."""
        headers = {'Authorization': api_key}
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json={'query': query, 'variables': variables},
            )
            response.raise_for_status()
            return response.json()

    async def get_issue_details(self, issue_id: str, api_key: str) -> Tuple[str, str]:
        """Get issue details from Linear API."""
        query = """
            query Issue($issueId: String!) {
              issue(id: $issueId) {
                id
                identifier
                title
                description
                syncedWith {
                    metadata {
                        ... on ExternalEntityInfoGithubMetadata {
                        owner
                        repo
                        }
                    }
                }
              }
            }
        """
        issue_payload = await self._query_api(query, {'issueId': issue_id}, api_key)

        if not issue_payload:
            raise ValueError(f'Issue with ID {issue_id} not found.')

        issue_data = issue_payload.get('data', {}).get('issue', {})
        title = issue_data.get('title', '')
        description = issue_data.get('description', '')
        synced_with = issue_data.get('syncedWith', [])
        owner = ''
        repo = ''
        if synced_with:
            owner = synced_with[0].get('metadata', {}).get('owner', '')
            repo = synced_with[0].get('metadata', {}).get('repo', '')

        if not title:
            raise ValueError(f'Issue with ID {issue_id} does not have a title.')

        if not description:
            raise ValueError(f'Issue with ID {issue_id} does not have a description.')

        if owner and repo:
            description += f'\n\nGit Repo: {owner}/{repo}'

        return title, description

    async def send_message(self, message: Message, issue_id: str, api_key: str):
        """Send message/comment to Linear issue."""
        query = """
            mutation CommentCreate($input: CommentCreateInput!) {
              commentCreate(input: $input) {
                success
                comment {
                  id
                }
              }
            }
        """
        variables = {'input': {'issueId': issue_id, 'body': message.message}}
        return await self._query_api(query, variables, api_key)

    async def _send_error_comment(
        self, issue_id: str, error_msg: str, workspace: LinearWorkspace | None
    ):
        """Send error comment to Linear issue."""
        if not workspace:
            logger.error('[Linear] Cannot send error comment - no workspace available')
            return

        try:
            api_key = self.token_manager.decrypt_text(workspace.svc_acc_api_key)
            await self.send_message(
                self.create_outgoing_message(msg=error_msg), issue_id, api_key
            )
        except Exception as e:
            logger.error(f'[Linear] Failed to send error comment: {str(e)}')

    async def _send_repo_selection_comment(self, linear_view: LinearViewInterface):
        """Send a comment with repository options for the user to choose."""
        try:
            comment_msg = (
                'I need to know which repository to work with. '
                'Please add it to your issue description or send a followup comment.'
            )

            api_key = self.token_manager.decrypt_text(
                linear_view.linear_workspace.svc_acc_api_key
            )

            await self.send_message(
                self.create_outgoing_message(msg=comment_msg),
                linear_view.job_context.issue_id,
                api_key,
            )

            logger.info(
                f'[Linear] Sent repository selection comment for issue {linear_view.job_context.issue_key}'
            )

        except Exception as e:
            logger.error(
                f'[Linear] Failed to send repository selection comment: {str(e)}'
            )
