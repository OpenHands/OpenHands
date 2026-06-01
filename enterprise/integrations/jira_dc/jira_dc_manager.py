import hashlib
import hmac
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
from fastapi import Request
from integrations.jira_dc.jira_dc_service_account import (
    resolve_jira_dc_service_account,
)
from integrations.jira_dc.jira_dc_types import (
    JiraDcViewInterface,
)
from integrations.jira_dc.jira_dc_view import (
    JiraDcExistingConversationView,
    JiraDcFactory,
)
from integrations.manager import Manager
from integrations.models import JobContext, Message
from integrations.utils import (
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
    filter_potential_repos_by_user_msg,
    get_account_not_linked_message,
    get_session_expired_message,
    get_user_not_found_message,
    infer_repo_from_message,
    markdown_to_jira_markup,
)
from jinja2 import Environment, FileSystemLoader
from server.auth.constants import JIRA_DC_ENABLE_OAUTH
from server.auth.saas_user_auth import get_user_auth_from_keycloak_id
from server.auth.token_manager import TokenManager
from storage.jira_dc_integration_store import JiraDcIntegrationStore
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.app_server.integrations.provider import ProviderHandler
from openhands.app_server.integrations.service_types import Comment, Repository
from openhands.app_server.shared import server_config
from openhands.app_server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)
from openhands.app_server.user_auth.user_auth import UserAuth
from openhands.app_server.utils.http_session import httpx_verify_option
from openhands.app_server.utils.logger import openhands_logger as logger

# Unicode codepoint of the emoji reaction posted to acknowledge an @openhands
# mention via Jira's internal reactions API. 1f44d = 👍 (thumbs up). Note:
# 1f440 (👀 eyes) is NOT in Jira DC's reaction palette, so thumbs-up is used.
JIRA_DC_REACTION_EMOJI_ID = '1f44d'

# Events the OpenHands webhook subscribes to, used when auto-enrolling the
# webhook in Jira. The resolver only creates jobs for a narrower subset in
# parse_webhook, but automations can subscribe to these broader issue/comment
# lifecycle events.
JIRA_DC_WEBHOOK_EVENTS = [
    'jira:issue_created',
    'jira:issue_updated',
    'jira:issue_deleted',
    'comment_created',
    'comment_updated',
    'comment_deleted',
]


def _extract_workspace_url(payload: Dict) -> str:
    """Return a Jira URL whose host identifies the configured workspace."""
    for url in _extract_workspace_urls(payload):
        return url
    return ''


def _extract_workspace_urls(payload: Dict) -> list[str]:
    """Return Jira URLs whose hosts identify the configured workspace."""
    paths = (
        ('comment', 'author', 'self'),
        ('user', 'self'),
        ('issue', 'self'),
        ('comment', 'self'),
    )

    urls = []
    for path in paths:
        value: object = payload
        for key in path:
            if not isinstance(value, dict):
                break
            value = value.get(key)
        else:
            if isinstance(value, str) and value:
                urls.append(value)

    return urls


def _extract_workspace_hosts(payload: Dict) -> set[str]:
    """Return hostnames from Jira URLs embedded in a webhook payload."""
    return {
        parsed.hostname.lower()
        for parsed in (urlparse(url) for url in _extract_workspace_urls(payload))
        if parsed.hostname
    }


class JiraDcManager(Manager[JiraDcViewInterface]):
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
        # In email-match mode (OAuth disabled) the workspace link is stored with
        # an 'unavailable' Jira account id, so the webhook's real Jira user key
        # can never match a stored row. Resolve the user by matching their Jira
        # email to their OpenHands email instead. In OAuth mode we resolve
        # strictly by the verified Jira account id and never fall back to email,
        # preserving the verification guarantee.
        if not JIRA_DC_ENABLE_OAUTH or not jira_dc_user_id or jira_dc_user_id == 'none':
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
        signature_valid, signature, payload, _ = await self.validate_request_context(
            request
        )
        return signature_valid, signature, payload

    async def validate_request_context(
        self, request: Request, workspace_id: int | None = None
    ) -> Tuple[bool, Optional[str], Optional[Dict], Optional[JiraDcWorkspace]]:
        """Verify Jira DC webhook signature and return the matched workspace."""
        signature_header = request.headers.get('x-hub-signature')
        signature = signature_header.split('=')[1] if signature_header else None
        body = await request.body()
        payload = await request.json()

        if not signature:
            logger.warning('[Jira DC] No signature found in webhook headers')
            return False, None, None, None

        if workspace_id is not None:
            workspace = await self.integration_store.get_workspace_by_id(workspace_id)
            payload_hosts = _extract_workspace_hosts(payload)
            if not payload_hosts:
                logger.warning(
                    '[Jira DC] No workspace host found in connection-scoped webhook payload'
                )
                return False, None, None, None
            if workspace and any(
                host != workspace.name.lower() for host in payload_hosts
            ):
                logger.warning(
                    '[Jira DC] Webhook payload hosts %s do not match connection %s (%s)',
                    sorted(payload_hosts),
                    workspace.id,
                    workspace.name,
                )
                return False, None, None, None
        else:
            workspace_name = ''
            parsed_url = urlparse(_extract_workspace_url(payload))
            if parsed_url.hostname:
                workspace_name = parsed_url.hostname

            if not workspace_name:
                logger.warning('[Jira DC] No workspace name found in webhook payload')
                return False, None, None, None

            workspace = await self.integration_store.get_workspace_by_name(
                workspace_name
            )

        if not workspace:
            logger.warning('[Jira DC] Could not identify workspace for webhook')
            return False, None, None, None

        if workspace.status != 'active':
            logger.warning(f'[Jira DC] Workspace {workspace.id} is not active')
            return False, None, None, None

        webhook_secret = self.token_manager.decrypt_text(workspace.webhook_secret)
        digest = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()

        if hmac.compare_digest(signature, digest):
            logger.info('[Jira DC] Webhook signature verified successfully')
            return True, signature, payload, workspace

        return False, None, None, None

    def parse_webhook(self, payload: Dict) -> JobContext | None:
        event_type = payload.get('webhookEvent')

        if event_type == 'comment_created':
            comment_data = payload.get('comment', {})
            comment = comment_data.get('body', '')
            comment_id = comment_data.get('id')

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
            comment_id = None
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
            comment_id=comment_id or '',
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

        try:
            service_account = resolve_jira_dc_service_account(
                workspace, self.token_manager
            )
        except Exception as e:
            logger.error(
                f'[Jira DC] Service account configuration is invalid: {str(e)}'
            )
            return

        # Prevent any recursive triggers from the service account
        if job_context.user_email == service_account.email:
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
            # Distinguish "no OpenHands account" from "account exists but not linked
            # to this workspace" so the reply is actionable (mirrors GitHub/BBDC).
            keycloak_user_id = await self.token_manager.get_user_id_from_user_email(
                job_context.user_email
            )
            if keycloak_user_id:
                error_msg = get_account_not_linked_message(job_context.display_name)
            else:
                error_msg = get_user_not_found_message(job_context.display_name)
            await self._send_error_comment(job_context, error_msg, workspace)
            return

        # Get issue details
        try:
            issue_title, issue_description = await self.get_issue_details(
                job_context, service_account.api_key
            )
            job_context.issue_title = issue_title
            job_context.issue_description = issue_description
            job_context.previous_comments = await self.get_issue_comments(
                job_context,
                service_account.api_key,
                bot_email=service_account.email,
            )
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

        await self._add_acknowledgement_reaction(job_context, workspace)
        await self.start_job(jira_dc_view)

    async def is_job_requested(
        self, message: Message, jira_dc_view: JiraDcViewInterface
    ) -> bool:
        """Check if a job is requested and handle repository selection."""
        if isinstance(jira_dc_view, JiraDcExistingConversationView):
            return True

        try:
            # Get user repositories
            user_repos: list[Repository] = await self._get_repositories(
                jira_dc_view.saas_user_auth
            )

            target_str = f'{jira_dc_view.job_context.issue_description}\n{jira_dc_view.job_context.user_msg}'
            mentioned_repos = infer_repo_from_message(target_str)

            # Try to infer repository from issue description
            match, repos = filter_potential_repos_by_user_msg(target_str, user_repos)

            if match:
                # Found exact repository match
                jira_dc_view.selected_repo = repos[0].full_name
                logger.info(f'[Jira DC] Inferred repository: {repos[0].full_name}')
                return True
            else:
                # No clear match - send repository selection comment
                matched_repos = [
                    repo
                    for repo in user_repos
                    if any(
                        mentioned_repo.lower() in repo.full_name.lower()
                        for mentioned_repo in mentioned_repos
                    )
                ]
                await self._send_repo_selection_comment(
                    jira_dc_view, mentioned_repos, matched_repos
                )
                return False

        except Exception as e:
            logger.error(f'[Jira DC] Error in is_job_requested: {str(e)}')
            return False

    async def start_job(self, jira_dc_view: JiraDcViewInterface) -> None:
        """Start a Jira DC job/conversation using V1 app conversation system."""
        try:
            user_info: JiraDcUser = jira_dc_view.jira_dc_user
            logger.info(
                f'[Jira DC] Starting job for user {user_info.keycloak_user_id} '
                f'issue {jira_dc_view.job_context.issue_key}',
            )

            # Create conversation using V1 app conversation system
            # The callback processor is registered automatically by the view
            conversation_id = await jira_dc_view.create_or_update_conversation(
                self.jinja_env
            )

            logger.info(
                f'[Jira DC] Created/Updated conversation {conversation_id} for issue {jira_dc_view.job_context.issue_key}'
            )

            # Send initial response
            msg_info = jira_dc_view.get_response_msg()

        except MissingSettingsError as e:
            logger.warning(f'[Jira DC] Missing settings error: {str(e)}')
            msg_info = f'Please re-login into [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except LLMAuthenticationError as e:
            logger.warning(f'[Jira DC] LLM authentication error: {str(e)}')
            msg_info = f'Please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except SessionExpiredError as e:
            logger.warning(f'[Jira DC] Session expired: {str(e)}')
            msg_info = get_session_expired_message()

        except Exception as e:
            logger.error(
                f'[Jira DC] Unexpected error starting job: {str(e)}', exc_info=True
            )
            msg_info = 'Sorry, there was an unexpected error starting the job. Please try again.'

        # Send response comment
        try:
            service_account = resolve_jira_dc_service_account(
                jira_dc_view.jira_dc_workspace, self.token_manager
            )
            await self.send_message(
                msg_info,
                issue_key=jira_dc_view.job_context.issue_key,
                base_api_url=jira_dc_view.job_context.base_api_url,
                svc_acc_api_key=service_account.api_key,
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
            if response.status_code == 401:
                logger.error(
                    '[Jira DC] 401 from %s. PAT length=%d prefix=%s '
                    'WWW-Authenticate=%r X-Seraph-LoginReason=%r '
                    'X-AUSERNAME=%r body=%s',
                    url,
                    len(svc_acc_api_key),
                    svc_acc_api_key[:6] if svc_acc_api_key else '',
                    response.headers.get('WWW-Authenticate'),
                    response.headers.get('X-Seraph-LoginReason'),
                    response.headers.get('X-AUSERNAME'),
                    response.text[:500],
                )
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

    async def get_issue_comments(
        self,
        job_context: JobContext,
        svc_acc_api_key: str,
        bot_email: str | None = None,
        max_comments: int = 15,
    ) -> list[Comment]:
        """Fetch the issue's comment thread for conversation context.

        Returns up to ``max_comments`` of the most recent comments in
        chronological (oldest-first) order, excluding the triggering comment
        (which is surfaced separately as the actionable request). Comments
        authored by the integration's service account are flagged via
        ``Comment.system`` so the prompt can label them as OpenHands' own prior
        replies rather than instructions. Best-effort: any failure returns an
        empty list so a transient comments-API issue never blocks the job.
        """
        url = (
            f'{job_context.base_api_url}/rest/api/2/issue/'
            f'{job_context.issue_key}/comment'
        )
        headers = {'Authorization': f'Bearer {svc_acc_api_key}'}
        # '-created' + reverse keeps the tail (most recent N) of long threads,
        # which is the relevant part, rather than the oldest N.
        params: dict[str, str | int] = {
            'orderBy': '-created',
            'maxResults': max_comments,
        }
        try:
            async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                raw_comments = response.json().get('comments', [])
        except Exception as e:
            logger.warning(
                f'[Jira DC] Failed to fetch comment thread for '
                f'{job_context.issue_key} (non-fatal, continuing without '
                f'history): {str(e)}'
            )
            return []

        comments: list[Comment] = []
        for raw in reversed(raw_comments):  # restore oldest -> newest order
            try:
                if str(raw.get('id', '')) == str(job_context.comment_id):
                    continue  # shown separately as the actionable request
                author = raw.get('author', {}) or {}
                author_email = author.get('emailAddress')
                comments.append(
                    Comment(
                        id=str(raw.get('id', '')),
                        body=raw.get('body', '') or '',
                        author=author.get('displayName')
                        or author.get('name')
                        or 'unknown',
                        created_at=raw.get('created'),
                        updated_at=raw.get('updated') or raw.get('created'),
                        system=bool(
                            bot_email and author_email and author_email == bot_email
                        ),
                    )
                )
            except Exception as e:
                logger.debug(f'[Jira DC] Skipping unparseable comment: {str(e)}')
                continue
        return comments

    async def send_message(
        self, message: str, issue_key: str, base_api_url: str, svc_acc_api_key: str
    ):
        """Send message/comment to Jira DC issue.

        Args:
            message: The message content to send (plain text string)
            issue_key: The Jira issue key (e.g., 'PROJ-123')
            base_api_url: The base API URL for the Jira DC instance
            svc_acc_api_key: Service account API key for authentication
        """
        url = f'{base_api_url}/rest/api/2/issue/{issue_key}/comment'
        headers = {'Authorization': f'Bearer {svc_acc_api_key}'}
        # Convert standard Markdown to Jira Wiki Markup for proper rendering
        data = {'body': markdown_to_jira_markup(message)}
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()

    async def add_reaction(
        self,
        comment_id: str,
        base_api_url: str,
        svc_acc_api_key: str,
        emoji_id: str = JIRA_DC_REACTION_EMOJI_ID,
    ):
        """Add an emoji reaction to a Jira DC comment as the service account.

        Uses Jira Data Center's internal reactions API (the endpoint the web UI
        calls). emoji_id is a Unicode codepoint string, e.g. '1f44d' for 👍.

        Args:
            comment_id: The id of the comment to react to.
            base_api_url: The base API URL for the Jira DC instance.
            svc_acc_api_key: Service account PAT used to authenticate.
            emoji_id: Unicode codepoint of the reaction emoji.
        """
        url = f'{base_api_url}/rest/internal/2/reactions'
        headers = {'Authorization': f'Bearer {svc_acc_api_key}'}
        data = {'commentId': comment_id, 'emojiId': emoji_id}
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()

    async def _add_acknowledgement_reaction(
        self, job_context: JobContext, workspace: JiraDcWorkspace
    ):
        """Acknowledge the @openhands mention with a best-effort reaction.

        Reactions are non-essential, so failures are logged, never raised.
        """
        if not job_context.comment_id:
            return
        try:
            service_account = resolve_jira_dc_service_account(
                workspace, self.token_manager
            )
            await self.add_reaction(
                comment_id=job_context.comment_id,
                base_api_url=job_context.base_api_url,
                svc_acc_api_key=service_account.api_key,
            )
            logger.info(
                f'[Jira DC] Reacted to comment {job_context.comment_id} on issue {job_context.issue_key}'
            )
        except Exception as e:
            logger.warning(
                f'[Jira DC] Failed to add acknowledgement reaction (non-fatal): {str(e)}'
            )

    async def register_webhook(
        self,
        base_api_url: str,
        admin_api_key: str,
        events_url: str,
        secret: str,
        name: str = 'OpenHands',
    ) -> int:
        """Create or update the OpenHands webhook in Jira DC via the admin API.

        Uses Jira Data Center's ``jira-webhook`` plugin REST API (the same one the
        admin UI calls). Idempotent: if a webhook already targets ``events_url`` it
        is updated in place (preserving its id); otherwise a new one is created.

        Args:
            base_api_url: Jira base URL, e.g. ``https://jira.example.com``.
            admin_api_key: A Jira admin PAT. Used only for this call; never stored.
            events_url: The OpenHands endpoint Jira should POST events to.
            secret: HMAC signing secret Jira will sign deliveries with. Must match
                the workspace's stored ``webhook_secret`` or verification rejects.
            name: Display name for the webhook.

        Returns:
            The id of the created or updated webhook.
        """
        base = base_api_url.rstrip('/')
        collection_url = f'{base}/rest/jira-webhook/1.0/webhooks'
        headers = {'Authorization': f'Bearer {admin_api_key}'}
        payload = {
            'name': name,
            'url': events_url,
            'events': JIRA_DC_WEBHOOK_EVENTS,
            'active': True,
            'scopeType': 'global',
            'configuration': {'SECRET': secret, 'EXCLUDE_BODY': 'false'},
        }

        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            # Idempotency: reuse any existing webhook already pointing at our URL.
            listing = await client.get(collection_url, headers=headers)
            listing.raise_for_status()
            existing = next(
                (w for w in listing.json() if w.get('url') == events_url), None
            )

            if existing:
                webhook_id = existing['id']
                response = await client.put(
                    f'{collection_url}/{webhook_id}',
                    headers=headers,
                    json={**payload, 'id': webhook_id},
                )
                response.raise_for_status()
                logger.info(f'[Jira DC] Updated webhook {webhook_id} -> {events_url}')
                return webhook_id

            response = await client.post(
                collection_url, headers=headers, json={**payload, 'id': None}
            )
            response.raise_for_status()
            webhook_id = response.json().get('id')
            logger.info(f'[Jira DC] Created webhook {webhook_id} -> {events_url}')
            return webhook_id

    async def delete_webhook(
        self,
        base_api_url: str,
        admin_api_key: str,
        events_url: str,
    ) -> bool:
        """Delete the OpenHands webhook from Jira DC, if present.

        Counterpart to :meth:`register_webhook`. Looks up the webhook that
        targets ``events_url`` and deletes it via the same ``jira-webhook``
        plugin REST API. Idempotent: returns ``False`` (not an error) when no
        matching webhook exists.

        Args:
            base_api_url: Jira base URL, e.g. ``https://jira.example.com``.
            admin_api_key: A Jira admin PAT. Used only for this call; never
                stored.
            events_url: The OpenHands endpoint whose webhook should be removed.

        Returns:
            True if a webhook was deleted; False if there was nothing to delete.
        """
        base = base_api_url.rstrip('/')
        collection_url = f'{base}/rest/jira-webhook/1.0/webhooks'
        headers = {'Authorization': f'Bearer {admin_api_key}'}

        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            listing = await client.get(collection_url, headers=headers)
            listing.raise_for_status()
            existing = next(
                (w for w in listing.json() if w.get('url') == events_url), None
            )
            if not existing:
                logger.info(
                    f'[Jira DC] No webhook found for {events_url}; nothing to delete'
                )
                return False

            webhook_id = existing['id']
            response = await client.delete(
                f'{collection_url}/{webhook_id}', headers=headers
            )
            response.raise_for_status()
            logger.info(f'[Jira DC] Deleted webhook {webhook_id} -> {events_url}')
            return True

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
            service_account = resolve_jira_dc_service_account(
                workspace, self.token_manager
            )
            await self.send_message(
                error_msg,
                issue_key=job_context.issue_key,
                base_api_url=job_context.base_api_url,
                svc_acc_api_key=service_account.api_key,
            )
        except Exception as e:
            logger.error(f'[Jira DC] Failed to send error comment: {str(e)}')

    async def _send_repo_selection_comment(
        self,
        jira_dc_view: JiraDcViewInterface,
        mentioned_repos: list[str] | None = None,
        matched_repos: list[Repository] | None = None,
    ):
        """Send a comment with repository options for the user to choose."""
        try:
            mentioned_repos = mentioned_repos or []
            matched_repo_names = [repo.full_name for repo in matched_repos or []]
            if not mentioned_repos:
                comment_msg = (
                    'Could not determine which repository to use. '
                    'Please mention the repository (e.g., owner/repo) in the issue '
                    'description or comment.'
                )
            elif len(matched_repo_names) > 1:
                comment_msg = (
                    f'Multiple repositories found: {", ".join(matched_repo_names)}. '
                    'Please specify exactly one repository in the issue description '
                    'or comment.'
                )
            else:
                comment_msg = (
                    f'Could not access any of the mentioned repositories: '
                    f'{", ".join(mentioned_repos)}. '
                    'Please ensure your OpenHands account has access to the '
                    'repository and it exists.'
                )

            service_account = resolve_jira_dc_service_account(
                jira_dc_view.jira_dc_workspace, self.token_manager
            )

            await self.send_message(
                comment_msg,
                issue_key=jira_dc_view.job_context.issue_key,
                base_api_url=jira_dc_view.job_context.base_api_url,
                svc_acc_api_key=service_account.api_key,
            )

            logger.info(
                f'[Jira] Sent repository selection comment for issue {jira_dc_view.job_context.issue_key}'
            )

        except Exception as e:
            logger.error(
                f'[Jira] Failed to send repository selection comment: {str(e)}'
            )
