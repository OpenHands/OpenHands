"""Jira integration manager.

This module orchestrates the processing of Jira webhook events:
1. Parse webhook payload (via JiraPayloadParser)
2. Validate workspace
3. Authenticate user
4. Create view with repository selection (via JiraFactory)
5. Start conversation job

The manager delegates payload parsing to JiraPayloadParser and view creation
to JiraFactory, keeping the orchestration logic clean and traceable.
"""

import httpx
from integrations.jira.jira_payload import (
    JiraPayloadError,
    JiraPayloadParser,
    JiraPayloadSkipped,
    JiraPayloadSuccess,
    JiraWebhookPayload,
)
from integrations.jira.jira_types import (
    JiraViewInterface,
    RepositoryNotFoundError,
    StartingConvoException,
)
from integrations.jira.jira_view import JiraFactory, JiraNewConversationView
from integrations.manager import Manager
from integrations.models import Message
from integrations.utils import (
    HOST,
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
    get_oh_labels,
    get_session_expired_message,
)
from jinja2 import Environment, FileSystemLoader
from server.auth.saas_user_auth import get_user_auth_from_keycloak_id
from server.auth.token_manager import TokenManager
from server.utils.conversation_callback_utils import register_callback_processor
from storage.jira_integration_store import JiraIntegrationStore
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace

from openhands.core.logger import openhands_logger as logger
from openhands.server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)
from openhands.server.user_auth.user_auth import UserAuth
from openhands.utils.http_session import httpx_verify_option

JIRA_CLOUD_API_URL = 'https://api.atlassian.com/ex/jira'

# Get OH labels for this environment
OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)


class JiraManager(Manager):
    """Manager for processing Jira webhook events.

    This class orchestrates the flow from webhook receipt to conversation creation,
    delegating parsing to JiraPayloadParser and view creation to JiraFactory.
    """

    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.integration_store = JiraIntegrationStore.get_instance()
        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'jira')
        )
        self.payload_parser = JiraPayloadParser(
            oh_label=OH_LABEL,
            inline_oh_label=INLINE_OH_LABEL,
        )

    async def receive_message(self, message: Message):
        """Process incoming Jira webhook message.

        Flow:
        1. Parse webhook payload
        2. Validate workspace exists and is active
        3. Authenticate user
        4. Create view (includes fetching issue details and selecting repository)
        5. Start job

        Each step has clear logging for traceability.
        """
        raw_payload = message.message.get('payload', {})

        # Step 1: Parse webhook payload
        logger.info(
            '[Jira] Received webhook',
            extra={'raw_payload': raw_payload},
        )

        parse_result = self.payload_parser.parse(raw_payload)

        if isinstance(parse_result, JiraPayloadSkipped):
            logger.info(
                '[Jira] Webhook skipped', extra={'reason': parse_result.skip_reason}
            )
            return

        if isinstance(parse_result, JiraPayloadError):
            logger.warning(
                '[Jira] Webhook parse failed', extra={'error': parse_result.error}
            )
            return

        payload = parse_result.payload
        logger.info(
            '[Jira] Processing webhook',
            extra={
                'event_type': payload.event_type.value,
                'issue_key': payload.issue_key,
                'user_email': payload.user_email,
            },
        )

        # Step 2: Validate workspace
        workspace = await self._get_active_workspace(payload)
        if not workspace:
            return

        # Step 3: Authenticate user
        jira_user, saas_user_auth = await self._authenticate_user(payload, workspace)
        if not jira_user or not saas_user_auth:
            return

        # Step 4: Create view (includes issue details fetch and repo selection)
        decrypted_api_key = self.token_manager.decrypt_text(workspace.svc_acc_api_key)

        try:
            view = await JiraFactory.create_view(
                payload=payload,
                workspace=workspace,
                user=jira_user,
                user_auth=saas_user_auth,
                decrypted_api_key=decrypted_api_key,
            )
        except RepositoryNotFoundError as e:
            logger.warning(
                '[Jira] Repository not found',
                extra={'issue_key': payload.issue_key, 'error': str(e)},
            )
            await self._send_error_from_payload(payload, workspace, str(e))
            return
        except StartingConvoException as e:
            logger.warning(
                '[Jira] View creation failed',
                extra={'issue_key': payload.issue_key, 'error': str(e)},
            )
            await self._send_error_from_payload(payload, workspace, str(e))
            return
        except Exception as e:
            logger.error(
                '[Jira] Unexpected error creating view',
                extra={'issue_key': payload.issue_key, 'error': str(e)},
                exc_info=True,
            )
            await self._send_error_from_payload(
                payload,
                workspace,
                'Failed to initialize conversation. Please try again.',
            )
            return

        # Step 5: Start job
        await self.start_job(view)

    async def _get_active_workspace(
        self, payload: JiraWebhookPayload
    ) -> JiraWorkspace | None:
        """Validate and return the workspace for the webhook.

        Returns None if:
        - Workspace not found
        - Workspace is inactive
        - Request is from service account (to prevent recursion)
        """
        workspace = await self.integration_store.get_workspace_by_name(
            payload.workspace_name
        )

        if not workspace:
            logger.warning(
                '[Jira] Workspace not found',
                extra={'workspace_name': payload.workspace_name},
            )
            # Can't send error without workspace credentials
            return None

        # Prevent recursive triggers from service account
        if payload.user_email == workspace.svc_acc_email:
            logger.debug(
                '[Jira] Ignoring service account trigger',
                extra={'workspace_name': payload.workspace_name},
            )
            return None

        if workspace.status != 'active':
            logger.warning(
                '[Jira] Workspace inactive',
                extra={'workspace_id': workspace.id, 'status': workspace.status},
            )
            await self._send_error_from_payload(
                payload, workspace, 'Jira integration is not active for your workspace.'
            )
            return None

        return workspace

    async def _authenticate_user(
        self, payload: JiraWebhookPayload, workspace: JiraWorkspace
    ) -> tuple[JiraUser | None, UserAuth | None]:
        """Authenticate the Jira user and get OpenHands auth."""
        jira_user = await self.integration_store.get_active_user(
            payload.account_id, workspace.id
        )

        if not jira_user:
            logger.warning(
                '[Jira] User not found or inactive',
                extra={
                    'account_id': payload.account_id,
                    'user_email': payload.user_email,
                    'workspace_id': workspace.id,
                },
            )
            await self._send_error_from_payload(
                payload,
                workspace,
                f'User {payload.user_email} is not authenticated or active in the Jira integration.',
            )
            return None, None

        saas_user_auth = await get_user_auth_from_keycloak_id(
            jira_user.keycloak_user_id
        )

        if not saas_user_auth:
            logger.warning(
                '[Jira] Failed to get OpenHands auth',
                extra={
                    'keycloak_user_id': jira_user.keycloak_user_id,
                    'user_email': payload.user_email,
                },
            )
            await self._send_error_from_payload(
                payload,
                workspace,
                f'User {payload.user_email} is not authenticated with OpenHands.',
            )
            return None, None

        return jira_user, saas_user_auth

    async def start_job(self, view: JiraViewInterface):
        """Start a Jira job/conversation."""
        # Import here to prevent circular import
        from server.conversation_callback_processor.jira_callback_processor import (
            JiraCallbackProcessor,
        )

        try:
            logger.info(
                '[Jira] Starting job',
                extra={
                    'issue_key': view.payload.issue_key,
                    'user_id': view.jira_user.keycloak_user_id,
                    'selected_repo': view.selected_repo,
                },
            )

            # Create conversation
            conversation_id = await view.create_or_update_conversation(self.jinja_env)

            logger.info(
                '[Jira] Conversation created',
                extra={
                    'conversation_id': conversation_id,
                    'issue_key': view.payload.issue_key,
                },
            )

            # Register callback processor for updates
            if isinstance(view, JiraNewConversationView):
                processor = JiraCallbackProcessor(
                    issue_key=view.payload.issue_key,
                    workspace_name=view.jira_workspace.name,
                )
                register_callback_processor(conversation_id, processor)
                logger.info(
                    '[Jira] Callback processor registered',
                    extra={'conversation_id': conversation_id},
                )

            # Send success response
            msg_info = view.get_response_msg()

        except MissingSettingsError as e:
            logger.warning(
                '[Jira] Missing settings error',
                extra={'issue_key': view.payload.issue_key, 'error': str(e)},
            )
            msg_info = f'Please re-login into [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except LLMAuthenticationError as e:
            logger.warning(
                '[Jira] LLM authentication error',
                extra={'issue_key': view.payload.issue_key, 'error': str(e)},
            )
            msg_info = f'Please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

        except SessionExpiredError as e:
            logger.warning(
                '[Jira] Session expired',
                extra={'issue_key': view.payload.issue_key, 'error': str(e)},
            )
            msg_info = get_session_expired_message()

        except StartingConvoException as e:
            logger.warning(
                '[Jira] Conversation start failed',
                extra={'issue_key': view.payload.issue_key, 'error': str(e)},
            )
            msg_info = str(e)

        except Exception as e:
            logger.error(
                '[Jira] Unexpected error starting job',
                extra={'issue_key': view.payload.issue_key, 'error': str(e)},
                exc_info=True,
            )
            msg_info = 'Sorry, there was an unexpected error starting the job. Please try again.'

        # Send response comment
        await self._send_comment(view, msg_info)

    async def send_message(
        self,
        message: Message,
        issue_key: str,
        jira_cloud_id: str,
        svc_acc_email: str,
        svc_acc_api_key: str,
    ):
        """Send a comment to a Jira issue."""
        url = (
            f'{JIRA_CLOUD_API_URL}/{jira_cloud_id}/rest/api/2/issue/{issue_key}/comment'
        )
        data = {'body': message.message}
        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(
                url, auth=(svc_acc_email, svc_acc_api_key), json=data
            )
            response.raise_for_status()
            return response.json()

    async def _send_comment(self, view: JiraViewInterface, msg: str):
        """Send a comment using credentials from the view."""
        try:
            api_key = self.token_manager.decrypt_text(
                view.jira_workspace.svc_acc_api_key
            )
            await self.send_message(
                self.create_outgoing_message(msg=msg),
                issue_key=view.payload.issue_key,
                jira_cloud_id=view.jira_workspace.jira_cloud_id,
                svc_acc_email=view.jira_workspace.svc_acc_email,
                svc_acc_api_key=api_key,
            )
        except Exception as e:
            logger.error(
                '[Jira] Failed to send comment',
                extra={'issue_key': view.payload.issue_key, 'error': str(e)},
            )

    async def _send_error_from_payload(
        self,
        payload: JiraWebhookPayload,
        workspace: JiraWorkspace,
        error_msg: str,
    ):
        """Send error comment before view is created (using payload directly)."""
        try:
            api_key = self.token_manager.decrypt_text(workspace.svc_acc_api_key)
            await self.send_message(
                self.create_outgoing_message(msg=error_msg),
                issue_key=payload.issue_key,
                jira_cloud_id=workspace.jira_cloud_id,
                svc_acc_email=workspace.svc_acc_email,
                svc_acc_api_key=api_key,
            )
        except Exception as e:
            logger.error(
                '[Jira] Failed to send error comment',
                extra={'issue_key': payload.issue_key, 'error': str(e)},
            )

    def get_workspace_name_from_payload(self, payload: dict) -> str | None:
        """Extract workspace name from Jira webhook payload.

        This method is used by the route for signature verification.
        """
        parse_result = self.payload_parser.parse(payload)
        if isinstance(parse_result, JiraPayloadSuccess):
            return parse_result.payload.workspace_name
        return None
