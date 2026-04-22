"""Jira view implementations and factory.

Views are responsible for:
- Holding the webhook payload and auth context
- Lazy-loading issue details from Jira API when needed
- Creating conversations with the selected repository
"""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import httpx
from integrations.jira.jira_payload import JiraWebhookPayload
from integrations.jira.jira_types import (
    JiraViewInterface,
    RepositoryNotFoundError,
    StartingConvoException,
)
from integrations.jira.jira_v1_callback_processor import (
    JiraV1CallbackProcessor,
)
from integrations.resolver_context import ResolverUserContext
from integrations.resolver_org_router import resolve_org_for_repo
from integrations.utils import (
    CONVERSATION_URL,
    infer_repo_from_message,
)
from jinja2 import Environment
from storage.jira_conversation import JiraConversation
from storage.jira_integration_store import JiraIntegrationStore
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace

from openhands.agent_server.models import SendMessageRequest
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler, ProviderType
from openhands.sdk import TextContent
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.utils.http_session import httpx_verify_option

JIRA_CLOUD_API_URL = 'https://api.atlassian.com/ex/jira'

integration_store = JiraIntegrationStore.get_instance()


@dataclass
class JiraNewConversationView(JiraViewInterface):
    """View for creating a new Jira conversation.

    This view holds the webhook payload directly and lazily fetches
    issue details when needed for rendering templates.
    """

    payload: JiraWebhookPayload
    saas_user_auth: UserAuth
    jira_user: JiraUser
    jira_workspace: JiraWorkspace
    selected_repo: str = ''
    conversation_id: str = ''

    # Lazy-loaded issue details (cached after first fetch)
    _issue_title: str | None = field(default=None, repr=False)
    _issue_description: str | None = field(default=None, repr=False)

    # Decrypted API key (set by factory)
    _decrypted_api_key: str = field(default='', repr=False)

    # Resolved org ID for V1 conversations
    resolved_org_id: UUID | None = None

    async def get_issue_details(self) -> tuple[str, str]:
        """Fetch issue details from Jira API (cached after first call).

        Returns:
            Tuple of (issue_title, issue_description)

        Raises:
            StartingConvoException: If issue details cannot be fetched
        """
        if self._issue_title is not None and self._issue_description is not None:
            return self._issue_title, self._issue_description

        try:
            url = f'{JIRA_CLOUD_API_URL}/{self.jira_workspace.jira_cloud_id}/rest/api/2/issue/{self.payload.issue_key}'
            async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
                response = await client.get(
                    url,
                    auth=(
                        self.jira_workspace.svc_acc_email,
                        self._decrypted_api_key,
                    ),
                )
                response.raise_for_status()
                issue_payload = response.json()

            if not issue_payload:
                raise StartingConvoException(
                    f'Issue {self.payload.issue_key} not found.'
                )

            self._issue_title = issue_payload.get('fields', {}).get('summary', '')
            self._issue_description = (
                issue_payload.get('fields', {}).get('description', '') or ''
            )

            if not self._issue_title:
                raise StartingConvoException(
                    f'Issue {self.payload.issue_key} does not have a title.'
                )

            logger.info(
                '[Jira] Fetched issue details',
                extra={
                    'issue_key': self.payload.issue_key,
                    'has_description': bool(self._issue_description),
                },
            )

            return self._issue_title, self._issue_description

        except httpx.HTTPStatusError as e:
            logger.error(
                '[Jira] Failed to fetch issue details',
                extra={
                    'issue_key': self.payload.issue_key,
                    'status': e.response.status_code,
                },
            )
            raise StartingConvoException(
                f'Failed to fetch issue details: HTTP {e.response.status_code}'
            )
        except Exception as e:
            if isinstance(e, StartingConvoException):
                raise
            logger.error(
                '[Jira] Failed to fetch issue details',
                extra={'issue_key': self.payload.issue_key, 'error': str(e)},
            )
            raise StartingConvoException(f'Failed to fetch issue details: {str(e)}')

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Get instructions for the conversation.

        This fetches issue details if not already cached.

        Returns:
            Tuple of (system_instructions, user_message)
        """
        issue_title, issue_description = await self.get_issue_details()

        instructions_template = jinja_env.get_template('jira_instructions.j2')
        instructions = instructions_template.render()

        user_msg_template = jinja_env.get_template('jira_new_conversation.j2')
        user_msg = user_msg_template.render(
            issue_key=self.payload.issue_key,
            issue_title=issue_title,
            issue_description=issue_description,
            user_message=self.payload.user_msg,
        )

        return instructions, user_msg

    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Create a new Jira conversation.

        Returns:
            The conversation ID

        Raises:
            StartingConvoException: If conversation creation fails
        """
        if not self.selected_repo:
            raise StartingConvoException('No repository selected for this conversation')

        jira_conversation = JiraConversation(
            conversation_id=self.conversation_id,
            issue_id=self.payload.issue_id,
            issue_key=self.payload.issue_key,
            jira_user_id=self.jira_user.id,
        )
        await integration_store.create_conversation(jira_conversation)

        conversation_metadata = await self._create_v1_metadata()
        await self._create_v1_conversation(jinja_env, conversation_metadata)
        return self.conversation_id

    async def _create_v1_metadata(self) -> ConversationMetadata:
        """Create conversation metadata for V1 conversations.

        The JiraConversation mapping is saved to the integration store (above), but
        V1 conversation metadata is managed by the app conversation system, not
        the legacy conversation store.
        """
        logger.info('[Jira]: Creating V1 metadata')

        # Generate a dummy conversation for V1 (not saved to store)
        self.conversation_id = uuid4().hex
        self.resolved_org_id = await self._get_resolved_org_id()

        return ConversationMetadata(
            conversation_id=self.conversation_id,
            selected_repository=self.selected_repo,
        )

    async def _create_v1_conversation(
        self,
        jinja_env: Environment,
        conversation_metadata: ConversationMetadata,
    ):
        """Create conversation using the new V1 app conversation system."""
        logger.info('[Jira]: Creating V1 conversation')

        initial_user_text = await self._get_v1_initial_user_message(jinja_env)

        # Create the initial message request
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=initial_user_text)]
        )

        # Create the Jira V1 callback processor
        jira_callback_processor = self._create_jira_v1_callback_processor()

        injector_state = InjectorState()

        # Create the V1 conversation start request
        start_request = AppConversationStartRequest(
            conversation_id=UUID(conversation_metadata.conversation_id),
            system_message_suffix=None,
            initial_message=initial_message,
            selected_repository=self.selected_repo,
            selected_branch=None,
            git_provider=ProviderType.GITHUB,
            title=f'Jira Issue {self.payload.issue_key}: {self._issue_title or "Unknown"}',
            trigger=ConversationTrigger.JIRA,
            processors=[jira_callback_processor],
        )

        # Set up the Jira user context for the V1 system
        jira_user_context = ResolverUserContext(
            saas_user_auth=self.saas_user_auth,
            resolver_org_id=self.resolved_org_id,
        )
        setattr(injector_state, USER_CONTEXT_ATTR, jira_user_context)

        async with get_app_conversation_service(
            injector_state
        ) as app_conversation_service:
            async for task in app_conversation_service.start_app_conversation(
                start_request
            ):
                if task.status == AppConversationStartTaskStatus.ERROR:
                    logger.error(f'Failed to start V1 conversation: {task.detail}')
                    raise RuntimeError(
                        f'Failed to start V1 conversation: {task.detail}'
                    )

    async def _get_v1_initial_user_message(self, jinja_env: Environment) -> str:
        """Build the initial user message for V1 resolver conversations."""
        issue_title, issue_description = await self.get_issue_details()

        user_msg_template = jinja_env.get_template('jira_new_conversation.j2')
        user_msg = user_msg_template.render(
            issue_key=self.payload.issue_key,
            issue_title=issue_title,
            issue_description=issue_description,
            user_message=self.payload.user_msg,
        )

        return user_msg

    def _create_jira_v1_callback_processor(self):
        """Create a V1 callback processor for Jira integration."""
        return JiraV1CallbackProcessor(
            svc_acc_email=self.jira_workspace.svc_acc_email,
            decrypted_api_key=self._decrypted_api_key,
            issue_key=self.payload.issue_key,
            jira_cloud_id=self.jira_workspace.jira_cloud_id,
        )

    async def _get_resolved_org_id(self) -> UUID | None:
        """Resolve the org ID for V1 conversations."""
        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        if not provider_tokens:
            return None

        try:
            provider_handler = ProviderHandler(provider_tokens)
            repository = await provider_handler.verify_repo_provider(self.selected_repo)
            resolved_org_id = await resolve_org_for_repo(
                provider=repository.git_provider.value,
                full_repo_name=self.selected_repo,
                keycloak_user_id=self.jira_user.keycloak_user_id,
            )
            return resolved_org_id
        except Exception as e:
            logger.warning(
                f'[Jira] Failed to resolve org for {self.selected_repo}: {e}'
            )
            return None

    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira."""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.payload.display_name} can [track my progress here|{conversation_link}]."


class JiraFactory:
    """Factory for creating Jira views.

    The factory is responsible for:
    - Creating the appropriate view type
    - Inferring and selecting the repository
    - Validating all required data is available

    Repository selection happens here so that view creation either
    succeeds with a valid repo or fails with a clear error.
    """

    @staticmethod
    async def _create_provider_handler(user_auth: UserAuth) -> ProviderHandler | None:
        """Create a ProviderHandler for the user."""
        provider_tokens = await user_auth.get_provider_tokens()
        if provider_tokens is None:
            return None

        access_token = await user_auth.get_access_token()
        user_id = await user_auth.get_user_id()

        return ProviderHandler(
            provider_tokens=provider_tokens,
            external_auth_token=access_token,
            external_auth_id=user_id,
        )

    @staticmethod
    def _extract_potential_repos(
        issue_key: str,
        issue_title: str,
        issue_description: str,
        user_msg: str,
    ) -> list[str]:
        """Extract potential repository names from issue content.

        Raises:
            RepositoryNotFoundError: If no potential repos found in text.
        """
        search_text = f'{issue_title}\n{issue_description}\n{user_msg}'
        potential_repos = infer_repo_from_message(search_text)

        if not potential_repos:
            raise RepositoryNotFoundError(
                'Could not determine which repository to use. '
                'Please mention the repository (e.g., owner/repo) in the issue description or comment.'
            )

        logger.info(
            '[Jira] Found potential repositories in issue content',
            extra={'issue_key': issue_key, 'potential_repos': potential_repos},
        )
        return potential_repos

    @staticmethod
    async def _verify_repos(
        issue_key: str,
        potential_repos: list[str],
        provider_handler: ProviderHandler,
    ) -> list[str]:
        """Verify which repos the user has access to."""
        verified_repos: list[str] = []

        for repo_name in potential_repos:
            try:
                repository = await provider_handler.verify_repo_provider(repo_name)
                verified_repos.append(repository.full_name)
                logger.debug(
                    '[Jira] Repository verification succeeded',
                    extra={'issue_key': issue_key, 'repository': repository.full_name},
                )
            except Exception as e:
                logger.debug(
                    '[Jira] Repository verification failed',
                    extra={
                        'issue_key': issue_key,
                        'repo_name': repo_name,
                        'error': str(e),
                    },
                )

        return verified_repos

    @staticmethod
    def _select_single_repo(
        issue_key: str,
        potential_repos: list[str],
        verified_repos: list[str],
    ) -> str:
        """Select exactly one repo from verified repos.

        Raises:
            RepositoryNotFoundError: If zero or multiple repos verified.
        """
        if len(verified_repos) == 0:
            raise RepositoryNotFoundError(
                f'Could not access any of the mentioned repositories: {", ".join(potential_repos)}. '
                'Please ensure you have access to the repository and it exists.'
            )

        if len(verified_repos) > 1:
            raise RepositoryNotFoundError(
                f'Multiple repositories found: {", ".join(verified_repos)}. '
                'Please specify exactly one repository in the issue description or comment.'
            )

        logger.info(
            '[Jira] Verified repository access',
            extra={'issue_key': issue_key, 'repository': verified_repos[0]},
        )
        return verified_repos[0]

    @staticmethod
    async def _infer_repository(
        payload: JiraWebhookPayload,
        user_auth: UserAuth,
        issue_title: str,
        issue_description: str,
    ) -> str:
        """Infer and verify the repository from issue content.

        Raises:
            RepositoryNotFoundError: If no valid repository can be determined.
        """
        provider_handler = await JiraFactory._create_provider_handler(user_auth)
        if not provider_handler:
            raise RepositoryNotFoundError(
                'No Git provider connected. Please connect a Git provider in OpenHands settings.'
            )

        potential_repos = JiraFactory._extract_potential_repos(
            payload.issue_key, issue_title, issue_description, payload.user_msg
        )

        verified_repos = await JiraFactory._verify_repos(
            payload.issue_key, potential_repos, provider_handler
        )

        return JiraFactory._select_single_repo(
            payload.issue_key, potential_repos, verified_repos
        )

    @staticmethod
    async def create_view(
        payload: JiraWebhookPayload,
        workspace: JiraWorkspace,
        user: JiraUser,
        user_auth: UserAuth,
        decrypted_api_key: str,
    ) -> JiraViewInterface:
        """Create a Jira view with repository already selected.

        This factory method:
        1. Creates the view with payload and auth context
        2. Fetches issue details (needed for repo inference)
        3. Infers and selects the repository

        If any step fails, an appropriate exception is raised with
        a user-friendly message.

        Args:
            payload: Parsed webhook payload
            workspace: The Jira workspace
            user: The Jira user
            user_auth: OpenHands user authentication
            decrypted_api_key: Decrypted service account API key

        Returns:
            A JiraViewInterface with selected_repo populated

        Raises:
            StartingConvoException: If view creation fails
            RepositoryNotFoundError: If repository cannot be determined
        """
        logger.info(
            '[Jira] Creating view',
            extra={
                'issue_key': payload.issue_key,
                'event_type': payload.event_type.value,
            },
        )

        # Create the view
        view = JiraNewConversationView(
            payload=payload,
            saas_user_auth=user_auth,
            jira_user=user,
            jira_workspace=workspace,
            _decrypted_api_key=decrypted_api_key,
        )

        # Fetch issue details (needed for repo inference)
        try:
            issue_title, issue_description = await view.get_issue_details()
        except StartingConvoException:
            raise  # Re-raise with original message
        except Exception as e:
            raise StartingConvoException(f'Failed to fetch issue details: {str(e)}')

        # Infer and select repository
        selected_repo = await JiraFactory._infer_repository(
            payload=payload,
            user_auth=user_auth,
            issue_title=issue_title,
            issue_description=issue_description,
        )

        view.selected_repo = selected_repo

        logger.info(
            '[Jira] View created successfully',
            extra={
                'issue_key': payload.issue_key,
                'selected_repo': selected_repo,
            },
        )

        return view
