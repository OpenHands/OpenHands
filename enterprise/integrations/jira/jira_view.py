"""Jira view implementations and factory.

Views are responsible for:
- Holding the webhook payload and auth context
- Lazy-loading issue details from Jira API when needed
- Creating conversations with the selected repository
"""

from dataclasses import dataclass, field

import httpx
from integrations.jira.jira_payload import JiraWebhookPayload
from integrations.jira.jira_types import (
    JiraViewInterface,
    RepositoryNotFoundError,
    StartingConvoException,
)
from integrations.utils import CONVERSATION_URL, infer_repo_from_message
from jinja2 import Environment
from storage.jira_conversation import JiraConversation
from storage.jira_integration_store import JiraIntegrationStore
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler
from openhands.server.services.conversation_service import create_new_conversation
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import ConversationTrigger
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
    selected_repo: str | None = None
    conversation_id: str = ''

    # Lazy-loaded issue details (cached after first fetch)
    _issue_title: str | None = field(default=None, repr=False)
    _issue_description: str | None = field(default=None, repr=False)

    # Decrypted API key (set by factory)
    _decrypted_api_key: str = field(default='', repr=False)

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

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        user_secrets = await self.saas_user_auth.get_secrets()
        instructions, user_msg = await self._get_instructions(jinja_env)

        try:
            agent_loop_info = await create_new_conversation(
                user_id=self.jira_user.keycloak_user_id,
                git_provider_tokens=provider_tokens,
                selected_repository=self.selected_repo,
                selected_branch=None,
                initial_user_msg=user_msg,
                conversation_instructions=instructions,
                image_urls=None,
                replay_json=None,
                conversation_trigger=ConversationTrigger.JIRA,
                custom_secrets=user_secrets.custom_secrets if user_secrets else None,
            )

            self.conversation_id = agent_loop_info.conversation_id

            logger.info(
                '[Jira] Created conversation',
                extra={
                    'conversation_id': self.conversation_id,
                    'issue_key': self.payload.issue_key,
                    'selected_repo': self.selected_repo,
                },
            )

            # Store Jira conversation mapping
            jira_conversation = JiraConversation(
                conversation_id=self.conversation_id,
                issue_id=self.payload.issue_id,
                issue_key=self.payload.issue_key,
                jira_user_id=self.jira_user.id,
            )

            await integration_store.create_conversation(jira_conversation)

            return self.conversation_id

        except Exception as e:
            if isinstance(e, StartingConvoException):
                raise
            logger.error(
                '[Jira] Failed to create conversation',
                extra={'issue_key': self.payload.issue_key, 'error': str(e)},
                exc_info=True,
            )
            raise StartingConvoException(f'Failed to create conversation: {str(e)}')

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
