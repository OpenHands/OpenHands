from types import MappingProxyType

from integrations.gitlab.gitlab_view import (
    GitlabFactory,
    GitlabInlineMRComment,
    GitlabIssue,
    GitlabIssueComment,
    GitlabMRComment,
    GitlabViewType,
)
from integrations.manager import Manager
from integrations.models import Message, SourceType
from integrations.types import ResolverViewInterface
from integrations.utils import (
    CONVERSATION_URL,
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
)
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from server.utils.conversation_callback_utils import register_callback_processor

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.gitlab.gitlab_service import GitLabServiceImpl
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.server.types import LLMAuthenticationError, MissingSettingsError
from openhands.storage.data_models.secrets import Secrets


class GitlabManager(Manager):
    def __init__(self, token_manager: TokenManager, data_collector: None = None):
        self.token_manager = token_manager

        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'gitlab')
        )

    def _confirm_incoming_source_type(self, message: Message):
        if message.source != SourceType.GITLAB:
            raise ValueError(f'Unexpected message source {message.source}')

    async def _user_has_write_access_to_repo(
        self, project_id: str, user_id: str
    ) -> bool:
        """
        Check if the user has write access to the repository (can pull/push changes and open merge requests).

        Args:
            project_id: The ID of the GitLab project
            username: The username of the user
            user_id: The GitLab user ID

        Returns:
            bool: True if the user has write access to the repository, False otherwise
        """

        keycloak_user_id = await self.token_manager.get_user_id_from_idp_user_id(
            user_id, ProviderType.GITLAB
        )
        if keycloak_user_id is None:
            logger.warning(f'Got invalid keyloak user id for GitLab User {user_id}')
            return False

        # Importing here prevents circular import
        from integrations.gitlab.gitlab_service import SaaSGitLabService

        gitlab_service: SaaSGitLabService = GitLabServiceImpl(
            external_auth_id=keycloak_user_id
        )

        return await gitlab_service.user_has_write_access(project_id)

    async def receive_message(self, message: Message):
        self._confirm_incoming_source_type(message)
        if await self.is_job_requested(message):
            gitlab_view = await GitlabFactory.create_gitlab_view_from_payload(
                message, self.token_manager
            )
            logger.info(
                f'[GitLab] Creating job for {gitlab_view.user_info.username} in {gitlab_view.full_repo_name}#{gitlab_view.issue_number}'
            )

            await self.start_job(gitlab_view)

    async def is_job_requested(self, message) -> bool:
        self._confirm_incoming_source_type(message)
        if not (
            GitlabFactory.is_labeled_issue(message)
            or GitlabFactory.is_issue_comment(message)
            or GitlabFactory.is_mr_comment(message)
            or GitlabFactory.is_mr_comment(message, inline=True)
        ):
            return False

        payload = message.message['payload']

        repo_obj = payload['project']
        project_id = repo_obj['id']
        selected_project = repo_obj['path_with_namespace']
        user = payload['user']
        user_id = user['id']
        username = user['username']

        logger.info(
            f'[GitLab] Checking permissions for {username} in {selected_project}'
        )

        has_write_access = await self._user_has_write_access_to_repo(
            project_id=str(project_id), user_id=user_id
        )

        logger.info(
            f'[GitLab]: {username} access in {selected_project}: {has_write_access}'
        )
        # Check if the user has write access to the repository
        return has_write_access

    async def send_message(self, message: Message, gitlab_view: ResolverViewInterface):
        """
        Send a message to GitLab based on the view type.

        Args:
            message: The message to send
            gitlab_view: The GitLab view object containing issue/PR/comment info
        """
        keycloak_user_id = gitlab_view.user_info.keycloak_user_id

        # Importing here prevents circular import
        from integrations.gitlab.gitlab_service import SaaSGitLabService

        gitlab_service: SaaSGitLabService = GitLabServiceImpl(
            external_auth_id=keycloak_user_id
        )

        outgoing_message = message.message

        if isinstance(gitlab_view, GitlabInlineMRComment) or isinstance(
            gitlab_view, GitlabMRComment
        ):
            await gitlab_service.reply_to_mr(
                gitlab_view.project_id,
                gitlab_view.issue_number,
                gitlab_view.discussion_id,
                message.message,
            )

        elif isinstance(gitlab_view, GitlabIssueComment):
            await gitlab_service.reply_to_issue(
                gitlab_view.project_id,
                gitlab_view.issue_number,
                gitlab_view.discussion_id,
                outgoing_message,
            )
        elif isinstance(gitlab_view, GitlabIssue):
            await gitlab_service.reply_to_issue(
                gitlab_view.project_id,
                gitlab_view.issue_number,
                None,  # no discussion id, issue is tagged
                outgoing_message,
            )
        else:
            logger.warning(
                f'[GitLab] Unsupported view type: {type(gitlab_view).__name__}'
            )

    async def start_job(self, gitlab_view: GitlabViewType):
        """
        Start a job for the GitLab view.

        Args:
            gitlab_view: The GitLab view object containing issue/PR/comment info
        """
        # Importing here prevents circular import
        from server.conversation_callback_processor.gitlab_callback_processor import (
            GitlabCallbackProcessor,
        )

        try:
            try:
                user_info = gitlab_view.user_info

                logger.info(
                    f'[GitLab] Starting job for {user_info.username} in {gitlab_view.full_repo_name}#{gitlab_view.issue_number}'
                )

                user_token = await self.token_manager.get_idp_token_from_idp_user_id(
                    str(user_info.user_id), ProviderType.GITLAB
                )

                if not user_token:
                    logger.warning(
                        f'[GitLab] No token found for user {user_info.username} (id={user_info.user_id})'
                    )
                    raise MissingSettingsError('Missing settings')

                logger.info(
                    f'[GitLab] Creating new conversation for user {user_info.username}'
                )

                secret_store = Secrets(
                    provider_tokens=MappingProxyType(
                        {
                            ProviderType.GITLAB: ProviderToken(
                                token=SecretStr(user_token),
                                user_id=str(user_info.user_id),
                            )
                        }
                    )
                )

                await gitlab_view.create_new_conversation(
                    self.jinja_env, secret_store.provider_tokens
                )

                conversation_id = gitlab_view.conversation_id

                logger.info(
                    f'[GitLab] Created conversation {conversation_id} for user {user_info.username}'
                )

                # Create a GitlabCallbackProcessor for this conversation
                processor = GitlabCallbackProcessor(
                    gitlab_view=gitlab_view,
                    send_summary_instruction=True,
                )

                # Register the callback processor
                register_callback_processor(conversation_id, processor)

                logger.info(
                    f'[GitLab] Created callback processor for conversation {conversation_id}'
                )

                conversation_link = CONVERSATION_URL.format(conversation_id)
                msg_info = f"I'm on it! {user_info.username} can [track my progress at all-hands.dev]({conversation_link})"

            except MissingSettingsError as e:
                logger.warning(
                    f'[GitLab] Missing settings error for user {user_info.username}: {str(e)}'
                )

                msg_info = f'@{user_info.username} please re-login into [OpenHands Cloud]({HOST_URL}) before starting a job.'

            except LLMAuthenticationError as e:
                logger.warning(
                    f'[GitLab] LLM authentication error for user {user_info.username}: {str(e)}'
                )

                msg_info = f'@{user_info.username} please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

            # Send the acknowledgment message
            msg = self.create_outgoing_message(msg_info)
            await self.send_message(msg, gitlab_view)

        except Exception as e:
            logger.exception(f'[GitLab] Error starting job: {str(e)}')
            msg = self.create_outgoing_message(
                msg='Uh oh! There was an unexpected error starting the job :('
            )
            await self.send_message(msg, gitlab_view)
