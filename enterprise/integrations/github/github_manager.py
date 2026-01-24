from types import MappingProxyType

from github import Auth, Github, GithubIntegration
from integrations.github.data_collector import GitHubDataCollector
from integrations.github.github_solvability import summarize_issue_solvability
from integrations.github.github_view import (
    GithubFactory,
    GithubFailingAction,
    GithubInlinePRComment,
    GithubIssue,
    GithubIssueComment,
    GithubPRComment,
)
from integrations.manager import Manager
from integrations.models import (
    Message,
    SourceType,
)
from integrations.types import ResolverViewInterface
from integrations.utils import (
    CONVERSATION_URL,
    HOST_URL,
    OPENHANDS_RESOLVER_TEMPLATES_DIR,
    get_session_expired_message,
)
from integrations.v1_utils import get_saas_user_auth
from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr
from server.auth.auth_error import ExpiredError
from server.auth.constants import GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY
from server.auth.token_manager import TokenManager
from server.utils.conversation_callback_utils import register_callback_processor

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)
from openhands.storage.data_models.secrets import Secrets
from openhands.utils.async_utils import call_sync_from_async


class GithubManager(Manager):
    def __init__(
        self, token_manager: TokenManager, data_collector: GitHubDataCollector
    ):
        self.token_manager = token_manager
        self.data_collector = data_collector
        self.github_integration = GithubIntegration(
            auth=Auth.AppAuth(GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY)
        )

        self.jinja_env = Environment(
            loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR + 'github')
        )

    def _confirm_incoming_source_type(self, message: Message):
        if message.source != SourceType.GITHUB:
            raise ValueError(f'Unexpected message source {message.source}')

    def _get_full_repo_name(self, repo_obj: dict) -> str:
        owner = repo_obj['owner']['login']
        repo_name = repo_obj['name']

        return f'{owner}/{repo_name}'

    def _get_installation_access_token(self, installation_id: str) -> str:
        # get_access_token is typed to only accept int, but it can handle str.
        token_data = self.github_integration.get_access_token(
            installation_id  # type: ignore[arg-type]
        )
        return token_data.token

    def _add_reaction(
        self, github_view: ResolverViewInterface, reaction: str, installation_token: str
    ):
        """Add a reaction to the GitHub issue, PR, or comment.

        Args:
            github_view: The GitHub view object containing issue/PR/comment info
            reaction: The reaction to add (e.g. "eyes", "+1", "-1", "laugh", "confused", "heart", "hooray", "rocket")
            installation_token: GitHub installation access token for API access
        """
        with Github(auth=Auth.Token(installation_token)) as github_client:
            repo = github_client.get_repo(github_view.full_repo_name)
            # Add reaction based on view type
            if isinstance(github_view, GithubInlinePRComment):
                pr = repo.get_pull(github_view.issue_number)
                inline_comment = pr.get_review_comment(github_view.comment_id)
                inline_comment.create_reaction(reaction)

            elif isinstance(github_view, (GithubIssueComment, GithubPRComment)):
                issue = repo.get_issue(github_view.issue_number)
                comment = issue.get_comment(github_view.comment_id)
                comment.create_reaction(reaction)
            else:
                issue = repo.get_issue(github_view.issue_number)
                issue.create_reaction(reaction)

    def _user_has_write_access_to_repo(
        self, installation_id: str, full_repo_name: str, username: str
    ) -> bool:
        """Check if the user is an owner, collaborator, or member of the repository."""
        with self.github_integration.get_github_for_installation(
            installation_id,  # type: ignore[arg-type]
            {},
        ) as repos:
            repository = repos.get_repo(full_repo_name)

            # Check if the user is a collaborator
            try:
                collaborator = repository.get_collaborator_permission(username)
                if collaborator in ['admin', 'write']:
                    return True
            except Exception:
                pass

            # If the above fails, check if the user is an owner or member
            org = repository.organization
            if org:
                user = org.get_members(username)
                return user is not None

            return False

    async def is_job_requested(self, message: Message) -> bool:
        self._confirm_incoming_source_type(message)

        installation_id = message.message['installation']
        payload = message.message.get('payload', {})
        repo_obj = payload.get('repository')
        if not repo_obj:
            return False
        username = payload.get('sender', {}).get('login')
        repo_name = self._get_full_repo_name(repo_obj)

        # Suggestions contain `@openhands` macro; avoid kicking off jobs for system recommendations
        if GithubFactory.is_pr_comment(
            message
        ) and GithubFailingAction.unqiue_suggestions_header in payload.get(
            'comment', {}
        ).get('body', ''):
            return False

        if GithubFactory.is_eligible_for_conversation_starter(
            message
        ) and self._user_has_write_access_to_repo(installation_id, repo_name, username):
            await GithubFactory.trigger_conversation_starter(message)

        if not (
            GithubFactory.is_labeled_issue(message)
            or GithubFactory.is_issue_comment(message)
            or GithubFactory.is_pr_comment(message)
            or GithubFactory.is_inline_pr_comment(message)
        ):
            return False

        logger.info(f'[GitHub] Checking permissions for {username} in {repo_name}')

        return self._user_has_write_access_to_repo(installation_id, repo_name, username)

    async def receive_message(self, message: Message):
        self._confirm_incoming_source_type(message)
        try:
            await call_sync_from_async(self.data_collector.process_payload, message)
        except Exception:
            logger.warning(
                '[Github]: Error processing payload for gh interaction', exc_info=True
            )

        if await self.is_job_requested(message):
            payload = message.message.get('payload', {})
            user_id = payload['sender']['id']
            keycloak_user_id = await self.token_manager.get_user_id_from_idp_user_id(
                user_id, ProviderType.GITHUB
            )
            github_view = await GithubFactory.create_github_view_from_payload(
                message, keycloak_user_id
            )
            logger.info(
                f'[GitHub] Creating job for {github_view.user_info.username} in {github_view.full_repo_name}#{github_view.issue_number}'
            )
            # Get the installation token
            installation_token = self._get_installation_access_token(
                github_view.installation_id
            )
            # Store the installation token
            self.token_manager.store_org_token(
                github_view.installation_id, installation_token
            )
            # Add eyes reaction to acknowledge we've read the request
            self._add_reaction(github_view, 'eyes', installation_token)
            await self.start_job(github_view)

    async def send_message(self, message: Message, github_view: ResolverViewInterface):
        installation_token = self.token_manager.load_org_token(
            github_view.installation_id
        )
        if not installation_token:
            logger.warning('Missing installation token')
            return

        outgoing_message = message.message

        if isinstance(github_view, GithubInlinePRComment):
            with Github(auth=Auth.Token(installation_token)) as github_client:
                repo = github_client.get_repo(github_view.full_repo_name)
                pr = repo.get_pull(github_view.issue_number)
                pr.create_review_comment_reply(
                    comment_id=github_view.comment_id, body=outgoing_message
                )

        elif (
            isinstance(github_view, GithubPRComment)
            or isinstance(github_view, GithubIssueComment)
            or isinstance(github_view, GithubIssue)
        ):
            with Github(auth=Auth.Token(installation_token)) as github_client:
                repo = github_client.get_repo(github_view.full_repo_name)
                issue = repo.get_issue(number=github_view.issue_number)
                issue.create_comment(outgoing_message)

        else:
            logger.warning('Unsupported location')
            return

    async def start_job(self, github_view: ResolverViewInterface):
        """Kick off a job with openhands agent.

        1. Get user credential
        2. Initialize new conversation with repo
        3. Save interaction data
        """
        # Importing here prevents circular import
        from server.conversation_callback_processor.github_callback_processor import (
            GithubCallbackProcessor,
        )

        try:
            msg_info = None

            try:
                user_info = github_view.user_info
                logger.info(
                    f'[GitHub] Starting job for user {user_info.username} (id={user_info.user_id})'
                )

                # Create conversation
                user_token = await self.token_manager.get_idp_token_from_idp_user_id(
                    str(user_info.user_id), ProviderType.GITHUB
                )

                if not user_token:
                    logger.warning(
                        f'[GitHub] No token found for user {user_info.username} (id={user_info.user_id})'
                    )
                    raise MissingSettingsError('Missing settings')

                logger.info(
                    f'[GitHub] Creating new conversation for user {user_info.username}'
                )

                secret_store = Secrets(
                    provider_tokens=MappingProxyType(
                        {
                            ProviderType.GITHUB: ProviderToken(
                                token=SecretStr(user_token),
                                user_id=str(user_info.user_id),
                            )
                        }
                    )
                )

                # We first initialize a conversation and generate the solvability report BEFORE starting the conversation runtime
                # This helps us accumulate llm spend without requiring a running runtime. This setups us up for
                #   1. If there is a problem starting the runtime we still have accumulated total conversation cost
                #   2. In the future, based on the report confidence we can conditionally start the conversation
                #   3. Once the conversation is started, its base cost will include the report's spend as well which allows us to control max budget per resolver task
                convo_metadata = await github_view.initialize_new_conversation()
                solvability_summary = None
                try:
                    if user_token:
                        solvability_summary = await summarize_issue_solvability(
                            github_view, user_token
                        )
                    else:
                        logger.warning(
                            '[Github]: No user token available for solvability analysis'
                        )
                except Exception as e:
                    logger.warning(
                        f'[Github]: Error summarizing issue solvability: {str(e)}'
                    )

                saas_user_auth = await get_saas_user_auth(
                    github_view.user_info.keycloak_user_id, self.token_manager
                )

                await github_view.create_new_conversation(
                    self.jinja_env,
                    secret_store.provider_tokens,
                    convo_metadata,
                    saas_user_auth,
                )

                conversation_id = github_view.conversation_id

                logger.info(
                    f'[GitHub] Created conversation {conversation_id} for user {user_info.username}'
                )

                if not github_view.v1_enabled:
                    # Create a GithubCallbackProcessor
                    processor = GithubCallbackProcessor(
                        github_view=github_view,
                        send_summary_instruction=True,
                    )

                    # Register the callback processor
                    register_callback_processor(conversation_id, processor)

                    logger.info(
                        f'[Github] Registered callback processor for conversation {conversation_id}'
                    )

                # Send message with conversation link
                conversation_link = CONVERSATION_URL.format(conversation_id)
                base_msg = f"I'm on it! {user_info.username} can [track my progress at all-hands.dev]({conversation_link})"
                # Combine messages: include solvability report with "I'm on it!" if successful
                if solvability_summary:
                    msg_info = f'{base_msg}\n\n{solvability_summary}'
                else:
                    msg_info = base_msg

            except MissingSettingsError as e:
                logger.warning(
                    f'[GitHub] Missing settings error for user {user_info.username}: {str(e)}'
                )

                msg_info = f'@{user_info.username} please re-login into [OpenHands Cloud]({HOST_URL}) before starting a job.'

            except LLMAuthenticationError as e:
                logger.warning(
                    f'[GitHub] LLM authentication error for user {user_info.username}: {str(e)}'
                )

                msg_info = f'@{user_info.username} please set a valid LLM API key in [OpenHands Cloud]({HOST_URL}) before starting a job.'

            except (ExpiredError, SessionExpiredError) as e:
                logger.warning(
                    f'[GitHub] Session expired for user {user_info.username}: {str(e)}'
                )

                msg_info = get_session_expired_message(user_info.username)

            msg = self.create_outgoing_message(msg_info)
            await self.send_message(msg, github_view)

        except Exception:
            logger.exception('[Github]: Error starting job')
            msg = self.create_outgoing_message(
                msg='Uh oh! There was an unexpected error starting the job :('
            )
            await self.send_message(msg, github_view)

        try:
            await self.data_collector.save_data(github_view)
        except Exception:
            logger.warning('[Github]: Error saving interaction data', exc_info=True)
