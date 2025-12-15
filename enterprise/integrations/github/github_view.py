from dataclasses import dataclass
from uuid import UUID, uuid4

from github import Github, GithubIntegration
from github.Issue import Issue
from integrations.github.github_types import (
    WorkflowRun,
    WorkflowRunGroup,
    WorkflowRunStatus,
)
from integrations.models import Message
from integrations.resolver_context import ResolverUserContext
from integrations.types import ResolverViewInterface, UserData
from integrations.utils import (
    ENABLE_PROACTIVE_CONVERSATION_STARTERS,
    ENABLE_V1_GITHUB_RESOLVER,
    HOST,
    HOST_URL,
    get_oh_labels,
    has_exact_mention,
)
from jinja2 import Environment
from server.auth.constants import GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY
from server.auth.token_manager import TokenManager
from server.config import get_config
from storage.database import session_maker
from storage.proactive_conversation_store import ProactiveConversationStore
from storage.saas_secrets_store import SaasSecretsStore
from storage.saas_settings_store import SaasSettingsStore

from openhands.agent_server.models import SendMessageRequest
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.github_service import GithubServiceImpl
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.integrations.service_types import Comment
from openhands.sdk import TextContent
from openhands.server.services.conversation_service import (
    initialize_conversation,
    start_conversation,
)
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.utils.async_utils import call_sync_from_async

OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)


async def get_user_proactive_conversation_setting(user_id: str | None) -> bool:
    """Get the user's proactive conversation setting.

    Args:
        user_id: The keycloak user ID

    Returns:
        True if proactive conversations are enabled for this user, False otherwise

    Note:
        This function checks both the global environment variable kill switch AND
        the user's individual setting. Both must be true for the function to return true.
    """

    # If no user ID is provided, we can't check user settings
    if not user_id:
        return False

    config = get_config()
    settings_store = SaasSettingsStore(
        user_id=user_id, session_maker=session_maker, config=config
    )

    settings = await call_sync_from_async(
        settings_store.get_user_settings_by_keycloak_id, user_id
    )

    if not settings or settings.enable_proactive_conversation_starters is None:
        return False

    return settings.enable_proactive_conversation_starters


async def get_user_v1_enabled_setting(user_id: str) -> bool:
    """Get the user's V1 conversation API setting.

    Args:
        user_id: The keycloak user ID

    Returns:
        True if V1 conversations are enabled for this user, False otherwise

    Note:
        This function checks both the global environment variable kill switch AND
        the user's individual setting. Both must be true for the function to return true.
    """
    # Check the global environment variable first
    if not ENABLE_V1_GITHUB_RESOLVER:
        return False

    config = get_config()
    settings_store = SaasSettingsStore(
        user_id=user_id, session_maker=session_maker, config=config
    )

    settings = await call_sync_from_async(
        settings_store.get_user_settings_by_keycloak_id, user_id
    )

    if not settings or settings.v1_enabled is None:
        return False

    return settings.v1_enabled


# =================================================
# SECTION: Github view types
# =================================================


@dataclass
class GithubIssue(ResolverViewInterface):
    issue_number: int
    installation_id: int
    full_repo_name: str
    is_public_repo: bool
    user_info: UserData
    raw_payload: Message
    conversation_id: str
    uuid: str | None
    should_extract: bool
    send_summary_instruction: bool
    title: str
    description: str
    previous_comments: list[Comment]
    v1: bool

    async def _load_resolver_context(self):
        github_service = GithubServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )

        self.previous_comments = await github_service.get_issue_or_pr_comments(
            self.full_repo_name, self.issue_number
        )

        (
            self.title,
            self.description,
        ) = await github_service.get_issue_or_pr_title_and_body(
            self.full_repo_name, self.issue_number
        )

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        user_instructions_template = jinja_env.get_template('issue_prompt.j2')

        user_instructions = user_instructions_template.render(
            issue_number=self.issue_number,
        )

        await self._load_resolver_context()

        conversation_instructions_template = jinja_env.get_template(
            'issue_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            issue_title=self.title,
            issue_body=self.description,
            previous_comments=self.previous_comments,
        )
        return user_instructions, conversation_instructions

    async def _get_user_secrets(self):
        secrets_store = SaasSecretsStore(
            self.user_info.keycloak_user_id, session_maker, get_config()
        )
        user_secrets = await secrets_store.load()

        return user_secrets.custom_secrets if user_secrets else None

    async def initialize_new_conversation(self) -> ConversationMetadata:
        # FIXME: Handle if initialize_conversation returns None

        v1_enabled = await get_user_v1_enabled_setting(self.user_info.keycloak_user_id)
        logger.info(
            f'[GitHub V1]: User flag found for {self.user_info.keycloak_user_id} is {v1_enabled}'
        )
        if v1_enabled:
            # Create dummy conversationm metadata
            # Don't save to conversation store
            # V1 conversations are stored in a separate table
            return ConversationMetadata(
                conversation_id=uuid4().hex, selected_repository=self.full_repo_name
            )

        conversation_metadata: ConversationMetadata = await initialize_conversation(  # type: ignore[assignment]
            user_id=self.user_info.keycloak_user_id,
            conversation_id=None,
            selected_repository=self.full_repo_name,
            selected_branch=None,
            conversation_trigger=ConversationTrigger.RESOLVER,
            git_provider=ProviderType.GITHUB,
        )
        self.conversation_id = conversation_metadata.conversation_id
        return conversation_metadata

    async def create_new_conversation(
        self,
        jinja_env: Environment,
        git_provider_tokens: PROVIDER_TOKEN_TYPE,
        conversation_metadata: ConversationMetadata,
        saas_user_auth: UserAuth,
    ):
        v1_enabled = await get_user_v1_enabled_setting(self.user_info.keycloak_user_id)
        logger.info(
            f'[GitHub V1]: User flag found for {self.user_info.keycloak_user_id} is {v1_enabled}'
        )
        if v1_enabled:
            try:
                # Use V1 app conversation service
                await self._create_v1_conversation(
                    jinja_env, saas_user_auth, conversation_metadata
                )
                return

            except Exception as e:
                logger.warning(f'Error checking V1 settings, falling back to V0: {e}')

        # Use existing V0 conversation service
        await self._create_v0_conversation(
            jinja_env, git_provider_tokens, conversation_metadata
        )

    async def _create_v0_conversation(
        self,
        jinja_env: Environment,
        git_provider_tokens: PROVIDER_TOKEN_TYPE,
        conversation_metadata: ConversationMetadata,
    ):
        """Create conversation using the legacy V0 system."""
        logger.info('[GitHub]: Creating V0 conversation')
        custom_secrets = await self._get_user_secrets()

        user_instructions, conversation_instructions = await self._get_instructions(
            jinja_env
        )

        await start_conversation(
            user_id=self.user_info.keycloak_user_id,
            git_provider_tokens=git_provider_tokens,
            custom_secrets=custom_secrets,
            initial_user_msg=user_instructions,
            image_urls=None,
            replay_json=None,
            conversation_id=conversation_metadata.conversation_id,
            conversation_metadata=conversation_metadata,
            conversation_instructions=conversation_instructions,
        )

    async def _create_v1_conversation(
        self,
        jinja_env: Environment,
        saas_user_auth: UserAuth,
        conversation_metadata: ConversationMetadata,
    ):
        """Create conversation using the new V1 app conversation system."""
        logger.info('[GitHub V1]: Creating V1 conversation')

        user_instructions, conversation_instructions = await self._get_instructions(
            jinja_env
        )

        # Create the initial message request
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=user_instructions)]
        )

        # Create the GitHub V1 callback processor
        github_callback_processor = self._create_github_v1_callback_processor()

        # Get the app conversation service and start the conversation
        injector_state = InjectorState()

        # Create the V1 conversation start request with the callback processor
        start_request = AppConversationStartRequest(
            conversation_id=UUID(conversation_metadata.conversation_id),
            system_message_suffix=conversation_instructions,
            initial_message=initial_message,
            selected_repository=self.full_repo_name,
            git_provider=ProviderType.GITHUB,
            title=f'GitHub Issue #{self.issue_number}: {self.title}',
            trigger=ConversationTrigger.RESOLVER,
            processors=[
                github_callback_processor
            ],  # Pass the callback processor directly
        )

        # Set up the GitHub user context for the V1 system
        github_user_context = ResolverUserContext(saas_user_auth=saas_user_auth)
        setattr(injector_state, USER_CONTEXT_ATTR, github_user_context)

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

        self.v1 = True

    def _create_github_v1_callback_processor(self):
        """Create a V1 callback processor for GitHub integration."""
        from openhands.app_server.event_callback.github_v1_callback_processor import (
            GithubV1CallbackProcessor,
        )

        # Create and return the GitHub V1 callback processor
        return GithubV1CallbackProcessor(
            github_view_data={
                'issue_number': self.issue_number,
                'full_repo_name': self.full_repo_name,
                'installation_id': self.installation_id,
            },
            send_summary_instruction=self.send_summary_instruction,
        )


@dataclass
class GithubIssueComment(GithubIssue):
    comment_body: str
    comment_id: int

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        user_instructions_template = jinja_env.get_template('issue_prompt.j2')

        await self._load_resolver_context()

        user_instructions = user_instructions_template.render(
            issue_comment=self.comment_body
        )

        conversation_instructions_template = jinja_env.get_template(
            'issue_conversation_instructions.j2'
        )

        conversation_instructions = conversation_instructions_template.render(
            issue_number=self.issue_number,
            issue_title=self.title,
            issue_body=self.description,
            previous_comments=self.previous_comments,
        )

        return user_instructions, conversation_instructions


@dataclass
class GithubPRComment(GithubIssueComment):
    branch_name: str

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        user_instructions_template = jinja_env.get_template('pr_update_prompt.j2')
        await self._load_resolver_context()

        user_instructions = user_instructions_template.render(
            pr_comment=self.comment_body,
        )

        conversation_instructions_template = jinja_env.get_template(
            'pr_update_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            pr_number=self.issue_number,
            branch_name=self.branch_name,
            pr_title=self.title,
            pr_body=self.description,
            comments=self.previous_comments,
        )

        return user_instructions, conversation_instructions

    async def initialize_new_conversation(self) -> ConversationMetadata:
        v1_enabled = await get_user_v1_enabled_setting(self.user_info.keycloak_user_id)
        logger.info(
            f'[GitHub V1]: User flag found for {self.user_info.keycloak_user_id} is {v1_enabled}'
        )
        if v1_enabled:
            # Create dummy conversationm metadata
            # Don't save to conversation store
            # V1 conversations are stored in a separate table
            return ConversationMetadata(
                conversation_id=uuid4().hex, selected_repository=self.full_repo_name
            )

        conversation_metadata: ConversationMetadata = await initialize_conversation(  # type: ignore[assignment]
            user_id=self.user_info.keycloak_user_id,
            conversation_id=None,
            selected_repository=self.full_repo_name,
            selected_branch=self.branch_name,
            conversation_trigger=ConversationTrigger.RESOLVER,
            git_provider=ProviderType.GITHUB,
        )

        self.conversation_id = conversation_metadata.conversation_id
        return conversation_metadata


@dataclass
class GithubInlinePRComment(GithubPRComment):
    file_location: str
    line_number: int
    comment_node_id: str

    async def _load_resolver_context(self):
        github_service = GithubServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )

        (
            self.title,
            self.description,
        ) = await github_service.get_issue_or_pr_title_and_body(
            self.full_repo_name, self.issue_number
        )

        self.previous_comments = await github_service.get_review_thread_comments(
            self.comment_node_id, self.full_repo_name, self.issue_number
        )

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        user_instructions_template = jinja_env.get_template('pr_update_prompt.j2')
        await self._load_resolver_context()

        user_instructions = user_instructions_template.render(
            pr_comment=self.comment_body,
        )

        conversation_instructions_template = jinja_env.get_template(
            'pr_update_conversation_instructions.j2'
        )

        conversation_instructions = conversation_instructions_template.render(
            pr_number=self.issue_number,
            pr_title=self.title,
            pr_body=self.description,
            branch_name=self.branch_name,
            file_location=self.file_location,
            line_number=self.line_number,
            comments=self.previous_comments,
        )

        return user_instructions, conversation_instructions

    def _create_github_v1_callback_processor(self):
        """Create a V1 callback processor for GitHub integration."""
        from openhands.app_server.event_callback.github_v1_callback_processor import (
            GithubV1CallbackProcessor,
        )

        # Create and return the GitHub V1 callback processor
        return GithubV1CallbackProcessor(
            github_view_data={
                'issue_number': self.issue_number,
                'full_repo_name': self.full_repo_name,
                'installation_id': self.installation_id,
                'comment_id': self.comment_id,
            },
            inline_pr_comment=True,
            send_summary_instruction=self.send_summary_instruction,
        )


@dataclass
class GithubFailingAction:
    unqiue_suggestions_header: str = (
        'Looks like there are a few issues preventing this PR from being merged!'
    )

    @staticmethod
    def get_latest_sha(pr: Issue) -> str:
        pr_obj = pr.as_pull_request()
        return pr_obj.head.sha

    @staticmethod
    def create_retrieve_workflows_callback(pr: Issue, head_sha: str):
        def get_all_workflows():
            repo = pr.repository
            workflows = repo.get_workflow_runs(head_sha=head_sha)

            runs = {}

            for workflow in workflows:
                conclusion = workflow.conclusion
                workflow_conclusion = WorkflowRunStatus.COMPLETED
                if conclusion is None:
                    workflow_conclusion = WorkflowRunStatus.PENDING  # type: ignore[unreachable]
                elif conclusion == WorkflowRunStatus.FAILURE.value:
                    workflow_conclusion = WorkflowRunStatus.FAILURE

                runs[str(workflow.id)] = WorkflowRun(
                    id=str(workflow.id), name=workflow.name, status=workflow_conclusion
                )

            return WorkflowRunGroup(runs=runs)

        return get_all_workflows

    @staticmethod
    def delete_old_comment_if_exists(pr: Issue):
        paginated_comments = pr.get_comments()
        for page in range(paginated_comments.totalCount):
            comments = paginated_comments.get_page(page)
            for comment in comments:
                if GithubFailingAction.unqiue_suggestions_header in comment.body:
                    comment.delete()

    @staticmethod
    def get_suggestions(
        failed_jobs: dict, pr_number: int, branch_name: str | None = None
    ) -> str:
        issues = []

        # Collect failing actions with their specific names
        if failed_jobs['actions']:
            failing_actions = failed_jobs['actions']
            issues.append(('GitHub Actions are failing:', False))
            for action in failing_actions:
                issues.append((action, True))

        if any(failed_jobs['merge conflict']):
            issues.append(('There are merge conflicts', False))

        # Format each line with proper indentation and dashes
        formatted_issues = []
        for issue, is_nested in issues:
            if is_nested:
                formatted_issues.append(f'  - {issue}')
            else:
                formatted_issues.append(f'- {issue}')
        issues_text = '\n'.join(formatted_issues)

        # Build list of possible suggestions based on actual issues
        suggestions = []
        branch_info = f' at branch `{branch_name}`' if branch_name else ''

        if any(failed_jobs['merge conflict']):
            suggestions.append(
                f'@OpenHands please fix the merge conflicts on PR #{pr_number}{branch_info}'
            )
        if any(failed_jobs['actions']):
            suggestions.append(
                f'@OpenHands please fix the failing actions on PR #{pr_number}{branch_info}'
            )

        # Take at most 2 suggestions
        suggestions = suggestions[:2]

        help_text = """If you'd like me to help, just leave a comment, like

```
{}
```

Feel free to include any additional details that might help me get this PR into a better state.

<sub><sup>You can manage your notification [settings]({})</sup></sub>""".format(
            '\n```\n\nor\n\n```\n'.join(suggestions), f'{HOST_URL}/settings/app'
        )

        return f'{GithubFailingAction.unqiue_suggestions_header}\n\n{issues_text}\n\n{help_text}'

    @staticmethod
    def leave_requesting_comment(pr: Issue, failed_runs: WorkflowRunGroup):
        failed_jobs: dict = {'actions': [], 'merge conflict': []}

        pr_obj = pr.as_pull_request()
        if not pr_obj.mergeable:
            failed_jobs['merge conflict'].append('Merge conflict detected')

        for _, workflow_run in failed_runs.runs.items():
            if workflow_run.status == WorkflowRunStatus.FAILURE:
                failed_jobs['actions'].append(workflow_run.name)

        logger.info(f'[GitHub] Found failing jobs for PR #{pr.number}: {failed_jobs}')

        # Get the branch name
        branch_name = pr_obj.head.ref

        # Get suggestions with branch name included
        suggestions = GithubFailingAction.get_suggestions(
            failed_jobs, pr.number, branch_name
        )

        GithubFailingAction.delete_old_comment_if_exists(pr)
        pr.create_comment(suggestions)


GithubViewType = (
    GithubInlinePRComment | GithubPRComment | GithubIssueComment | GithubIssue
)


# =================================================
# SECTION: Factory to create appriorate Github view
# =================================================


class GithubFactory:
    @staticmethod
    def is_labeled_issue(message: Message):
        payload = message.message.get('payload', {})
        action = payload.get('action', '')

        if action == 'labeled' and 'label' in payload and 'issue' in payload:
            label_name = payload['label'].get('name', '')
            if label_name == OH_LABEL:
                return True

        return False

    @staticmethod
    def is_issue_comment(message: Message):
        payload = message.message.get('payload', {})
        action = payload.get('action', '')

        if (
            action == 'created'
            and 'comment' in payload
            and 'issue' in payload
            and 'pull_request' not in payload['issue']
        ):
            comment_body = payload['comment']['body']
            if has_exact_mention(comment_body, INLINE_OH_LABEL):
                return True

        return False

    @staticmethod
    def is_pr_comment(message: Message):
        payload = message.message.get('payload', {})
        action = payload.get('action', '')

        if (
            action == 'created'
            and 'comment' in payload
            and 'issue' in payload
            and 'pull_request' in payload['issue']
        ):
            comment_body = payload['comment'].get('body', '')
            if has_exact_mention(comment_body, INLINE_OH_LABEL):
                return True

        return False

    @staticmethod
    def is_inline_pr_comment(message: Message):
        payload = message.message.get('payload', {})
        action = payload.get('action', '')

        if action == 'created' and 'comment' in payload and 'pull_request' in payload:
            comment_body = payload['comment'].get('body', '')
            if has_exact_mention(comment_body, INLINE_OH_LABEL):
                return True

        return False

    @staticmethod
    def is_eligible_for_conversation_starter(message: Message):
        if not ENABLE_PROACTIVE_CONVERSATION_STARTERS:
            return False

        payload = message.message.get('payload', {})
        action = payload.get('action', '')

        if not (action == 'completed' and 'workflow_run' in payload):
            return False

        return True

    @staticmethod
    async def trigger_conversation_starter(message: Message):
        """Trigger a conversation starter when a workflow fails.

        This is the updated version that checks user settings.
        """
        payload = message.message.get('payload', {})
        workflow_payload = payload['workflow_run']
        status = WorkflowRunStatus.COMPLETED

        if workflow_payload['conclusion'] == 'failure':
            status = WorkflowRunStatus.FAILURE
        elif workflow_payload['conclusion'] is None:
            status = WorkflowRunStatus.PENDING

        workflow_run = WorkflowRun(
            id=str(workflow_payload['id']), name=workflow_payload['name'], status=status
        )

        selected_repo = GithubFactory.get_full_repo_name(payload['repository'])
        head_branch = payload['workflow_run']['head_branch']

        # Get the user ID to check their settings
        user_id = None
        try:
            sender_id = payload['sender']['id']
            token_manager = TokenManager()
            user_id = await token_manager.get_user_id_from_idp_user_id(
                sender_id, ProviderType.GITHUB
            )
        except (KeyError, Exception) as e:
            logger.warning(
                f'Failed to get user ID for proactive conversation check: {str(e)}'
            )

        # Check if proactive conversations are enabled for this user
        if not await get_user_proactive_conversation_setting(user_id):
            return False

        def _interact_with_github() -> Issue | None:
            with GithubIntegration(
                GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY
            ) as integration:
                access_token = integration.get_access_token(
                    payload['installation']['id']
                ).token

            with Github(access_token) as gh:
                repo = gh.get_repo(selected_repo)
                login = (
                    payload['organization']['login']
                    if 'organization' in payload
                    else payload['sender']['login']
                )

                # See if a pull request is open
                open_pulls = repo.get_pulls(state='open', head=f'{login}:{head_branch}')
                if open_pulls.totalCount > 0:
                    prs = open_pulls.get_page(0)
                    relevant_pr = prs[0]
                    issue = repo.get_issue(number=relevant_pr.number)
                    return issue

            return None

        issue: Issue | None = await call_sync_from_async(_interact_with_github)
        if not issue:
            return False

        incoming_commit = payload['workflow_run']['head_sha']
        latest_sha = GithubFailingAction.get_latest_sha(issue)
        if latest_sha != incoming_commit:
            # Return as this commit is not the latest
            return False

        convo_store = ProactiveConversationStore()
        workflow_group = await convo_store.store_workflow_information(
            provider=ProviderType.GITHUB,
            repo_id=payload['repository']['id'],
            incoming_commit=incoming_commit,
            workflow=workflow_run,
            pr_number=issue.number,
            get_all_workflows=GithubFailingAction.create_retrieve_workflows_callback(
                issue, incoming_commit
            ),
        )

        if not workflow_group:
            return False

        logger.info(
            f'[GitHub] Workflow completed for {selected_repo}#{issue.number} on branch {head_branch}'
        )
        GithubFailingAction.leave_requesting_comment(issue, workflow_group)

        return False

    @staticmethod
    def get_full_repo_name(repo_obj: dict) -> str:
        owner = repo_obj['owner']['login']
        repo_name = repo_obj['name']
        return f'{owner}/{repo_name}'

    @staticmethod
    async def create_github_view_from_payload(
        message: Message, keycloak_user_id: str
    ) -> ResolverViewInterface:
        """Create the appropriate class (GithubIssue or GithubPRComment) based on the payload.
        Also return metadata about the event (e.g., action type).
        """
        payload = message.message.get('payload', {})
        repo_obj = payload['repository']
        user_id = payload['sender']['id']
        username = payload['sender']['login']

        selected_repo = GithubFactory.get_full_repo_name(repo_obj)
        is_public_repo = not repo_obj.get('private', True)
        user_info = UserData(
            user_id=user_id, username=username, keycloak_user_id=keycloak_user_id
        )

        installation_id = message.message['installation']

        if GithubFactory.is_labeled_issue(message):
            issue_number = payload['issue']['number']
            logger.info(
                f'[GitHub] Creating view for labeled issue from {username} in {selected_repo}#{issue_number}'
            )
            return GithubIssue(
                issue_number=issue_number,
                installation_id=installation_id,
                full_repo_name=selected_repo,
                is_public_repo=is_public_repo,
                raw_payload=message,
                user_info=user_info,
                conversation_id='',
                uuid=str(uuid4()),
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                v1=False,
            )

        elif GithubFactory.is_issue_comment(message):
            issue_number = payload['issue']['number']
            comment_body = payload['comment']['body']
            comment_id = payload['comment']['id']
            logger.info(
                f'[GitHub] Creating view for issue comment from {username} in {selected_repo}#{issue_number}'
            )
            return GithubIssueComment(
                issue_number=issue_number,
                comment_body=comment_body,
                comment_id=comment_id,
                installation_id=installation_id,
                full_repo_name=selected_repo,
                is_public_repo=is_public_repo,
                raw_payload=message,
                user_info=user_info,
                conversation_id='',
                uuid=None,
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                v1=False,
            )

        elif GithubFactory.is_pr_comment(message):
            issue_number = payload['issue']['number']
            logger.info(
                f'[GitHub] Creating view for PR comment from {username} in {selected_repo}#{issue_number}'
            )

            access_token = ''
            with GithubIntegration(
                GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY
            ) as integration:
                access_token = integration.get_access_token(installation_id).token

            head_ref = None
            with Github(access_token) as gh:
                repo = gh.get_repo(selected_repo)
                pull_request = repo.get_pull(issue_number)
                head_ref = pull_request.head.ref
                logger.info(
                    f'[GitHub] Found PR branch {head_ref} for {selected_repo}#{issue_number}'
                )

            comment_id = payload['comment']['id']
            return GithubPRComment(
                issue_number=issue_number,
                branch_name=head_ref,
                comment_body=payload['comment']['body'],
                comment_id=comment_id,
                installation_id=installation_id,
                full_repo_name=selected_repo,
                is_public_repo=is_public_repo,
                raw_payload=message,
                user_info=user_info,
                conversation_id='',
                uuid=None,
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                v1=False,
            )

        elif GithubFactory.is_inline_pr_comment(message):
            pr_number = payload['pull_request']['number']
            branch_name = payload['pull_request']['head']['ref']
            comment_id = payload['comment']['id']
            comment_node_id = payload['comment']['node_id']
            file_path = payload['comment']['path']
            line_number = payload['comment']['line']
            logger.info(
                f'[GitHub] Creating view for inline PR comment from {username} in {selected_repo}#{pr_number} at {file_path}'
            )

            return GithubInlinePRComment(
                issue_number=pr_number,
                branch_name=branch_name,
                comment_body=payload['comment']['body'],
                comment_node_id=comment_node_id,
                comment_id=comment_id,
                file_location=file_path,
                line_number=line_number,
                installation_id=installation_id,
                full_repo_name=selected_repo,
                is_public_repo=is_public_repo,
                raw_payload=message,
                user_info=user_info,
                conversation_id='',
                uuid=None,
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                v1=False,
            )

        else:
            raise ValueError(
                "Invalid payload: must contain either 'issue' or 'pull_request'"
            )
