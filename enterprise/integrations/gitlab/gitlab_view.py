from dataclasses import dataclass
from uuid import UUID, uuid4

from integrations.models import Message
from integrations.resolver_context import ResolverUserContext
from integrations.types import ResolverViewInterface, UserData
from integrations.utils import (
    ENABLE_V1_GITLAB_RESOLVER,
    HOST,
    get_oh_labels,
    get_user_v1_enabled_setting,
    has_exact_mention,
)
from jinja2 import Environment
from server.auth.token_manager import TokenManager
from server.config import get_config
from storage.saas_secrets_store import SaasSecretsStore

from openhands.agent_server.models import SendMessageRequest
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.gitlab.gitlab_service import GitLabServiceImpl
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

OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)
CONFIDENTIAL_NOTE = 'confidential_note'
NOTE_TYPES = ['note', CONFIDENTIAL_NOTE]


async def is_v1_enabled_for_gitlab_resolver(user_id: str) -> bool:
    return await get_user_v1_enabled_setting(user_id) and ENABLE_V1_GITLAB_RESOLVER


# =================================================
# SECTION: Factory to create appriorate Gitlab view
# =================================================


@dataclass
class GitlabIssue(ResolverViewInterface):
    installation_id: str  # Webhook installation ID for Gitlab (comes from our DB)
    issue_number: int
    project_id: int
    full_repo_name: str
    is_public_repo: bool
    user_info: UserData
    raw_payload: Message
    conversation_id: str
    should_extract: bool
    send_summary_instruction: bool
    title: str
    description: str
    previous_comments: list[Comment]
    is_mr: bool
    v1_enabled: bool

    def _get_branch_name(self) -> str | None:
        return getattr(self, 'branch_name', None)

    async def _load_resolver_context(self):
        gitlab_service = GitLabServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )

        self.previous_comments = await gitlab_service.get_issue_or_mr_comments(
            str(self.project_id), self.issue_number, is_mr=self.is_mr
        )

        (
            self.title,
            self.description,
        ) = await gitlab_service.get_issue_or_mr_title_and_body(
            str(self.project_id), self.issue_number, is_mr=self.is_mr
        )

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        user_instructions_template = jinja_env.get_template('issue_prompt.j2')
        await self._load_resolver_context()

        user_instructions = user_instructions_template.render(
            issue_number=self.issue_number,
        )

        conversation_instructions_template = jinja_env.get_template(
            'issue_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            issue_title=self.title,
            issue_body=self.description,
            comments=self.previous_comments,
        )

        return user_instructions, conversation_instructions

    async def _get_user_secrets(self):
        secrets_store = SaasSecretsStore(self.user_info.keycloak_user_id, get_config())
        user_secrets = await secrets_store.load()

        return user_secrets.custom_secrets if user_secrets else None

    async def initialize_new_conversation(self) -> ConversationMetadata:
        # v1_enabled is already set at construction time in the factory method
        # This is the source of truth for the conversation type
        if self.v1_enabled:
            # Create dummy conversation metadata
            # Don't save to conversation store
            # V1 conversations are stored in a separate table
            self.conversation_id = uuid4().hex
            return ConversationMetadata(
                conversation_id=self.conversation_id,
                selected_repository=self.full_repo_name,
            )

        conversation_metadata: ConversationMetadata = await initialize_conversation(  # type: ignore[assignment]
            user_id=self.user_info.keycloak_user_id,
            conversation_id=None,
            selected_repository=self.full_repo_name,
            selected_branch=self._get_branch_name(),
            conversation_trigger=ConversationTrigger.RESOLVER,
            git_provider=ProviderType.GITLAB,
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
        # v1_enabled is already set at construction time in the factory method
        if self.v1_enabled:
            # Use V1 app conversation service
            await self._create_v1_conversation(
                jinja_env, saas_user_auth, conversation_metadata
            )
        else:
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
        logger.info('[GitLab]: Creating V0 conversation')
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
        logger.info('[GitLab V1]: Creating V1 conversation')

        user_instructions, conversation_instructions = await self._get_instructions(
            jinja_env
        )

        # Create the initial message request
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=user_instructions)]
        )

        # Create the GitLab V1 callback processor
        gitlab_callback_processor = self._create_gitlab_v1_callback_processor()

        # Get the app conversation service and start the conversation
        injector_state = InjectorState()

        # Determine the title based on whether it's an MR or issue
        title_prefix = 'GitLab MR' if self.is_mr else 'GitLab Issue'
        title = f'{title_prefix} #{self.issue_number}: {self.title}'

        # Create the V1 conversation start request with the callback processor
        start_request = AppConversationStartRequest(
            conversation_id=UUID(conversation_metadata.conversation_id),
            system_message_suffix=conversation_instructions,
            initial_message=initial_message,
            selected_repository=self.full_repo_name,
            selected_branch=self._get_branch_name(),
            git_provider=ProviderType.GITLAB,
            title=title,
            trigger=ConversationTrigger.RESOLVER,
            processors=[
                gitlab_callback_processor
            ],  # Pass the callback processor directly
        )

        # Set up the GitLab user context for the V1 system
        gitlab_user_context = ResolverUserContext(saas_user_auth=saas_user_auth)
        setattr(injector_state, USER_CONTEXT_ATTR, gitlab_user_context)

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

    def _create_gitlab_v1_callback_processor(self):
        """Create a V1 callback processor for GitLab integration."""
        from integrations.gitlab.gitlab_v1_callback_processor import (
            GitlabV1CallbackProcessor,
        )

        # Create and return the GitLab V1 callback processor
        return GitlabV1CallbackProcessor(
            gitlab_view_data={
                'issue_number': self.issue_number,
                'project_id': self.project_id,
                'full_repo_name': self.full_repo_name,
                'installation_id': self.installation_id,
                'keycloak_user_id': self.user_info.keycloak_user_id,
                'is_mr': self.is_mr,
                'discussion_id': getattr(self, 'discussion_id', None),
            },
            send_summary_instruction=self.send_summary_instruction,
        )


@dataclass
class GitlabIssueComment(GitlabIssue):
    comment_body: str
    discussion_id: str
    confidential: bool

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
            comments=self.previous_comments,
        )

        return user_instructions, conversation_instructions


@dataclass
class GitlabMRComment(GitlabIssueComment):
    branch_name: str

    def _get_branch_name(self) -> str | None:
        return self.branch_name

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        user_instructions_template = jinja_env.get_template('mr_update_prompt.j2')
        await self._load_resolver_context()

        user_instructions = user_instructions_template.render(
            mr_comment=self.comment_body,
        )

        conversation_instructions_template = jinja_env.get_template(
            'mr_update_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            mr_number=self.issue_number,
            branch_name=self.branch_name,
            mr_title=self.title,
            mr_body=self.description,
            comments=self.previous_comments,
        )

        return user_instructions, conversation_instructions


@dataclass
class GitlabInlineMRComment(GitlabMRComment):
    file_location: str
    line_number: int

    async def _load_resolver_context(self):
        gitlab_service = GitLabServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )

        (
            self.title,
            self.description,
        ) = await gitlab_service.get_issue_or_mr_title_and_body(
            str(self.project_id), self.issue_number, is_mr=self.is_mr
        )

        self.previous_comments = await gitlab_service.get_review_thread_comments(
            str(self.project_id), self.issue_number, self.discussion_id
        )

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        user_instructions_template = jinja_env.get_template('mr_update_prompt.j2')
        await self._load_resolver_context()

        user_instructions = user_instructions_template.render(
            mr_comment=self.comment_body,
        )

        conversation_instructions_template = jinja_env.get_template(
            'mr_update_conversation_instructions.j2'
        )

        conversation_instructions = conversation_instructions_template.render(
            mr_number=self.issue_number,
            mr_title=self.title,
            mr_body=self.description,
            branch_name=self.branch_name,
            file_location=self.file_location,
            line_number=self.line_number,
            comments=self.previous_comments,
        )

        return user_instructions, conversation_instructions


GitlabViewType = (
    GitlabInlineMRComment | GitlabMRComment | GitlabIssueComment | GitlabIssue
)


class GitlabFactory:
    @staticmethod
    def is_labeled_issue(message: Message) -> bool:
        payload = message.message['payload']
        object_kind = payload.get('object_kind')
        event_type = payload.get('event_type')

        if object_kind == 'issue' and event_type == 'issue':
            changes = payload.get('changes', {})
            labels = changes.get('labels', {})
            previous = labels.get('previous', [])
            current = labels.get('current', [])

            previous_labels = [obj['title'] for obj in previous]
            current_labels = [obj['title'] for obj in current]

            if OH_LABEL not in previous_labels and OH_LABEL in current_labels:
                return True

        return False

    @staticmethod
    def is_issue_comment(message: Message) -> bool:
        payload = message.message['payload']
        object_kind = payload.get('object_kind')
        event_type = payload.get('event_type')
        issue = payload.get('issue')

        if object_kind == 'note' and event_type in NOTE_TYPES and issue:
            comment_body = payload.get('object_attributes', {}).get('note', '')
            return has_exact_mention(comment_body, INLINE_OH_LABEL)

        return False

    @staticmethod
    def is_mr_comment(message: Message, inline=False) -> bool:
        payload = message.message['payload']
        object_kind = payload.get('object_kind')
        event_type = payload.get('event_type')
        merge_request = payload.get('merge_request')

        if not (object_kind == 'note' and event_type in NOTE_TYPES and merge_request):
            return False

        # Check whether not belongs to MR
        object_attributes = payload.get('object_attributes', {})
        noteable_type = object_attributes.get('noteable_type')

        if noteable_type != 'MergeRequest':
            return False

        # Check whether comment is inline
        change_position = object_attributes.get('change_position')
        if inline and not change_position:
            return False
        if not inline and change_position:
            return False

        # Check body
        comment_body = object_attributes.get('note', '')
        return has_exact_mention(comment_body, INLINE_OH_LABEL)

    @staticmethod
    def determine_if_confidential(event_type: str):
        return event_type == CONFIDENTIAL_NOTE

    @staticmethod
    async def create_gitlab_view_from_payload(
        message: Message, token_manager: TokenManager
    ) -> GitlabViewType:
        payload = message.message['payload']
        installation_id = message.message['installation_id']
        user = payload['user']
        user_id = user['id']
        username = user['username']
        repo_obj = payload['project']
        selected_project = repo_obj['path_with_namespace']
        is_public_repo = repo_obj['visibility_level'] == 0
        project_id = payload['object_attributes']['project_id']

        keycloak_user_id = await token_manager.get_user_id_from_idp_user_id(
            user_id, ProviderType.GITLAB
        )

        user_info = UserData(
            user_id=user_id, username=username, keycloak_user_id=keycloak_user_id
        )

        # Check v1_enabled at construction time - this is the source of truth
        v1_enabled = (
            await is_v1_enabled_for_gitlab_resolver(keycloak_user_id)
            if keycloak_user_id
            else False
        )
        logger.info(
            f'[GitLab V1]: User flag found for {keycloak_user_id} is {v1_enabled}'
        )

        if GitlabFactory.is_labeled_issue(message):
            issue_iid = payload['object_attributes']['iid']

            logger.info(
                f'[GitLab] Creating view for labeled issue from {username} in {selected_project}#{issue_iid}'
            )
            return GitlabIssue(
                installation_id=installation_id,
                issue_number=issue_iid,
                project_id=project_id,
                full_repo_name=selected_project,
                is_public_repo=is_public_repo,
                user_info=user_info,
                raw_payload=message,
                conversation_id='',
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                is_mr=False,
                v1_enabled=v1_enabled,
            )

        elif GitlabFactory.is_issue_comment(message):
            event_type = payload['event_type']
            issue_iid = payload['issue']['iid']
            object_attributes = payload['object_attributes']
            discussion_id = object_attributes['discussion_id']
            comment_body = object_attributes['note']
            logger.info(
                f'[GitLab] Creating view for issue comment from {username} in {selected_project}#{issue_iid}'
            )

            return GitlabIssueComment(
                installation_id=installation_id,
                comment_body=comment_body,
                issue_number=issue_iid,
                discussion_id=discussion_id,
                project_id=project_id,
                confidential=GitlabFactory.determine_if_confidential(event_type),
                full_repo_name=selected_project,
                is_public_repo=is_public_repo,
                user_info=user_info,
                raw_payload=message,
                conversation_id='',
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                is_mr=False,
                v1_enabled=v1_enabled,
            )

        elif GitlabFactory.is_mr_comment(message):
            event_type = payload['event_type']
            merge_request_iid = payload['merge_request']['iid']
            branch_name = payload['merge_request']['source_branch']
            object_attributes = payload['object_attributes']
            discussion_id = object_attributes['discussion_id']
            comment_body = object_attributes['note']
            logger.info(
                f'[GitLab] Creating view for merge request comment from {username} in {selected_project}#{merge_request_iid}'
            )

            return GitlabMRComment(
                installation_id=installation_id,
                comment_body=comment_body,
                issue_number=merge_request_iid,  # Using issue_number as mr_number for compatibility
                discussion_id=discussion_id,
                project_id=project_id,
                full_repo_name=selected_project,
                is_public_repo=is_public_repo,
                user_info=user_info,
                raw_payload=message,
                conversation_id='',
                should_extract=True,
                send_summary_instruction=True,
                confidential=GitlabFactory.determine_if_confidential(event_type),
                branch_name=branch_name,
                title='',
                description='',
                previous_comments=[],
                is_mr=True,
                v1_enabled=v1_enabled,
            )

        elif GitlabFactory.is_mr_comment(message, inline=True):
            event_type = payload['event_type']
            merge_request_iid = payload['merge_request']['iid']
            branch_name = payload['merge_request']['source_branch']
            object_attributes = payload['object_attributes']
            comment_body = object_attributes['note']
            position_info = object_attributes['position']
            discussion_id = object_attributes['discussion_id']
            file_location = object_attributes['position']['new_path']
            line_number = (
                position_info.get('new_line') or position_info.get('old_line') or 0
            )

            logger.info(
                f'[GitLab] Creating view for inline merge request comment from {username} in {selected_project}#{merge_request_iid}'
            )

            return GitlabInlineMRComment(
                installation_id=installation_id,
                issue_number=merge_request_iid,  # Using issue_number as mr_number for compatibility
                discussion_id=discussion_id,
                project_id=project_id,
                full_repo_name=selected_project,
                is_public_repo=is_public_repo,
                user_info=user_info,
                raw_payload=message,
                conversation_id='',
                should_extract=True,
                send_summary_instruction=True,
                confidential=GitlabFactory.determine_if_confidential(event_type),
                branch_name=branch_name,
                file_location=file_location,
                line_number=line_number,
                comment_body=comment_body,
                title='',
                description='',
                previous_comments=[],
                is_mr=True,
                v1_enabled=v1_enabled,
            )

        raise ValueError(f'Unhandled GitLab webhook event: {message}')
