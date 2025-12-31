from dataclasses import dataclass

from integrations.models import Message
from integrations.types import ResolverViewInterface, UserData
from integrations.utils import HOST, get_oh_labels, has_exact_mention
from jinja2 import Environment
from server.auth.token_manager import TokenManager
from server.config import get_config
from storage.database import session_maker
from storage.saas_secrets_store import SaasSecretsStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.gitlab.gitlab_service import GitLabServiceImpl
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.integrations.service_types import Comment
from openhands.server.services.conversation_service import create_new_conversation
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)
CONFIDENTIAL_NOTE = 'confidential_note'
NOTE_TYPES = ['note', CONFIDENTIAL_NOTE]

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
        secrets_store = SaasSecretsStore(
            self.user_info.keycloak_user_id, session_maker, get_config()
        )
        user_secrets = await secrets_store.load()

        return user_secrets.custom_secrets if user_secrets else None

    async def create_new_conversation(
        self, jinja_env: Environment, git_provider_tokens: PROVIDER_TOKEN_TYPE
    ):
        custom_secrets = await self._get_user_secrets()

        user_instructions, conversation_instructions = await self._get_instructions(
            jinja_env
        )
        agent_loop_info = await create_new_conversation(
            user_id=self.user_info.keycloak_user_id,
            git_provider_tokens=git_provider_tokens,
            custom_secrets=custom_secrets,
            selected_repository=self.full_repo_name,
            selected_branch=None,
            initial_user_msg=user_instructions,
            conversation_instructions=conversation_instructions,
            image_urls=None,
            conversation_trigger=ConversationTrigger.RESOLVER,
            replay_json=None,
        )
        self.conversation_id = agent_loop_info.conversation_id
        return self.conversation_id


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

    async def create_new_conversation(
        self, jinja_env: Environment, git_provider_tokens: PROVIDER_TOKEN_TYPE
    ):
        custom_secrets = await self._get_user_secrets()

        user_instructions, conversation_instructions = await self._get_instructions(
            jinja_env
        )
        agent_loop_info = await create_new_conversation(
            user_id=self.user_info.keycloak_user_id,
            git_provider_tokens=git_provider_tokens,
            custom_secrets=custom_secrets,
            selected_repository=self.full_repo_name,
            selected_branch=self.branch_name,
            initial_user_msg=user_instructions,
            conversation_instructions=conversation_instructions,
            image_urls=None,
            conversation_trigger=ConversationTrigger.RESOLVER,
            replay_json=None,
        )
        self.conversation_id = agent_loop_info.conversation_id
        return self.conversation_id


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
    ) -> ResolverViewInterface:
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
            )
