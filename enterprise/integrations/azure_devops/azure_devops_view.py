from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse
from uuid import UUID, uuid4

from integrations.azure_devops.azure_devops_service import SaaSAzureDevOpsService
from integrations.models import Message
from integrations.resolver_context import ResolverUserContext
from integrations.resolver_org_router import resolve_org_for_repo
from integrations.types import ResolverViewInterface, UserData
from integrations.utils import HOST, get_oh_labels, has_exact_mention
from jinja2 import Environment

from openhands.agent_server.models import SendMessageRequest
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
    ConversationTrigger,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl,
)
from openhands.app_server.integrations.azure_devops.service.webhooks import (
    AZURE_DEVOPS_PR_COMMENT_EVENT as PR_COMMENT_EVENT,
)
from openhands.app_server.integrations.azure_devops.service.webhooks import (
    AZURE_DEVOPS_WORK_ITEM_COMMENT_EVENT as WORK_ITEM_COMMENT_EVENT,
)
from openhands.app_server.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.app_server.integrations.service_types import Comment
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.app_server.user_auth.user_auth import UserAuth
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk import TextContent

OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)
OPENHANDS_COMMENT_MARKER = '<!-- openhands-azure-devops-resolver -->'


def mark_openhands_comment(comment: str) -> str:
    if OPENHANDS_COMMENT_MARKER in comment:
        return comment
    return f'{OPENHANDS_COMMENT_MARKER}\n{comment}'


def _is_openhands_comment(comment: str) -> bool:
    return OPENHANDS_COMMENT_MARKER in comment


def _strip_ref_prefix(ref_name: str | None) -> str | None:
    if not ref_name:
        return None
    return ref_name.removeprefix('refs/heads/')


def _extract_org_from_url(raw_url: str | None) -> str:
    if not raw_url:
        return ''

    parsed = urlparse(raw_url)
    hostname = parsed.hostname or ''
    path_parts = [part for part in parsed.path.split('/') if part]

    if hostname.endswith('dev.azure.com') and path_parts:
        return path_parts[0]

    if hostname.endswith('.visualstudio.com'):
        return hostname.split('.')[0]

    return ''


def _extract_org(payload: dict[str, Any]) -> str:
    resource_containers = payload.get('resourceContainers') or {}
    for container_name in ('account', 'collection', 'project'):
        org = _extract_org_from_url(
            (resource_containers.get(container_name) or {}).get('baseUrl')
        )
        if org:
            return org

    resource = payload.get('resource') or {}
    for candidate in (
        ((resource.get('pullRequest') or {}).get('repository') or {}).get('remoteUrl'),
        ((resource.get('pullRequest') or {}).get('repository') or {}).get('url'),
        ((resource.get('comment') or {}).get('_links') or {})
        .get('self', {})
        .get('href'),
        ((resource.get('_links') or {}).get('self') or {}).get('href'),
        resource.get('url'),
    ):
        org = _extract_org_from_url(candidate)
        if org:
            return org

    return ''


def _extract_work_item_comment(payload: dict[str, Any]) -> str:
    fields = (payload.get('resource') or {}).get('fields') or {}
    history = fields.get('System.History') or ''
    if history:
        return history

    detailed = (payload.get('detailedMessage') or {}).get('text') or ''
    # detailedMessage is "<actor> commented\n<body>"; drop the first line.
    return detailed.split('\n', 1)[-1].strip() if '\n' in detailed else ''


def _select_project_repo(
    repos: list[dict[str, Any]], project_name: str
) -> dict[str, Any] | None:
    """Resolve a project's repo deterministically: name match, else the sole
    repo. Returns None when ambiguous (multiple repos, no name match) so we
    fail closed rather than guess."""
    if not repos:
        return None
    for repo in repos:
        if (repo.get('name') or '').lower() == project_name.lower():
            return repo
    if len(repos) == 1:
        return repos[0]
    return None


def _actor_username(actor: dict[str, Any]) -> str:
    return (
        actor.get('displayName')
        or actor.get('uniqueName')
        or actor.get('id')
        or 'unknown'
    )


def actor_email(actor: dict[str, Any]) -> str:
    unique_name = str(actor.get('uniqueName') or '').strip()
    if '@' in unique_name:
        return unique_name
    display_name = str(actor.get('displayName') or '').strip()
    match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', display_name)
    return match.group(0) if match else ''


@dataclass
class AzureDevOpsItem(ResolverViewInterface):
    installation_id: str
    issue_number: int
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
    project_name: str
    project_id: str
    repository_id: str

    def _get_branch_name(self) -> str | None:
        # Only PR comments carry a branch; overridden there.
        return None

    async def _load_resolver_context(self) -> None:
        azure_service = AzureDevOpsServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )
        self.previous_comments = await azure_service.get_issue_or_pr_comments(
            self.full_repo_name,
            self.issue_number,
            max_comments=100,
        )
        (
            self.title,
            self.description,
        ) = await azure_service.get_issue_or_pr_title_and_body(
            self.full_repo_name,
            self.issue_number,
        )

    async def initialize_new_conversation(self) -> UUID:
        self.resolved_org_id = await resolve_org_for_repo(
            provider=ProviderType.AZURE_DEVOPS.value,
            full_repo_name=self.full_repo_name,
            keycloak_user_id=self.user_info.keycloak_user_id,
        )
        conversation_id = uuid4()
        self.conversation_id = conversation_id.hex
        return conversation_id

    async def create_new_conversation(
        self,
        jinja_env: Environment,
        git_provider_tokens: PROVIDER_TOKEN_TYPE,
        conversation_id: UUID,
        saas_user_auth: UserAuth,
    ) -> None:
        user_instructions, conversation_instructions = await self._get_instructions(
            jinja_env
        )
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=user_instructions)]
        )

        from integrations.azure_devops.azure_devops_v1_callback_processor import (
            AzureDevOpsV1CallbackProcessor,
        )

        callback_processor = AzureDevOpsV1CallbackProcessor(
            azure_devops_view_data={
                'repository': self.full_repo_name,
                'issue_number': self.issue_number,
                'keycloak_user_id': self.user_info.keycloak_user_id,
                'is_pr': isinstance(self, AzureDevOpsPRComment),
                'thread_id': getattr(self, 'thread_id', None),
            },
            should_request_summary=self.send_summary_instruction,
        )

        title_prefix = (
            'Azure DevOps PR'
            if isinstance(self, AzureDevOpsPRComment)
            else 'Azure DevOps Work Item'
        )
        start_request = AppConversationStartRequest(
            conversation_id=conversation_id,
            system_message_suffix=conversation_instructions or None,
            initial_message=initial_message,
            selected_repository=self.full_repo_name,
            selected_branch=self._get_branch_name(),
            git_provider=ProviderType.AZURE_DEVOPS,
            title=f'{title_prefix} #{self.issue_number}: {self.title}',
            trigger=ConversationTrigger.RESOLVER,
            processors=[callback_processor],
        )

        injector_state = InjectorState()
        azure_user_context = ResolverUserContext(
            saas_user_auth=saas_user_auth,
            resolver_org_id=self.resolved_org_id,
        )
        setattr(injector_state, USER_CONTEXT_ATTR, azure_user_context)

        async with get_app_conversation_service(
            injector_state
        ) as app_conversation_service:
            async for task in app_conversation_service.start_app_conversation(
                start_request
            ):
                if task.status == AppConversationStartTaskStatus.ERROR:
                    logger.error(
                        f'[Azure DevOps] Failed to start V1 conversation: {task.detail}'
                    )
                    raise RuntimeError(
                        f'Failed to start V1 conversation: {task.detail}'
                    )


@dataclass
class AzureDevOpsPRComment(AzureDevOpsItem):
    comment_id: int | None
    comment_body: str
    thread_id: int | None
    branch_name: str | None

    def _get_branch_name(self) -> str | None:
        return self.branch_name

    async def _load_resolver_context(self) -> None:
        # Call PR-specific methods directly so a brand-new PR with no prior
        # comments (empty list) never falls back to unrelated work-item discussion.
        azure_service = AzureDevOpsServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )
        self.previous_comments = await azure_service.get_pr_comments(
            self.full_repo_name,
            self.issue_number,
            max_comments=100,
        )
        (
            self.title,
            self.description,
        ) = await azure_service.get_issue_or_pr_title_and_body(
            self.full_repo_name,
            self.issue_number,
        )

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        await self._load_resolver_context()

        user_instructions_template = jinja_env.get_template('pr_update_prompt.j2')
        user_instructions = user_instructions_template.render(
            pr_comment=self.comment_body
        )

        conversation_instructions_template = jinja_env.get_template(
            'pr_update_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            pr_number=self.issue_number,
            branch_name=self.branch_name or '',
            pr_title=self.title,
            pr_body=self.description,
            comments=self.previous_comments,
        )
        return user_instructions, conversation_instructions


@dataclass
class AzureDevOpsWorkItemComment(AzureDevOpsItem):
    comment_body: str

    async def _load_resolver_context(self) -> None:
        # Load work-item context directly; never probe PRs (a PR can share the id).
        azure_service = AzureDevOpsServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )
        self.previous_comments = await azure_service.get_work_item_comments(
            self.full_repo_name,
            self.issue_number,
            max_comments=100,
        )
        (
            self.title,
            self.description,
        ) = await azure_service.get_work_item_title_and_body(self.issue_number)

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        await self._load_resolver_context()

        user_instructions_template = jinja_env.get_template('issue_prompt.j2')
        user_instructions = user_instructions_template.render(
            issue_comment=self.comment_body,
            issue_number=self.issue_number,
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


AzureDevOpsViewType = AzureDevOpsPRComment | AzureDevOpsWorkItemComment


class AzureDevOpsFactory:
    @staticmethod
    def event_key(message: Message) -> str:
        return str(
            message.message.get('event_key')
            or (message.message.get('payload') or {}).get('eventType')
            or ''
        )

    @staticmethod
    def extract_actor(message: Message) -> dict[str, Any]:
        payload = message.message.get('payload') or {}
        resource = payload.get('resource') or {}
        if AzureDevOpsFactory.event_key(message) == PR_COMMENT_EVENT:
            return (resource.get('comment') or {}).get('author') or {}
        if AzureDevOpsFactory.event_key(message) == WORK_ITEM_COMMENT_EVENT:
            revised_by = resource.get('revisedBy')
            if isinstance(revised_by, dict):
                return revised_by
            changed_by = (resource.get('fields') or {}).get('System.ChangedBy')
            if isinstance(changed_by, dict):
                return changed_by
            if isinstance(changed_by, str):
                return {'displayName': changed_by, 'uniqueName': changed_by}
        return {}

    @staticmethod
    def is_pr_comment(message: Message) -> bool:
        if AzureDevOpsFactory.event_key(message) != PR_COMMENT_EVENT:
            return False
        payload = message.message.get('payload') or {}
        comment = ((payload.get('resource') or {}).get('comment') or {}).get(
            'content'
        ) or ''
        if _is_openhands_comment(comment):
            return False
        return has_exact_mention(comment, INLINE_OH_LABEL)

    @staticmethod
    def is_work_item_comment(message: Message) -> bool:
        if AzureDevOpsFactory.event_key(message) != WORK_ITEM_COMMENT_EVENT:
            return False
        payload = message.message.get('payload') or {}
        comment = _extract_work_item_comment(payload)
        if _is_openhands_comment(comment):
            return False
        return has_exact_mention(comment, INLINE_OH_LABEL)

    @staticmethod
    async def create_azure_devops_view_from_payload(
        message: Message,
        keycloak_user_id: str,
    ) -> AzureDevOpsViewType | None:
        payload = message.message.get('payload') or {}
        resource = payload.get('resource') or {}
        actor = AzureDevOpsFactory.extract_actor(message)
        username = _actor_username(actor)
        actor_id = str(actor.get('id') or actor.get('uniqueName') or username)
        org = _extract_org(payload)
        if not org:
            # Without an org, full_repo_name would be malformed ('/project/repo').
            logger.warning(
                '[Azure DevOps] Could not determine organization from payload; '
                'skipping.'
            )
            return None

        if AzureDevOpsFactory.is_pr_comment(message):
            pull_request = resource.get('pullRequest') or {}
            repository = pull_request.get('repository') or {}
            project = repository.get('project') or {}
            project_name = project.get('name') or ''
            project_id = str(project.get('id') or '')
            repository_id = str(repository.get('id') or '')
            repo_name = repository.get('name') or ''
            pr_number = int(pull_request.get('pullRequestId') or 0)
            comment = resource.get('comment') or {}
            thread_href = ((comment.get('_links') or {}).get('threads') or {}).get(
                'href'
            ) or ''
            thread_match = re.search(r'/threads/(\d+)(?:$|[/?#])', thread_href)
            full_repo_name = f'{org}/{project_name}/{repo_name}'
            user_info = UserData(
                user_id=actor_id,
                username=username,
                keycloak_user_id=keycloak_user_id,
            )
            return AzureDevOpsPRComment(
                installation_id=full_repo_name,
                issue_number=pr_number,
                full_repo_name=full_repo_name,
                is_public_repo=False,
                user_info=user_info,
                raw_payload=message,
                conversation_id='',
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                project_name=project_name,
                project_id=project_id,
                repository_id=repository_id,
                comment_id=comment.get('id'),
                comment_body=comment.get('content') or '',
                thread_id=int(thread_match.group(1)) if thread_match else None,
                branch_name=_strip_ref_prefix(pull_request.get('sourceRefName')),
            )

        if AzureDevOpsFactory.is_work_item_comment(message):
            fields = resource.get('fields') or {}
            project_name = fields.get('System.TeamProject') or ''
            work_item_id = int(resource.get('id') or 0)

            # Work items aren't tied to a repo; resolve a real one for the project
            # instead of fabricating a project-named repo that may not exist.
            azure_service = cast(
                SaaSAzureDevOpsService,
                AzureDevOpsServiceImpl(external_auth_id=keycloak_user_id),
            )
            repos = await azure_service.get_project_repositories(project_name)
            repo = _select_project_repo(repos, project_name)
            if repo is None:
                logger.warning(
                    f'[Azure DevOps] Could not unambiguously resolve a repository '
                    f'for work item in project {org}/{project_name}; skipping.'
                )
                return None
            repo_name = repo.get('name') or ''
            repository_id = str(repo.get('id') or '')
            project_id = str((repo.get('project') or {}).get('id') or '')
            full_repo_name = f'{org}/{project_name}/{repo_name}'
            user_info = UserData(
                user_id=actor_id,
                username=username,
                keycloak_user_id=keycloak_user_id,
            )
            return AzureDevOpsWorkItemComment(
                installation_id=full_repo_name,
                issue_number=work_item_id,
                full_repo_name=full_repo_name,
                is_public_repo=False,
                user_info=user_info,
                raw_payload=message,
                conversation_id='',
                should_extract=True,
                send_summary_instruction=True,
                title='',
                description='',
                previous_comments=[],
                project_name=project_name,
                project_id=project_id,
                repository_id=repository_id,
                comment_body=_extract_work_item_comment(payload),
            )

        # Unreachable: receive_message only calls this after is_job_requested() is True.
        raise ValueError(f'Unhandled Azure DevOps webhook event: {message}')
