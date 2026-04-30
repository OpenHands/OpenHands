from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from uuid import UUID, uuid4

from integrations.models import Message
from integrations.resolver_context import ResolverUserContext
from integrations.resolver_org_router import resolve_org_for_repo
from integrations.types import ResolverViewInterface, UserData
from integrations.utils import (
    HOST,
    get_oh_labels,
    has_exact_mention,
)
from jinja2 import Environment

from openhands.agent_server.models import SendMessageRequest
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
    ConversationTrigger,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCServiceImpl,
)
from openhands.app_server.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderType,
)
from openhands.app_server.integrations.service_types import Comment
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.app_server.user_auth.user_auth import UserAuth
from openhands.core.logger import openhands_logger as logger
from openhands.sdk import TextContent

OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)


def extract_actor_slug(actor: dict) -> str:
    """Extract a stable user identifier from a Bitbucket DC ``actor`` block.

    DC's stable identifier is the user ``slug`` (lowercase username); ``id``
    is the numeric DB id and ``name`` is also the username on most
    installations. Prefer ``slug`` for downstream API filters and dedup,
    fall back to ``name``, and finally to the numeric ``id`` as a string.
    Returns an empty string when none are present.
    """
    return actor.get('slug') or actor.get('name') or str(actor.get('id') or '') or ''


PR_COMMENT_EVENTS = ('pr:comment:added', 'pr:comment:edited')


# =============================================================================
# Bitbucket Data Center view dataclasses
# =============================================================================


@dataclass
class BitbucketDCPR(ResolverViewInterface):
    """Base view representing a Bitbucket Data Center PR-level resolver trigger."""

    installation_id: str  # f"{project_key}/{repo_slug}" — DC has no per-hook UUID
    issue_number: int  # PR id (named issue_number for parity with the interface)
    project_key: str
    repo_slug: str
    full_repo_name: str  # f"{project_key}/{repo_slug}"
    is_public_repo: bool
    user_info: UserData
    raw_payload: Message
    conversation_id: str
    should_extract: bool
    send_summary_instruction: bool
    title: str
    description: str
    previous_comments: list[Comment]
    branch_name: str | None

    def _get_branch_name(self) -> str | None:
        return self.branch_name

    async def _load_resolver_context(self) -> None:
        bitbucket_service = BitbucketDCServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )
        self.previous_comments = await bitbucket_service.get_pr_comments(
            self.project_key, self.repo_slug, self.issue_number
        )
        (
            self.title,
            self.description,
        ) = await bitbucket_service.get_pr_title_and_body(
            self.project_key, self.repo_slug, self.issue_number
        )

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        await self._load_resolver_context()

        user_instructions_template = jinja_env.get_template('pr_update_prompt.j2')
        user_instructions = user_instructions_template.render(pr_comment='')

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

    async def initialize_new_conversation(self) -> UUID:
        self.resolved_org_id = await resolve_org_for_repo(
            provider=ProviderType.BITBUCKET_DATA_CENTER.value,
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

        from integrations.bitbucket_data_center.bitbucket_dc_v1_callback_processor import (
            BitbucketDCV1CallbackProcessor,
        )

        callback_processor = BitbucketDCV1CallbackProcessor(
            bitbucket_dc_view_data={
                'pr_id': self.issue_number,
                'project_key': self.project_key,
                'repo_slug': self.repo_slug,
                'full_repo_name': self.full_repo_name,
                'installation_id': self.installation_id,
                'keycloak_user_id': self.user_info.keycloak_user_id,
                'parent_comment_id': getattr(self, 'parent_comment_id', None),
            },
            should_request_summary=self.send_summary_instruction,
        )

        title = f'Bitbucket DC PR #{self.issue_number}: {self.title}'
        start_request = AppConversationStartRequest(
            conversation_id=conversation_id,
            system_message_suffix=conversation_instructions,
            initial_message=initial_message,
            selected_repository=self.full_repo_name,
            selected_branch=self._get_branch_name(),
            git_provider=ProviderType.BITBUCKET_DATA_CENTER,
            title=title,
            trigger=ConversationTrigger.RESOLVER,
            processors=[callback_processor],
        )

        injector_state = InjectorState()
        bitbucket_user_context = ResolverUserContext(
            saas_user_auth=saas_user_auth,
            resolver_org_id=self.resolved_org_id,
        )
        setattr(injector_state, USER_CONTEXT_ATTR, bitbucket_user_context)

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


@dataclass
class BitbucketDCPRComment(BitbucketDCPR):
    comment_body: str
    parent_comment_id: int | None

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
class BitbucketDCInlinePRComment(BitbucketDCPRComment):
    file_location: str
    line_number: int
    line_type: str  # 'ADDED' | 'CONTEXT' | 'REMOVED'
    file_type: str  # 'TO' | 'FROM'

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
            file_location=self.file_location,
            line_number=self.line_number,
        )
        return user_instructions, conversation_instructions


BitbucketDCViewType = BitbucketDCInlinePRComment | BitbucketDCPRComment | BitbucketDCPR


# =============================================================================
# Factory
# =============================================================================


class BitbucketDCFactory:
    """Inspect a Bitbucket Data Center webhook payload and decide which view to build.

    Bitbucket DC fires ``pr:comment:added`` (and ``pr:comment:edited``)
    when a PR receives a comment. The resolver activates when the comment
    body contains a case-insensitive mention of the configured
    ``@openhands`` handle. Inline comments include an ``anchor`` block
    with the file path and line number.
    """

    @staticmethod
    def is_pr_comment(message: Message, inline: bool = False) -> bool:
        event_key = message.message.get('event_key')
        payload = message.message.get('payload') or {}
        if event_key not in PR_COMMENT_EVENTS:
            return False
        comment = payload.get('comment') or {}
        body = comment.get('text') or ''
        if not has_exact_mention(body, INLINE_OH_LABEL):
            return False
        # DC indicates inline comments by presence of a ``commentAnchor``
        # alongside the comment, or an ``anchor`` block on the comment
        # itself in some payload variants.
        is_inline = bool(payload.get('commentAnchor') or comment.get('anchor'))
        return is_inline if inline else not is_inline

    @staticmethod
    async def create_bitbucket_dc_view_from_payload(
        message: Message, keycloak_user_id: str
    ) -> BitbucketDCViewType:
        """Build a view from a Bitbucket DC webhook payload.

        ``keycloak_user_id`` is the OpenHands user that **installed** the
        webhook (looked up by ``(project_key, repo_slug)`` from the
        ``bitbucket_dc_webhook`` table by the manager). The resolver acts
        on Bitbucket DC as the installer.
        """
        payload = cast(dict, message.message['payload'])
        installation_id = cast(str, message.message.get('installation_id') or '')

        actor = payload.get('actor') or {}
        actor_slug = extract_actor_slug(actor)
        username = actor.get('displayName') or actor.get('name') or 'unknown'

        pull_request = payload.get('pullRequest') or {}
        to_ref = pull_request.get('toRef') or {}
        repository = to_ref.get('repository') or {}
        project = repository.get('project') or {}
        project_key = project.get('key') or ''
        repo_slug = repository.get('slug') or ''
        full_repo_name = (
            f'{project_key}/{repo_slug}' if project_key and repo_slug else ''
        )
        is_public_repo = bool(repository.get('public'))

        raw_pr_id = pull_request.get('id')
        if not raw_pr_id:
            # ``is_pr_comment`` already validated the event key; a missing
            # PR id here means a malformed payload.
            raise ValueError(
                f'Invalid PR id in Bitbucket DC webhook payload: {pull_request}'
            )
        pr_id = int(raw_pr_id)
        from_ref = pull_request.get('fromRef') or {}
        branch_name = (
            from_ref.get('displayId')
            or (from_ref.get('id') or '').replace('refs/heads/', '')
            or None
        )

        user_info = UserData(
            user_id=actor_slug,
            username=username,
            keycloak_user_id=keycloak_user_id,
        )

        comment = payload.get('comment') or {}
        comment_body = comment.get('text') or ''
        parent_comment_id = (comment.get('parent') or {}).get('id')

        # DC carries the inline anchor at the payload level under
        # ``commentAnchor``; some variants nest it under ``comment.anchor``.
        anchor = payload.get('commentAnchor') or comment.get('anchor') or {}

        common_kwargs: dict = dict(
            installation_id=installation_id,
            issue_number=pr_id,
            project_key=project_key,
            repo_slug=repo_slug,
            full_repo_name=full_repo_name,
            is_public_repo=is_public_repo,
            user_info=user_info,
            raw_payload=message,
            conversation_id='',
            should_extract=True,
            send_summary_instruction=True,
            title='',
            description='',
            previous_comments=[],
            branch_name=branch_name,
        )

        if BitbucketDCFactory.is_pr_comment(message, inline=True):
            logger.info(
                f'[Bitbucket DC] Creating view for inline PR comment from '
                f'{username} in {full_repo_name}#{pr_id}'
            )
            return BitbucketDCInlinePRComment(
                **common_kwargs,
                comment_body=comment_body,
                parent_comment_id=parent_comment_id,
                file_location=anchor.get('path') or '',
                line_number=anchor.get('line') or 0,
                line_type=anchor.get('lineType') or 'CONTEXT',
                file_type=anchor.get('fileType') or 'TO',
            )

        if BitbucketDCFactory.is_pr_comment(message):
            logger.info(
                f'[Bitbucket DC] Creating view for PR comment from '
                f'{username} in {full_repo_name}#{pr_id}'
            )
            return BitbucketDCPRComment(
                **common_kwargs,
                comment_body=comment_body,
                parent_comment_id=parent_comment_id,
            )

        raise ValueError(f'Unhandled Bitbucket DC webhook event: {message}')
