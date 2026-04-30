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
from openhands.app_server.integrations.bitbucket.bitbucket_service import (
    BitBucketServiceImpl,
)
from openhands.app_server.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderType,
)
from openhands.app_server.integrations.service_types import Comment
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.app_server.user_auth.user_auth import UserAuth
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk import TextContent

OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)

PR_COMMENT_EVENT = 'pullrequest:comment_created'


# =============================================================================
# Bitbucket Cloud view dataclasses
# =============================================================================


@dataclass
class BitbucketPR(ResolverViewInterface):
    """Base view representing a Bitbucket Cloud PR-level resolver trigger."""

    installation_id: str  # webhook UUID stored in DB
    issue_number: (
        int  # Bitbucket PR id (named issue_number for parity with the interface)
    )
    workspace: str
    repo_slug: str
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
    branch_name: str | None

    def _get_branch_name(self) -> str | None:
        return self.branch_name

    async def _load_resolver_context(self) -> None:
        bitbucket_service = BitBucketServiceImpl(
            external_auth_id=self.user_info.keycloak_user_id
        )
        self.previous_comments = await bitbucket_service.get_pr_comments(
            self.workspace, self.repo_slug, self.issue_number
        )
        (
            self.title,
            self.description,
        ) = await bitbucket_service.get_pr_title_and_body(
            self.workspace, self.repo_slug, self.issue_number
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
            provider=ProviderType.BITBUCKET.value,
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

        from integrations.bitbucket.bitbucket_v1_callback_processor import (
            BitbucketV1CallbackProcessor,
        )

        callback_processor = BitbucketV1CallbackProcessor(
            bitbucket_view_data={
                'pr_id': self.issue_number,
                'workspace': self.workspace,
                'repo_slug': self.repo_slug,
                'full_repo_name': self.full_repo_name,
                'installation_id': self.installation_id,
                'keycloak_user_id': self.user_info.keycloak_user_id,
                'parent_comment_id': getattr(self, 'parent_comment_id', None),
            },
            should_request_summary=self.send_summary_instruction,
        )

        title = f'Bitbucket PR #{self.issue_number}: {self.title}'
        start_request = AppConversationStartRequest(
            conversation_id=conversation_id,
            system_message_suffix=conversation_instructions,
            initial_message=initial_message,
            selected_repository=self.full_repo_name,
            selected_branch=self._get_branch_name(),
            git_provider=ProviderType.BITBUCKET,
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
class BitbucketPRComment(BitbucketPR):
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
class BitbucketInlinePRComment(BitbucketPRComment):
    file_location: str
    line_number: int

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


BitbucketViewType = BitbucketInlinePRComment | BitbucketPRComment | BitbucketPR


# =============================================================================
# Factory
# =============================================================================


class BitbucketFactory:
    """Inspect a Bitbucket Cloud webhook payload and decide which view to build.

    Bitbucket Cloud has no labels on PRs, so the only resolver trigger today
    is a ``pullrequest:comment_created`` event whose comment body contains a
    case-insensitive mention of the configured ``@openhands`` handle.
    Bitbucket Cloud Issues were sunset by Atlassian in 2024, so there is
    no issue-comment trigger.
    """

    @staticmethod
    def is_pr_comment(message: Message, inline: bool = False) -> bool:
        event_key = message.message.get('event_key')
        payload = message.message.get('payload') or {}
        if event_key != PR_COMMENT_EVENT:
            return False
        comment = payload.get('comment') or {}
        body = (comment.get('content') or {}).get('raw') or ''
        if not has_exact_mention(body, INLINE_OH_LABEL):
            return False
        is_inline = bool(comment.get('inline'))
        return is_inline if inline else not is_inline

    @staticmethod
    async def create_bitbucket_view_from_payload(
        message: Message, keycloak_user_id: str
    ) -> BitbucketViewType:
        """Build a view from a webhook payload.

        ``keycloak_user_id`` is the OpenHands user that **installed** the
        webhook (looked up from the ``bitbucket_webhook`` table by the
        manager). The resolver acts on Bitbucket as the installer, mirroring
        the GitHub-App "runs as installation" model — Bitbucket Cloud's
        built-in IdP cannot populate a per-actor ``bitbucket_id`` Keycloak
        attribute, so we cannot reliably map a comment author back to a
        Keycloak user.
        """
        payload = cast(dict, message.message['payload'])
        installation_id = cast(str, message.message.get('installation_id') or '')

        actor = payload.get('actor') or {}
        # Prefer Bitbucket Cloud's stable ``account_id`` over ``uuid``: ``uuid``
        # is brace-wrapped (``"{9339e48a-…}"``) and the canonical user
        # identifier per Atlassian. ``user_info.user_id`` carries the
        # commenter's Bitbucket id for reference; auth runs through the
        # installer-scoped ``keycloak_user_id``.
        actor_id = (
            actor.get('account_id') or (actor.get('uuid') or '').strip('{}') or ''
        )
        username = actor.get('display_name') or actor.get('nickname') or 'unknown'

        repository = payload.get('repository') or {}
        full_repo_name = repository.get('full_name') or ''
        workspace, _, repo_slug = full_repo_name.partition('/')
        is_public_repo = not repository.get('is_private', True)

        pull_request = payload.get('pullrequest') or {}
        raw_pr_id = pull_request.get('id')
        if not raw_pr_id:
            # ``is_pr_comment`` already validated the event key, so a
            # missing/zero PR id here means a malformed payload from the
            # webhook sender — fail fast with a clear error rather than
            # building a view with ``issue_number=0`` whose downstream
            # API calls would 404 with a less helpful trace.
            raise ValueError(
                f'Invalid PR id in Bitbucket webhook payload: {pull_request}'
            )
        pr_id = int(raw_pr_id)
        branch_name = ((pull_request.get('source') or {}).get('branch') or {}).get(
            'name'
        )

        user_info = UserData(
            user_id=actor_id,
            username=username,
            keycloak_user_id=keycloak_user_id,
        )

        comment = payload.get('comment') or {}
        comment_body = (comment.get('content') or {}).get('raw') or ''
        parent_comment_id = (comment.get('parent') or {}).get('id')
        inline_info = comment.get('inline') or {}

        common_kwargs: dict = dict(
            installation_id=installation_id,
            issue_number=pr_id,
            workspace=workspace,
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

        if BitbucketFactory.is_pr_comment(message, inline=True):
            logger.info(
                f'[Bitbucket] Creating view for inline PR comment from '
                f'{username} in {full_repo_name}#{pr_id}'
            )
            return BitbucketInlinePRComment(
                **common_kwargs,
                comment_body=comment_body,
                parent_comment_id=parent_comment_id,
                file_location=inline_info.get('path') or '',
                line_number=inline_info.get('to') or inline_info.get('from') or 0,
            )

        if BitbucketFactory.is_pr_comment(message):
            logger.info(
                f'[Bitbucket] Creating view for PR comment from '
                f'{username} in {full_repo_name}#{pr_id}'
            )
            return BitbucketPRComment(
                **common_kwargs,
                comment_body=comment_body,
                parent_comment_id=parent_comment_id,
            )

        raise ValueError(f'Unhandled Bitbucket webhook event: {message}')
