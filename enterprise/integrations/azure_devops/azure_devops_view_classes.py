"""Azure DevOps view classes for conversation initialization.

These classes implement the ResolverViewInterface to enable OpenHands job execution
for Azure DevOps work items and pull requests.

Class hierarchy (following GitHub's pattern):
- AzureDevOpsBase: Common fields and methods (abstract)
  - AzureDevOpsWorkItem: Work item (Bug, Task, User Story, etc.)
  - AzureDevOpsPRComment: General PR discussion comment
    - AzureDevOpsInlinePRComment: Inline code review comment with file/line context
"""

from abc import abstractmethod
from typing import Union
from uuid import UUID, uuid4

from integrations.models import Message
from integrations.resolver_context import ResolverUserContext
from integrations.types import ResolverViewInterface, UserData
from integrations.utils import (
    ENABLE_V1_AZURE_DEVOPS_RESOLVER,
    get_user_v1_enabled_setting,
)
from jinja2 import Environment
from pydantic.dataclasses import dataclass
from server.config import get_config
from storage.database import session_maker
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


async def is_v1_enabled_for_azure_devops_resolver(user_id: str | None) -> bool:
    """Check if V1 is enabled for Azure DevOps resolver for the given user.

    Args:
        user_id: The keycloak user ID

    Returns:
        True if V1 is enabled for both the user and the resolver feature flag
    """
    if not user_id:
        return False
    return (
        await get_user_v1_enabled_setting(user_id) and ENABLE_V1_AZURE_DEVOPS_RESOLVER
    )


@dataclass
class AzureDevOpsBase(ResolverViewInterface):
    """Base class for Azure DevOps views with common fields and methods.

    This class contains all shared functionality between work items and PR comments,
    eliminating code duplication across view types.
    """

    # Common identifiers
    organization: str
    project_name: str
    full_repo_name: str  # Format: org/project/repo

    # Repository info
    is_public_repo: bool

    # User and payload
    user_info: UserData
    raw_payload: Message

    # Conversation tracking
    conversation_id: str
    uuid: str | None

    # Behavior flags
    should_extract: bool
    send_summary_instruction: bool

    # Content
    title: str
    description: str
    previous_comments: list[Comment]

    # Required by ResolverViewInterface (Azure DevOps doesn't have installation_id)
    # Callers must pass 0 explicitly since Azure DevOps doesn't use installation_id
    installation_id: int
    # issue_number is set to work_item_id or pr_id by callers
    issue_number: int

    # V1 support (set by initialize_new_conversation)
    v1_enabled: bool

    async def _get_user_secrets(self):
        """Get user secrets from the SaaS secrets store."""
        secrets_store = SaasSecretsStore(
            self.user_info.keycloak_user_id, session_maker, get_config()
        )
        user_secrets = await secrets_store.load()
        return user_secrets.custom_secrets if user_secrets else None

    @abstractmethod
    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Get user and conversation instructions. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_conversation_title(self) -> str:
        """Get the conversation title. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_selected_branch(self) -> str | None:
        """Get the selected branch name. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _create_azure_devops_v1_callback_processor(self):
        """Create a V1 callback processor. Must be implemented by subclasses."""
        pass

    async def initialize_new_conversation(self) -> ConversationMetadata:
        """Initialize a new conversation for this view."""
        # Check if V1 is enabled for this user
        self.v1_enabled = await is_v1_enabled_for_azure_devops_resolver(
            self.user_info.keycloak_user_id
        )

        logger.info(
            f'[Azure DevOps V1]: User flag found for {self.user_info.keycloak_user_id} is {self.v1_enabled}'
        )

        selected_branch = self._get_selected_branch()

        if self.v1_enabled:
            # V1: Create dummy metadata, don't save to V0 conversation store
            # V1 conversations are stored in a separate table
            self.conversation_id = uuid4().hex
            return ConversationMetadata(
                trigger=ConversationTrigger.RESOLVER,
                conversation_id=self.conversation_id,
                title=self._get_conversation_title(),
                user_id=self.user_info.keycloak_user_id,
                selected_repository=self.full_repo_name,
                selected_branch=selected_branch,
                git_provider=ProviderType.AZURE_DEVOPS,
            )

        # V0: Use existing logic
        conversation_metadata: ConversationMetadata = await initialize_conversation(  # type: ignore[assignment]
            user_id=self.user_info.keycloak_user_id,
            conversation_id=None,
            selected_repository=self.full_repo_name,
            selected_branch=selected_branch,
            conversation_trigger=ConversationTrigger.RESOLVER,
            git_provider=ProviderType.AZURE_DEVOPS,
        )
        self.conversation_id = conversation_metadata.conversation_id
        return conversation_metadata

    async def create_new_conversation(
        self,
        jinja_env: Environment,
        git_provider_tokens: PROVIDER_TOKEN_TYPE,
        conversation_metadata: ConversationMetadata,
        saas_user_auth: UserAuth | None = None,
    ):
        """Create and start a new conversation for this view."""
        logger.info(
            f'[Azure DevOps V1]: User flag found for {self.user_info.keycloak_user_id} is {self.v1_enabled}'
        )

        if self.v1_enabled:
            if saas_user_auth is None:
                raise ValueError('saas_user_auth is required for V1 conversations')
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
        logger.info('[Azure DevOps V1]: Creating V1 conversation')

        user_instructions, conversation_instructions = await self._get_instructions(
            jinja_env
        )

        # Create the initial message request
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=user_instructions)]
        )

        # Create the Azure DevOps V1 callback processor
        azure_devops_callback_processor = (
            self._create_azure_devops_v1_callback_processor()
        )

        # Get the app conversation service and start the conversation
        injector_state = InjectorState()

        # Create the V1 conversation start request with the callback processor
        start_request = AppConversationStartRequest(
            conversation_id=UUID(conversation_metadata.conversation_id),
            system_message_suffix=conversation_instructions,
            initial_message=initial_message,
            selected_repository=self.full_repo_name,
            selected_branch=self._get_selected_branch(),
            git_provider=ProviderType.AZURE_DEVOPS,
            title=self._get_conversation_title(),
            trigger=ConversationTrigger.RESOLVER,
            processors=[azure_devops_callback_processor],
        )

        # Set up the Azure DevOps user context for the V1 system
        azure_devops_user_context = ResolverUserContext(saas_user_auth=saas_user_auth)
        setattr(injector_state, USER_CONTEXT_ATTR, azure_devops_user_context)

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


# Common default/protected branch names
DEFAULT_BRANCH_NAMES = frozenset(
    {
        'main',
        'master',
        'develop',
        'development',
        'dev',
        'trunk',
        'release',
        'production',
        'prod',
        'staging',
    }
)


def is_default_branch(branch_name: str | None) -> bool:
    """Check if a branch name is a default/protected branch.

    Args:
        branch_name: The branch name to check

    Returns:
        True if the branch is a default/protected branch
    """
    if not branch_name:
        return False
    # Normalize branch name (lowercase, strip refs/heads/ prefix if present)
    normalized = branch_name.lower().strip()
    if normalized.startswith('refs/heads/'):
        normalized = normalized[11:]
    return normalized in DEFAULT_BRANCH_NAMES


@dataclass
class AzureDevOpsWorkItem(AzureDevOpsBase):
    """View for Azure DevOps Work Item (Bug, Task, User Story, etc.) with @openhands mention or assignment."""

    work_item_id: int
    work_item_type: str  # Bug, Task, User Story, etc.
    selected_branch: str | None = None  # Branch from work item development section
    comment_body: str | None = (
        None  # Comment text when triggered by @mention in comment
    )
    repository_linked: bool = (
        True  # Whether work item has a linked repository in development section
    )

    def _get_conversation_title(self) -> str:
        """Get the conversation title for a work item."""
        return f'Azure DevOps Work Item #{self.work_item_id}: {self.title}'

    def _get_selected_branch(self) -> str | None:
        """Get the selected branch for a work item."""
        return self.selected_branch

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Get user and conversation instructions for work item."""
        # Use Jinja template for user instructions (like GitHub does)
        user_instructions_template = jinja_env.get_template('issue_prompt.j2')

        # If triggered by comment, use the comment text
        # Otherwise, use default message to fix the work item
        user_instructions = user_instructions_template.render(
            issue_comment=self.comment_body, issue_number=self.work_item_id
        )

        # Add work item context
        context = f"""Please address this Azure DevOps {self.work_item_type}:
Title: {self.title}
Description: {self.description}

Repository: {self.full_repo_name}
Work Item URL: https://dev.azure.com/{self.organization}/{self.project_name}/_workitems/edit/{self.work_item_id}
"""
        user_instructions = context + '\n' + user_instructions

        conversation_instructions_template = jinja_env.get_template(
            'issue_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            issue_number=self.work_item_id,
            issue_title=self.title,
            issue_body=self.description,
            previous_comments=self.previous_comments,
            selected_branch=self.selected_branch,
            repository_linked=self.repository_linked,
            is_default_branch=is_default_branch(self.selected_branch),
        )
        return user_instructions, conversation_instructions

    def _create_azure_devops_v1_callback_processor(self):
        """Create a V1 callback processor for work item."""
        from server.conversation_callback_processor.azure_devops_v1_callback_processor import (
            AzureDevOpsV1CallbackProcessor,
        )

        return AzureDevOpsV1CallbackProcessor(
            azure_devops_view_data={
                'work_item_id': self.work_item_id,
                'full_repo_name': self.full_repo_name,
                'organization': self.organization,
                'project_name': self.project_name,
            },
            should_request_summary=self.send_summary_instruction,
            is_pr_comment=False,
            thread_id=None,
        )


@dataclass
class AzureDevOpsPRComment(AzureDevOpsBase):
    """View for Azure DevOps Pull Request general discussion comment with @openhands mention."""

    pr_id: int
    repository_name: str
    branch_name: str | None  # PR source branch name
    thread_id: int | None = None  # Thread ID for replying to the original thread

    def _get_conversation_title(self) -> str:
        """Get the conversation title for a PR comment."""
        return f'Azure DevOps PR #{self.pr_id}: {self.title}'

    def _get_selected_branch(self) -> str | None:
        """Get the selected branch for a PR comment."""
        return self.branch_name

    def _get_pr_url(self) -> str:
        """Get the PR URL."""
        return f'https://dev.azure.com/{self.organization}/{self.project_name}/_git/{self.repository_name}/pullrequest/{self.pr_id}'

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Get user and conversation instructions for PR discussion comment."""
        user_instructions = f"""Please address this Azure DevOps Pull Request discussion comment:
PR #{self.pr_id}: {self.title}
Description: {self.description}

Repository: {self.full_repo_name}
PR URL: {self._get_pr_url()}
"""

        # Add previous comments if any
        if self.previous_comments:
            user_instructions += '\n\nPrevious comments:\n'
            for comment in self.previous_comments:
                user_instructions += f'- {comment.author}: {comment.body}\n'

        conversation_instructions_template = jinja_env.get_template(
            'pr_update_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            pr_number=self.pr_id,
            pr_title=self.title,
            pr_body=self.description,
            previous_comments=self.previous_comments,
            branch_name=self.branch_name,
        )
        return user_instructions, conversation_instructions

    def _create_azure_devops_v1_callback_processor(self):
        """Create a V1 callback processor for PR comment."""
        from server.conversation_callback_processor.azure_devops_v1_callback_processor import (
            AzureDevOpsV1CallbackProcessor,
        )

        return AzureDevOpsV1CallbackProcessor(
            azure_devops_view_data={
                'pr_id': self.pr_id,
                'full_repo_name': self.full_repo_name,
                'organization': self.organization,
                'project_name': self.project_name,
                'repository_name': self.repository_name,
            },
            should_request_summary=self.send_summary_instruction,
            is_pr_comment=True,
            thread_id=self.thread_id,
        )


@dataclass
class AzureDevOpsInlinePRComment(AzureDevOpsPRComment):
    """View for Azure DevOps Pull Request inline code review comment with file/line context."""

    thread_context: dict | None = (
        None  # File path and line position for inline comments
    )

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Get user and conversation instructions for inline PR comment."""
        user_instructions = f"""Please address this Azure DevOps Pull Request inline code review comment:
PR #{self.pr_id}: {self.title}
Description: {self.description}

Repository: {self.full_repo_name}
PR URL: {self._get_pr_url()}
"""

        # Add inline comment location context
        if self.thread_context:
            file_path = self.thread_context.get('filePath', 'unknown')
            right_line = self.thread_context.get('rightFileEnd', {}).get(
                'line', 'unknown'
            )
            user_instructions += (
                f'\nInline comment location: {file_path}:{right_line}\n'
            )

        # Add previous comments if any
        if self.previous_comments:
            user_instructions += '\n\nPrevious comments:\n'
            for comment in self.previous_comments:
                user_instructions += f'- {comment.author}: {comment.body}\n'

        conversation_instructions_template = jinja_env.get_template(
            'pr_update_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            pr_number=self.pr_id,
            pr_title=self.title,
            pr_body=self.description,
            previous_comments=self.previous_comments,
            branch_name=self.branch_name,
            file_location=self.thread_context.get('filePath')
            if self.thread_context
            else None,
            line_number=self.thread_context.get('rightFileEnd', {}).get('line')
            if self.thread_context
            else None,
        )
        return user_instructions, conversation_instructions


# Type alias for all Azure DevOps view types
AzureDevOpsViewType = Union[
    AzureDevOpsWorkItem, AzureDevOpsPRComment, AzureDevOpsInlinePRComment
]
