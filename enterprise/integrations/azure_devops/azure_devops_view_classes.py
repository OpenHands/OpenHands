"""Azure DevOps view classes for conversation initialization.

These classes implement the ResolverViewInterface to enable OpenHands job execution
for Azure DevOps work items and pull requests.
"""

from typing import Union

from integrations.models import Message
from integrations.types import ResolverViewInterface, UserData
from jinja2 import Environment
from pydantic.dataclasses import dataclass
from server.config import get_config
from storage.database import session_maker
from storage.saas_secrets_store import SaasSecretsStore

from openhands.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.integrations.service_types import Comment
from openhands.server.services.conversation_service import (
    initialize_conversation,
    start_conversation,
)
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)


@dataclass
class AzureDevOpsWorkItem(ResolverViewInterface):
    """View for Azure DevOps Work Item (Bug, Task, User Story, etc.) with @openhands mention or assignment."""

    work_item_id: int
    project_name: str
    organization: str
    full_repo_name: str  # Format: org/project/repo (or org/project if not linked)
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
    work_item_type: str  # Bug, Task, User Story, etc.
    selected_branch: str | None = None  # Branch from work item development section
    comment_body: str | None = (
        None  # Comment text when triggered by @mention in comment
    )
    repository_linked: bool = (
        True  # Whether work item has a linked repository in development section
    )

    async def _get_user_secrets(self):
        secrets_store = SaasSecretsStore(
            self.user_info.keycloak_user_id, session_maker, get_config()
        )
        user_secrets = await secrets_store.load()
        return user_secrets.custom_secrets if user_secrets else None

    async def _get_instructions(self, jinja_env: Environment):
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
            is_comment_trigger=self.comment_body is not None,
        )
        return user_instructions, conversation_instructions

    async def initialize_new_conversation(self) -> ConversationMetadata:
        """Initialize a new conversation for this work item."""
        conversation_metadata: ConversationMetadata = await initialize_conversation(  # type: ignore[assignment]
            user_id=self.user_info.keycloak_user_id,
            conversation_id=None,
            selected_repository=self.full_repo_name,
            selected_branch=self.selected_branch,
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
    ):
        """Create and start a new conversation for this work item."""
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


@dataclass
class AzureDevOpsPRComment(ResolverViewInterface):
    """View for Azure DevOps Pull Request comment with @openhands mention."""

    pr_id: int
    project_name: str
    organization: str
    repository_name: str
    full_repo_name: str  # Format: org/project/repo
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
    comment_body: str  # The triggering comment text
    is_inline: bool  # True if inline code review comment
    thread_context: dict | None  # File path and line position for inline comments
    branch_name: str | None  # PR source branch name

    async def _get_user_secrets(self):
        secrets_store = SaasSecretsStore(
            self.user_info.keycloak_user_id, session_maker, get_config()
        )
        user_secrets = await secrets_store.load()
        return user_secrets.custom_secrets if user_secrets else None

    async def _get_instructions(self, jinja_env: Environment):
        """Get user and conversation instructions for PR."""
        # Use template for user instructions (similar to GitHub)
        user_instructions_template = jinja_env.get_template('pr_update_prompt.j2')
        user_instructions = user_instructions_template.render(
            pr_comment=self.comment_body
        )

        # Extract file location and line number for inline comments
        file_location = None
        line_number = None
        if self.is_inline and self.thread_context:
            file_location = self.thread_context.get('filePath', 'unknown')
            right_line_end = self.thread_context.get('rightFileEnd', {})
            line_number = right_line_end.get('line', 'unknown')

        conversation_instructions_template = jinja_env.get_template(
            'pr_update_conversation_instructions.j2'
        )
        conversation_instructions = conversation_instructions_template.render(
            pr_number=self.pr_id,
            pr_title=self.title,
            pr_body=self.description,
            comments=self.previous_comments,
            branch_name=self.branch_name,
            file_location=file_location,
            line_number=line_number,
        )
        return user_instructions, conversation_instructions

    async def initialize_new_conversation(self) -> ConversationMetadata:
        """Initialize a new conversation for this PR."""
        conversation_metadata: ConversationMetadata = await initialize_conversation(  # type: ignore[assignment]
            user_id=self.user_info.keycloak_user_id,
            conversation_id=None,
            selected_repository=self.full_repo_name,
            selected_branch=self.branch_name,
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
    ):
        """Create and start a new conversation for this PR."""
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


# Type alias for all Azure DevOps view types
AzureDevOpsViewType = Union[AzureDevOpsWorkItem, AzureDevOpsPRComment]
