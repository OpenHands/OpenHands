from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.models import OpenHandsModel, SendMessageRequest
from openhands.agent_server.utils import OpenHandsUUID, utc_now
from openhands.app_server.event_callback.event_callback_models import (
    EventCallbackProcessor,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.integrations.service_types import ProviderType
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.llm import MetricsSnapshot
from openhands.sdk.plugin import PluginSource
from openhands.storage.data_models.conversation_metadata import ConversationTrigger


class AgentType(Enum):
    """Agent type for conversation."""

    DEFAULT = 'default'
    PLAN = 'plan'


class PluginSpec(PluginSource):
    """Specification for loading a plugin into a conversation.

    Extends SDK's PluginSource with user-provided plugin configuration parameters.
    Inherits source, ref, and repo_path fields along with their validation.
    """

    parameters: dict[str, Any] | None = Field(
        default=None,
        description='User-provided values for plugin input parameters',
    )

    @property
    def display_name(self) -> str:
        """Extract a friendly display name from the plugin source.

        Examples:
            - 'github:owner/repo' -> 'repo'
            - 'https://github.com/owner/repo.git' -> 'repo.git'
            - '/local/path' -> 'path'
        """
        return self.source.split('/')[-1] if '/' in self.source else self.source

    def format_params_as_text(self, indent: str = '') -> str | None:
        """Format parameters as a readable text block for display.

        Args:
            indent: Optional prefix to add before each parameter line.

        Returns:
            Formatted parameters string, or None if no parameters.
        """
        if not self.parameters:
            return None
        return '\n'.join(
            f'{indent}- {key}: {value}' for key, value in self.parameters.items()
        )


class AppConversationInfo(BaseModel):
    """Conversation info which does not contain status."""

    id: OpenHandsUUID = Field(default_factory=uuid4)

    created_by_user_id: str | None
    sandbox_id: str

    selected_repository: str | None = None
    selected_branch: str | None = None
    git_provider: ProviderType | None = None
    title: str | None = None
    trigger: ConversationTrigger | None = None
    pr_number: list[int] = Field(default_factory=list)
    llm_model: str | None = None

    metrics: MetricsSnapshot | None = None

    parent_conversation_id: OpenHandsUUID | None = None
    sub_conversation_ids: list[OpenHandsUUID] = Field(default_factory=list)

    public: bool | None = None

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AppConversationSortOrder(Enum):
    CREATED_AT = 'CREATED_AT'
    CREATED_AT_DESC = 'CREATED_AT_DESC'
    UPDATED_AT = 'UPDATED_AT'
    UPDATED_AT_DESC = 'UPDATED_AT_DESC'
    TITLE = 'TITLE'
    TITLE_DESC = 'TITLE_DESC'


class AppConversationInfoPage(BaseModel):
    items: list[AppConversationInfo]
    next_page_id: str | None = None


class AppConversation(AppConversationInfo):  # type: ignore
    sandbox_status: SandboxStatus = Field(
        default=SandboxStatus.MISSING,
        description='Current sandbox status. Will be MISSING if the sandbox does not exist.',
    )
    execution_status: ConversationExecutionStatus | None = Field(
        default=None,
        description='Current agent status. Will be None if the sandbox_status is not RUNNING',
    )
    conversation_url: str | None = Field(
        default=None, description='The URL where the conversation may be accessed'
    )
    session_api_key: str | None = Field(
        default=None, description='The Session Api Key for REST operations.'
    )

    # JSON fields for complex data types
    pr_number: list[int] = Field(default_factory=list)
    metrics: MetricsSnapshot | None = Field(default=None)


class AppConversationPage(BaseModel):
    items: list[AppConversation]
    next_page_id: str | None = None


class AppConversationStartRequest(OpenHandsModel):
    """Start conversation request object.

    Although a user can go directly to the sandbox and start conversations, they
    would need to manually supply required startup parameters such as LLM key. Starting
    from the app server copies these from the user info.
    """

    sandbox_id: str | None = Field(default=None)
    conversation_id: UUID | None = Field(default=None)
    initial_message: SendMessageRequest | None = None
    system_message_suffix: str | None = None
    processors: list[EventCallbackProcessor] | None = Field(default=None)
    llm_model: str | None = None

    # Git parameters
    selected_repository: str | None = None
    selected_branch: str | None = None
    git_provider: ProviderType | None = None
    title: str | None = None
    trigger: ConversationTrigger | None = None
    pr_number: list[int] = Field(default_factory=list)
    parent_conversation_id: OpenHandsUUID | None = None
    agent_type: AgentType = Field(default=AgentType.DEFAULT)

    public: bool | None = None

    # Plugin parameters - for loading remote plugins into the conversation
    plugins: list[PluginSpec] | None = Field(
        default=None,
        description=(
            'List of plugins to load for this conversation. Plugins are loaded '
            'and their skills/MCP config are merged into the agent.'
        ),
    )


class AppConversationUpdateRequest(BaseModel):
    public: bool


class AppConversationStartTaskStatus(Enum):
    WORKING = 'WORKING'
    WAITING_FOR_SANDBOX = 'WAITING_FOR_SANDBOX'
    PREPARING_REPOSITORY = 'PREPARING_REPOSITORY'
    RUNNING_SETUP_SCRIPT = 'RUNNING_SETUP_SCRIPT'
    SETTING_UP_GIT_HOOKS = 'SETTING_UP_GIT_HOOKS'
    SETTING_UP_SKILLS = 'SETTING_UP_SKILLS'
    STARTING_CONVERSATION = 'STARTING_CONVERSATION'
    READY = 'READY'
    ERROR = 'ERROR'


class AppConversationStartTaskSortOrder(Enum):
    CREATED_AT = 'CREATED_AT'
    CREATED_AT_DESC = 'CREATED_AT_DESC'
    UPDATED_AT = 'UPDATED_AT'
    UPDATED_AT_DESC = 'UPDATED_AT_DESC'


class AppConversationStartTask(OpenHandsModel):
    """Object describing the start process for an app conversation.

    Because starting an app conversation can be slow (And can involve starting a sandbox),
    we kick off a background task for it. Once the conversation is started, the app_conversation_id
    is populated.
    """

    id: OpenHandsUUID = Field(default_factory=uuid4)
    created_by_user_id: str | None
    status: AppConversationStartTaskStatus = AppConversationStartTaskStatus.WORKING
    detail: str | None = None
    app_conversation_id: OpenHandsUUID | None = Field(
        default=None, description='The id of the app_conversation, if READY'
    )
    sandbox_id: str | None = Field(
        default=None, description='The id of the sandbox, if READY'
    )
    agent_server_url: str | None = Field(
        default=None, description='The agent server url, if READY'
    )
    request: AppConversationStartRequest
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AppConversationStartTaskPage(OpenHandsModel):
    items: list[AppConversationStartTask]
    next_page_id: str | None = None


class SkillResponse(BaseModel):
    """Response model for skills endpoint."""

    name: str
    type: Literal['repo', 'knowledge', 'agentskills']
    content: str
    triggers: list[str] = []
