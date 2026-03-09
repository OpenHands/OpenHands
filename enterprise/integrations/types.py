from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from jinja2 import Environment
from pydantic import BaseModel

if TYPE_CHECKING:
    from integrations.models import Message

    from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
    from openhands.server.user_auth.user_auth import UserAuth
    from openhands.storage.data_models.conversation_metadata import ConversationMetadata


class GitLabResourceType(Enum):
    GROUP = 'group'
    SUBGROUP = 'subgroup'
    PROJECT = 'project'


class PRStatus(Enum):
    CLOSED = 'CLOSED'
    MERGED = 'MERGED'


class UserData(BaseModel):
    user_id: int
    username: str
    keycloak_user_id: str


class IncidentComponent(BaseModel):
    id: str
    name: str
    groupname: str | None = None
    current_status: str


class Incident(BaseModel):
    id: str
    name: str
    status: str
    url: str
    last_update_at: str
    last_update_message: str
    current_worst_impact: str
    affected_components: list[IncidentComponent]


class Maintenance(BaseModel):
    id: str
    name: str
    status: str
    last_update_at: str
    last_update_message: str
    url: str
    affected_components: list[IncidentComponent]
    starts_at: str | None = None
    ends_at: str | None = None
    started_at: str | None = None
    scheduled_end_at: str


class WidgetResponse(BaseModel):
    ongoing_incidents: list[Incident]
    in_progress_maintenances: list[Maintenance]
    scheduled_maintenances: list[Maintenance]


@dataclass
class SummaryExtractionTracker:
    conversation_id: str
    should_extract: bool
    send_summary_instruction: bool


@dataclass
class ResolverViewInterface(SummaryExtractionTracker):
    # installation_id type varies by provider:
    # - GitHub: int (GitHub App installation ID)
    # - GitLab: str (webhook installation ID from our DB)
    installation_id: int | str
    user_info: UserData
    issue_number: int
    full_repo_name: str
    is_public_repo: bool
    raw_payload: 'Message'

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized."""
        raise NotImplementedError()

    async def initialize_new_conversation(self) -> 'ConversationMetadata':
        """Initialize a new conversation and return metadata.

        For V1 conversations, creates a dummy ConversationMetadata.
        For V0 conversations, initializes through the conversation store.
        """
        raise NotImplementedError()

    async def create_new_conversation(
        self,
        jinja_env: Environment,
        git_provider_tokens: 'PROVIDER_TOKEN_TYPE',
        conversation_metadata: 'ConversationMetadata',
        saas_user_auth: 'UserAuth',
    ) -> None:
        """Create a new conversation.

        Args:
            jinja_env: Jinja2 environment for template rendering
            git_provider_tokens: Token mapping for git providers
            conversation_metadata: Metadata for the conversation
            saas_user_auth: User authentication for SaaS
        """
        raise NotImplementedError()
