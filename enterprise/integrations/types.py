from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from jinja2 import Environment
from pydantic import BaseModel

if TYPE_CHECKING:
    from integrations.models import Message


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
        "Instructions passed when conversation is first initialized"
        raise NotImplementedError()

    async def create_new_conversation(
        self, jinja_env: Environment, *args: Any, **kwargs: Any
    ) -> Any:
        """Create a new conversation.

        Signature varies by provider implementation:
        - GitHub: (jinja_env, git_provider_tokens, conversation_metadata, saas_user_auth)
        - GitLab: (jinja_env, git_provider_tokens)
        """
        raise NotImplementedError()
