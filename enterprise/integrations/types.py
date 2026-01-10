from dataclasses import dataclass
from enum import Enum

from jinja2 import Environment
from pydantic import BaseModel


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
    installation_id: int
    user_info: UserData
    issue_number: int
    full_repo_name: str
    is_public_repo: bool
    raw_payload: dict

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        "Instructions passed when conversation is first initialized"
        raise NotImplementedError()

    async def create_new_conversation(self, jinja_env: Environment, token: str):
        "Create a new conversation"
        raise NotImplementedError()
