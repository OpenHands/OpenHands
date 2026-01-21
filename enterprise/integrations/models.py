from enum import Enum

from pydantic import BaseModel

from openhands.core.schema import AgentState


class SourceType(str, Enum):
    GITHUB = 'github'
    GITLAB = 'gitlab'
    AZURE_DEVOPS = 'azure_devops'
    OPENHANDS = 'openhands'
    SLACK = 'slack'
    JIRA = 'jira'
    JIRA_DC = 'jira_dc'
    LINEAR = 'linear'


class Message(BaseModel):
    source: SourceType
    message: str | dict
    ephemeral: bool = False


class JobContext(BaseModel):
    issue_id: str
    issue_key: str
    user_msg: str
    user_email: str
    display_name: str
    platform_user_id: str = ''
    workspace_name: str
    base_api_url: str = ''
    issue_title: str = ''
    issue_description: str = ''


class JobResult:
    result: str
    explanation: str


class GithubResolverJob:
    type: SourceType
    status: AgentState
    result: JobResult
    owner: str
    repo: str
    installation_token: str
    issue_number: int
    runtime_id: int
    created_at: int
    completed_at: int
