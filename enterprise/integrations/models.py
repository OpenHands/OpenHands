from enum import Enum
from typing import Any

from pydantic import BaseModel


class SourceType(str, Enum):
    GITHUB = 'github'
    GITLAB = 'gitlab'
    BITBUCKET = 'bitbucket'
    BITBUCKET_DATA_CENTER = 'bitbucket_data_center'
    OPENHANDS = 'openhands'
    SLACK = 'slack'
    JIRA = 'jira'
    JIRA_DC = 'jira_dc'
    LINEAR = 'linear'


class Message(BaseModel):
    """Message model for incoming webhook payloads from integrations.

    Note: This model is intended for INCOMING messages only.
    For outgoing messages (e.g., sending comments to GitHub/GitLab),
    pass strings directly to the send_message methods instead of
    wrapping them in a Message object.
    """

    source: SourceType
    message: dict[str, Any]
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
