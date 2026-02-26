"""Type definitions and interfaces for Jira integration."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from jinja2 import Environment
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace

from openhands.server.user_auth.user_auth import UserAuth

if TYPE_CHECKING:
    from integrations.jira.jira_payload import JiraWebhookPayload


class JiraViewInterface(ABC):
    """Interface for Jira views that handle different types of Jira interactions.

    Views hold the webhook payload directly rather than duplicating fields,
    and fetch issue details lazily when needed.
    """

    # Core data - view holds these references
    payload: 'JiraWebhookPayload'
    saas_user_auth: UserAuth
    jira_user: JiraUser
    jira_workspace: JiraWorkspace

    # Mutable state set during processing
    selected_repo: str | None
    conversation_id: str

    @abstractmethod
    async def get_issue_details(self) -> tuple[str, str]:
        """Fetch and cache issue title and description from Jira API.

        Returns:
            Tuple of (issue_title, issue_description)
        """
        pass

    @abstractmethod
    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Create or update a conversation and return the conversation ID."""
        pass

    @abstractmethod
    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira."""
        pass


class StartingConvoException(Exception):
    """Exception raised when starting a conversation fails.

    This provides user-friendly error messages that can be sent back to Jira.
    """

    pass


class RepositoryNotFoundError(Exception):
    """Raised when a repository cannot be determined from the issue.

    This is a separate error domain from StartingConvoException - it represents
    a precondition failure (no repo configured/found) rather than a conversation
    creation failure. The manager catches this and converts it to a user-friendly
    message.
    """

    pass
