from abc import ABC, abstractmethod

from integrations.models import JobContext
from jinja2 import Environment
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.server.user_auth.user_auth import UserAuth


class JiraDcViewInterface(ABC):
    """Interface for Jira DC views that handle different types of Jira DC interactions."""

    job_context: JobContext
    saas_user_auth: UserAuth
    jira_dc_user: JiraDcUser
    jira_dc_workspace: JiraDcWorkspace
    selected_repo: str | None
    conversation_id: str

    @abstractmethod
    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Get initial instructions for the conversation."""
        pass

    @abstractmethod
    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Create or update a conversation and return the conversation ID."""
        pass

    @abstractmethod
    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira DC."""
        pass


class StartingConvoException(Exception):
    """Exception raised when starting a conversation fails."""

    pass
