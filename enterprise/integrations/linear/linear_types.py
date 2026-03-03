from abc import ABC, abstractmethod

from integrations.models import JobContext
from jinja2 import Environment
from storage.linear_user import LinearUser
from storage.linear_workspace import LinearWorkspace

from openhands.server.user_auth.user_auth import UserAuth


class LinearViewInterface(ABC):
    """Interface for Linear views that handle different types of Linear interactions."""

    job_context: JobContext
    saas_user_auth: UserAuth
    linear_user: LinearUser
    linear_workspace: LinearWorkspace
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
        """Get the response message to send back to Linear."""
        pass


class StartingConvoException(Exception):
    """Exception raised when starting a conversation fails."""

    pass
