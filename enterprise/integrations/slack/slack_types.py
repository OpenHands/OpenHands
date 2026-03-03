from abc import ABC, abstractmethod

from integrations.types import SummaryExtractionTracker
from jinja2 import Environment
from storage.slack_user import SlackUser

from openhands.server.user_auth.user_auth import UserAuth


class SlackMessageView(ABC):
    """Minimal interface for sending messages to Slack.

    This base class contains only the fields needed to send messages,
    without requiring user authentication. Used by both authenticated
    and unauthenticated Slack views.
    """

    bot_access_token: str
    slack_user_id: str
    channel_id: str
    message_ts: str
    thread_ts: str | None


class SlackViewInterface(SlackMessageView, SummaryExtractionTracker, ABC):
    """Interface for authenticated Slack views that can create conversations.

    All fields are required (non-None) because this interface is only used
    for users who have linked their Slack account to OpenHands.
    """

    user_msg: str
    slack_to_openhands_user: SlackUser
    saas_user_auth: UserAuth
    selected_repo: str | None
    should_extract: bool
    send_summary_instruction: bool
    conversation_id: str
    team_id: str
    v1_enabled: bool

    @abstractmethod
    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized"""
        pass

    @abstractmethod
    async def create_or_update_conversation(self, jinja_env: Environment):
        """Create a new conversation"""
        pass

    @abstractmethod
    def get_response_msg(self) -> str:
        pass


class StartingConvoException(Exception):
    """Raised when trying to send message to a conversation that's is still starting up"""
