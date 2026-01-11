from abc import ABC, abstractmethod

from integrations.types import SummaryExtractionTracker
from jinja2 import Environment
from storage.slack_user import SlackUser

from openhands.server.user_auth.user_auth import UserAuth


class SlackViewInterface(SummaryExtractionTracker, ABC):
    bot_access_token: str
    user_msg: str | None
    slack_user_id: str
    slack_to_openhands_user: SlackUser | None
    saas_user_auth: UserAuth | None
    channel_id: str
    message_ts: str
    thread_ts: str | None
    selected_repo: str | None
    should_extract: bool
    send_summary_instruction: bool
    conversation_id: str
    team_id: str
    v1_enabled: bool

    @abstractmethod
    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
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
