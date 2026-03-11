from abc import ABC, abstractmethod
from dataclasses import dataclass

from integrations.types import SummaryExtractionTracker
from jinja2 import Environment
from storage.slack_user import SlackUser

from openhands.server.user_auth.user_auth import UserAuth


@dataclass
class SlackMessageView:
    """Minimal view for sending messages to Slack.

    This class contains only the fields needed to send messages,
    without requiring user authentication. Can be used directly for
    simple message operations or as a base class for more complex views.
    """

    bot_access_token: str
    slack_user_id: str
    channel_id: str
    message_ts: str
    thread_ts: str | None
    team_id: str

    def to_log_context(self) -> dict:
        """Return dict suitable for structured logging."""
        return {
            'slack_channel_id': self.channel_id,
            'slack_user_id': self.slack_user_id,
            'slack_team_id': self.team_id,
            'slack_thread_ts': self.thread_ts,
            'slack_message_ts': self.message_ts,
        }

    @classmethod
    async def from_payload(
        cls,
        payload: dict,
        slack_team_store,
    ) -> 'SlackMessageView | None':
        """Create a view from a raw Slack payload.

        This factory method handles the various payload formats from different
        Slack interactions (events, form submissions, block suggestions).

        Args:
            payload: Raw Slack payload dictionary
            slack_team_store: Store for retrieving bot tokens

        Returns:
            SlackMessageView if all required fields are available,
            None if required fields are missing or bot token unavailable.
        """
        from openhands.core.logger import openhands_logger as logger

        team_id = payload.get('team', {}).get('id') or payload.get('team_id')
        channel_id = (
            payload.get('container', {}).get('channel_id')
            or payload.get('channel', {}).get('id')
            or payload.get('channel_id')
        )
        user_id = payload.get('user', {}).get('id') or payload.get('slack_user_id')
        message_ts = payload.get('message_ts', '')
        thread_ts = payload.get('thread_ts')

        if not team_id or not channel_id or not user_id:
            logger.warning(
                'slack_message_view_from_payload_missing_fields',
                extra={
                    'has_team_id': bool(team_id),
                    'has_channel_id': bool(channel_id),
                    'has_user_id': bool(user_id),
                    'payload_keys': list(payload.keys()),
                },
            )
            return None

        bot_token = await slack_team_store.get_team_bot_token(team_id)
        if not bot_token:
            logger.warning(
                'slack_message_view_from_payload_no_bot_token',
                extra={'team_id': team_id},
            )
            return None

        return cls(
            bot_access_token=bot_token,
            slack_user_id=user_id,
            channel_id=channel_id,
            message_ts=message_ts,
            thread_ts=thread_ts,
            team_id=team_id,
        )


class SlackViewInterface(SlackMessageView, SummaryExtractionTracker, ABC):
    """Interface for authenticated Slack views that can create conversations.

    All fields are required (non-None) because this interface is only used
    for users who have linked their Slack account to OpenHands.

    Inherits from SlackMessageView:
        bot_access_token, slack_user_id, channel_id, message_ts, thread_ts, team_id
    """

    user_msg: str
    slack_to_openhands_user: SlackUser
    saas_user_auth: UserAuth
    selected_repo: str | None
    should_extract: bool
    send_summary_instruction: bool
    conversation_id: str
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
    """Raised when trying to send message to a conversation that is still starting up."""
