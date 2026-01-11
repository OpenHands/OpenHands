from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import sessionmaker
from storage.database import session_maker
from storage.slack_conversation import SlackConversation


@dataclass
class SlackConversationStore:
    session_maker: sessionmaker

    async def get_slack_conversation(
        self, channel_id: str, parent_id: str
    ) -> SlackConversation | None:
        """Get a slack conversation by channel_id and message_ts.
        Both parameters are required to match for a conversation to be returned.
        """
        with session_maker() as session:
            conversation = (
                session.query(SlackConversation)
                .filter(SlackConversation.channel_id == channel_id)
                .filter(SlackConversation.parent_id == parent_id)
                .first()
            )

            return conversation

    async def create_slack_conversation(
        self, slack_converstion: SlackConversation
    ) -> None:
        with self.session_maker() as session:
            session.merge(slack_converstion)
            session.commit()

    @classmethod
    def get_instance(cls) -> SlackConversationStore:
        return SlackConversationStore(session_maker)
