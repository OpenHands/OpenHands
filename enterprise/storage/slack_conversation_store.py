from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from storage.database import a_session_maker
from storage.slack_conversation import SlackConversation


@dataclass
class SlackConversationStore:
    async def get_slack_conversation(
        self, channel_id: str, parent_id: str
    ) -> SlackConversation | None:
        """Get a slack conversation by channel_id and message_ts.
        Both parameters are required to match for a conversation to be returned.
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(SlackConversation).where(
                    SlackConversation.channel_id == channel_id,
                    SlackConversation.parent_id == parent_id,
                )
            )
            return result.scalar_one_or_none()

    async def create_slack_conversation(
        self, slack_conversation: SlackConversation
    ) -> None:
        async with a_session_maker() as session:
            await session.merge(slack_conversation)
            await session.commit()

    @classmethod
    def get_instance(cls) -> SlackConversationStore:
        return SlackConversationStore()
