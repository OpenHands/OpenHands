"""Unit tests for SlackConversationStore."""

from unittest.mock import patch

import pytest
from sqlalchemy import select
from storage.slack_conversation import SlackConversation
from storage.slack_conversation_store import SlackConversationStore


@pytest.fixture
def slack_conversation_store():
    """Create SlackConversationStore instance."""
    return SlackConversationStore()


class TestSlackConversationStore:
    """Test cases for SlackConversationStore."""

    @pytest.mark.asyncio
    async def test_create_slack_conversation_persists_to_database(
        self, slack_conversation_store, async_session_maker
    ):
        """Test that create_slack_conversation actually stores data in the database.

        This test verifies that the await statement is present before session.merge().
        Without the await, the data won't be persisted and subsequent lookups will
        return None even though we just created the conversation.
        """
        channel_id = 'C123456'
        parent_id = '1234567890.123456'
        conversation_id = 'conv-test-123'
        keycloak_user_id = 'user-123'

        slack_conversation = SlackConversation(
            conversation_id=conversation_id,
            channel_id=channel_id,
            keycloak_user_id=keycloak_user_id,
            parent_id=parent_id,
        )

        with patch(
            'storage.slack_conversation_store.a_session_maker', async_session_maker
        ):
            # Create the slack conversation
            await slack_conversation_store.create_slack_conversation(slack_conversation)

            # Verify we can retrieve the conversation using the store method
            result = await slack_conversation_store.get_slack_conversation(
                channel_id=channel_id,
                parent_id=parent_id,
            )

        # This assertion would fail if the await was missing before session.merge()
        # because the data wouldn't be persisted to the database
        assert result is not None, (
            'Slack conversation was not persisted to the database. '
            'Ensure await is used before session.merge() in create_slack_conversation.'
        )
        assert result.conversation_id == conversation_id
        assert result.channel_id == channel_id
        assert result.parent_id == parent_id
        assert result.keycloak_user_id == keycloak_user_id

        # Also verify directly in the database
        async with async_session_maker() as session:
            db_result = await session.execute(
                select(SlackConversation).where(
                    SlackConversation.channel_id == channel_id,
                    SlackConversation.parent_id == parent_id,
                )
            )
            db_conversation = db_result.scalar_one_or_none()
            assert db_conversation is not None
            assert db_conversation.conversation_id == conversation_id
