"""Tests for SQLPendingMessageService.

This module tests the SQL implementation of PendingMessageService,
covering message queuing, retrieval, counting, deletion, and
conversation_id updates using SQLite as a mock database.
"""

from typing import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.agent_server.models import TextContent
from openhands.app_server.pending_messages.pending_message_models import (
    PendingMessageResponse,
)
from openhands.app_server.pending_messages.pending_message_service import (
    SQLPendingMessageService,
)
from openhands.app_server.utils.sql_utils import Base


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as db_session:
        yield db_session


@pytest.fixture
def service(async_session) -> SQLPendingMessageService:
    """Create a SQLPendingMessageService instance for testing."""
    return SQLPendingMessageService(db_session=async_session)


@pytest.fixture
def sample_content() -> list[TextContent]:
    """Create sample message content for testing."""
    return [TextContent(text='Hello, this is a test message')]


class TestSQLPendingMessageService:
    """Test suite for SQLPendingMessageService."""

    @pytest.mark.asyncio
    async def test_add_message_creates_message_with_correct_data(
        self,
        service: SQLPendingMessageService,
        sample_content: list[TextContent],
    ):
        """Test that add_message creates a message with the expected fields."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'

        # Act
        response = await service.add_message(
            conversation_id=conversation_id,
            content=sample_content,
            role='user',
        )

        # Assert
        assert isinstance(response, PendingMessageResponse)
        assert response.queued is True
        assert response.id is not None

        # Verify the message was stored correctly
        messages = await service.get_pending_messages(conversation_id)
        assert len(messages) == 1
        assert messages[0].conversation_id == conversation_id
        assert len(messages[0].content) == 1
        assert isinstance(messages[0].content[0], TextContent)
        assert messages[0].content[0].text == sample_content[0].text
        assert messages[0].role == 'user'
        assert messages[0].created_at is not None

    @pytest.mark.asyncio
    async def test_add_message_returns_correct_queue_position(
        self,
        service: SQLPendingMessageService,
        sample_content: list[TextContent],
    ):
        """Test that queue position increments correctly for each message."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'

        # Act - Add three messages
        response1 = await service.add_message(conversation_id, sample_content)
        response2 = await service.add_message(conversation_id, sample_content)
        response3 = await service.add_message(conversation_id, sample_content)

        # Assert
        assert response1.position == 1
        assert response2.position == 2
        assert response3.position == 3

    @pytest.mark.asyncio
    async def test_get_pending_messages_returns_messages_ordered_by_created_at(
        self,
        service: SQLPendingMessageService,
    ):
        """Test that messages are returned in chronological order."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        contents = [
            [TextContent(text='First message')],
            [TextContent(text='Second message')],
            [TextContent(text='Third message')],
        ]

        for content in contents:
            await service.add_message(conversation_id, content)

        # Act
        messages = await service.get_pending_messages(conversation_id)

        # Assert
        assert len(messages) == 3
        assert messages[0].content[0].text == 'First message'
        assert messages[1].content[0].text == 'Second message'
        assert messages[2].content[0].text == 'Third message'

    @pytest.mark.asyncio
    async def test_get_pending_messages_returns_empty_list_when_none_exist(
        self,
        service: SQLPendingMessageService,
    ):
        """Test that an empty list is returned for a conversation with no messages."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'

        # Act
        messages = await service.get_pending_messages(conversation_id)

        # Assert
        assert messages == []

    @pytest.mark.asyncio
    async def test_count_pending_messages_returns_correct_count(
        self,
        service: SQLPendingMessageService,
        sample_content: list[TextContent],
    ):
        """Test that count_pending_messages returns the correct number."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        other_conversation_id = f'task-{uuid4().hex}'

        # Add 3 messages to first conversation
        for _ in range(3):
            await service.add_message(conversation_id, sample_content)

        # Add 2 messages to second conversation
        for _ in range(2):
            await service.add_message(other_conversation_id, sample_content)

        # Act
        count1 = await service.count_pending_messages(conversation_id)
        count2 = await service.count_pending_messages(other_conversation_id)
        count_empty = await service.count_pending_messages('nonexistent')

        # Assert
        assert count1 == 3
        assert count2 == 2
        assert count_empty == 0

    @pytest.mark.asyncio
    async def test_delete_messages_for_conversation_removes_all_messages(
        self,
        service: SQLPendingMessageService,
        sample_content: list[TextContent],
    ):
        """Test that delete_messages_for_conversation removes all messages and returns count."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        other_conversation_id = f'task-{uuid4().hex}'

        # Add messages to both conversations
        for _ in range(3):
            await service.add_message(conversation_id, sample_content)
        await service.add_message(other_conversation_id, sample_content)

        # Act
        deleted_count = await service.delete_messages_for_conversation(conversation_id)

        # Assert
        assert deleted_count == 3
        assert await service.count_pending_messages(conversation_id) == 0
        # Other conversation should be unaffected
        assert await service.count_pending_messages(other_conversation_id) == 1

    @pytest.mark.asyncio
    async def test_delete_messages_for_conversation_returns_zero_when_none_exist(
        self,
        service: SQLPendingMessageService,
    ):
        """Test that deleting from nonexistent conversation returns 0."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'

        # Act
        deleted_count = await service.delete_messages_for_conversation(conversation_id)

        # Assert
        assert deleted_count == 0

    @pytest.mark.asyncio
    async def test_update_conversation_id_updates_all_matching_messages(
        self,
        service: SQLPendingMessageService,
        sample_content: list[TextContent],
    ):
        """Test that update_conversation_id updates all messages with the old ID."""
        # Arrange
        old_conversation_id = f'task-{uuid4().hex}'
        new_conversation_id = str(uuid4())
        unrelated_conversation_id = f'task-{uuid4().hex}'

        # Add messages to old conversation
        for _ in range(3):
            await service.add_message(old_conversation_id, sample_content)

        # Add message to unrelated conversation
        await service.add_message(unrelated_conversation_id, sample_content)

        # Act
        updated_count = await service.update_conversation_id(
            old_conversation_id, new_conversation_id
        )

        # Assert
        assert updated_count == 3

        # Verify old conversation has no messages
        assert await service.count_pending_messages(old_conversation_id) == 0

        # Verify new conversation has all messages
        messages = await service.get_pending_messages(new_conversation_id)
        assert len(messages) == 3
        for msg in messages:
            assert msg.conversation_id == new_conversation_id

        # Verify unrelated conversation is unchanged
        assert await service.count_pending_messages(unrelated_conversation_id) == 1

    @pytest.mark.asyncio
    async def test_update_conversation_id_returns_zero_when_no_match(
        self,
        service: SQLPendingMessageService,
    ):
        """Test that updating nonexistent conversation_id returns 0."""
        # Arrange
        old_conversation_id = f'task-{uuid4().hex}'
        new_conversation_id = str(uuid4())

        # Act
        updated_count = await service.update_conversation_id(
            old_conversation_id, new_conversation_id
        )

        # Assert
        assert updated_count == 0

    @pytest.mark.asyncio
    async def test_messages_are_isolated_between_conversations(
        self,
        service: SQLPendingMessageService,
    ):
        """Test that operations on one conversation don't affect others."""
        # Arrange
        conv1 = f'task-{uuid4().hex}'
        conv2 = f'task-{uuid4().hex}'

        await service.add_message(conv1, [TextContent(text='Conv1 msg')])
        await service.add_message(conv2, [TextContent(text='Conv2 msg')])

        # Act
        messages1 = await service.get_pending_messages(conv1)
        messages2 = await service.get_pending_messages(conv2)

        # Assert
        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0].content[0].text == 'Conv1 msg'
        assert messages2[0].content[0].text == 'Conv2 msg'
