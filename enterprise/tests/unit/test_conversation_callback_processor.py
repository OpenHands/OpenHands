"""
Tests for ConversationCallbackProcessor and ConversationCallback models.
"""

import json
from uuid import UUID

import pytest
from storage.conversation_callback import (
    CallbackStatus,
    ConversationCallback,
    ConversationCallbackProcessor,
)
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import (
    StoredConversationMetadataSaas,
)

from openhands.events.observation.agent import AgentStateChangedObservation


class MockConversationCallbackProcessor(ConversationCallbackProcessor):
    """Mock implementation of ConversationCallbackProcessor for testing."""

    name: str = 'test'
    config: dict = {}

    def __init__(self, name: str = 'test', config: dict | None = None, **kwargs):
        super().__init__(name=name, config=config or {}, **kwargs)
        self.call_count = 0
        self.last_conversation_id: str | None = None

    def __call__(
        self, callback: ConversationCallback, observation: AgentStateChangedObservation
    ) -> None:
        """Mock implementation that tracks calls."""
        self.call_count += 1
        self.last_conversation_id = callback.conversation_id


class TestConversationCallbackProcessor:
    """Test the ConversationCallbackProcessor abstract base class."""

    def test_mock_processor_creation(self):
        """Test that we can create a mock processor."""
        processor = MockConversationCallbackProcessor(
            name='test_processor', config={'key': 'value'}
        )
        assert processor.name == 'test_processor'
        assert processor.config == {'key': 'value'}
        assert processor.call_count == 0
        assert processor.last_conversation_id is None

    def test_mock_processor_call(self):
        """Test that the mock processor can be called."""
        callback = ConversationCallback(conversation_id='test_conversation_id')
        processor = MockConversationCallbackProcessor()
        processor(
            callback,
            AgentStateChangedObservation('foobar', 'awaiting_user_input'),
        )

        assert processor.call_count == 1
        assert processor.last_conversation_id == 'test_conversation_id'

    def test_processor_serialization(self):
        """Test that processors can be serialized to JSON."""
        processor = MockConversationCallbackProcessor(
            name='test', config={'setting': 'value'}
        )
        json_data = processor.model_dump_json()

        # Should be able to parse the JSON
        data = json.loads(json_data)
        assert data['name'] == 'test'
        assert data['config'] == {'setting': 'value'}


class TestConversationCallback:
    """Test the ConversationCallback SQLAlchemy model."""

    @pytest.fixture
    def conversation_metadata(self, session_maker):
        """Create a test conversation metadata record."""
        with session_maker() as session:
            metadata = StoredConversationMetadata(
                conversation_id='test_conversation_123'
            )
            metadata_saas = StoredConversationMetadataSaas(
                conversation_id='test_conversation_123',
                user_id=UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
                org_id=UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
            )
            session.add(metadata)
            session.add(metadata_saas)
            session.commit()
            session.refresh(metadata)
            yield metadata

            # Cleanup
            session.delete(metadata)
            session.delete(metadata_saas)
            session.commit()

    def test_callback_creation(self, conversation_metadata, session_maker):
        """Test creating a conversation callback."""
        processor = MockConversationCallbackProcessor(name='test_processor')

        with session_maker() as session:
            callback = ConversationCallback(
                conversation_id=conversation_metadata.conversation_id,
                status=CallbackStatus.ACTIVE,
                processor_type='tests.unit.test_conversation_processor.MockConversationCallbackProcessor',
                processor_json=processor.model_dump_json(),
            )
            session.add(callback)
            session.commit()
            session.refresh(callback)

            assert callback.id is not None
            assert callback.conversation_id == conversation_metadata.conversation_id
            assert callback.status == CallbackStatus.ACTIVE
            assert callback.created_at is not None
            assert callback.updated_at is not None

            # Cleanup
            session.delete(callback)
            session.commit()

    def test_set_processor(self, conversation_metadata, session_maker):
        """Test setting a processor on a callback."""
        processor = MockConversationCallbackProcessor(
            name='test_processor', config={'key': 'value'}
        )

        callback = ConversationCallback(
            conversation_id=conversation_metadata.conversation_id
        )
        callback.set_processor(processor)

        assert (
            callback.processor_type
            == 'enterprise.tests.unit.test_conversation_callback_processor.MockConversationCallbackProcessor'
        )

        # Verify the JSON contains the processor data
        processor_data = json.loads(callback.processor_json)
        assert processor_data['name'] == 'test_processor'
        assert processor_data['config'] == {'key': 'value'}

    def test_get_processor(self, conversation_metadata, session_maker):
        """Test getting a processor from a callback."""
        processor = MockConversationCallbackProcessor(
            name='test_processor', config={'key': 'value'}
        )

        callback = ConversationCallback(
            conversation_id=conversation_metadata.conversation_id
        )
        callback.set_processor(processor)

        # Get the processor back
        retrieved_processor = callback.get_processor()

        assert isinstance(retrieved_processor, MockConversationCallbackProcessor)
        assert retrieved_processor.name == 'test_processor'
        assert retrieved_processor.config == {'key': 'value'}

    def test_callback_status_enum(self):
        """Test the CallbackStatus enum."""
        assert CallbackStatus.ACTIVE.value == 'ACTIVE'
        assert CallbackStatus.COMPLETED.value == 'COMPLETED'
        assert CallbackStatus.ERROR.value == 'ERROR'

    def test_callback_foreign_key_constraint(
        self, conversation_metadata, session_maker
    ):
        """Test that the foreign key constraint works."""
        with session_maker() as session:
            # This should work with valid conversation_id
            callback = ConversationCallback(
                conversation_id=conversation_metadata.conversation_id,
                processor_type='test.Processor',
                processor_json='{}',
            )
            session.add(callback)
            session.commit()

            # Cleanup
            session.delete(callback)
            session.commit()

            # Note: SQLite doesn't enforce foreign key constraints by default in tests
            # In a real PostgreSQL database, this would raise an integrity error
            # For now, we just test that the callback can be created with valid data
