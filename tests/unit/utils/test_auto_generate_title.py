"""Tests for the auto-generate title functionality."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from openhands.events.action import MessageAction
from openhands.events.event import EventSource
from openhands.events.event_store import EventStore
from openhands.llm.llm_registry import LLMRegistry
from openhands.storage.data_models.settings import Settings
from openhands.storage.memory import InMemoryFileStore
from openhands.utils.conversation_summary import auto_generate_title


@pytest.mark.asyncio
async def test_auto_generate_title_with_llm():
    """Test auto-generating a title using LLM."""
    # Mock dependencies
    file_store = InMemoryFileStore()
    llm_registry = MagicMock(spec=LLMRegistry)

    # Create test conversation with a user message
    conversation_id = 'test-conversation'
    user_id = 'test-user'

    # Create a mock event
    user_message = MessageAction(
        content='Help me write a Python script to analyze data'
    )
    user_message._source = EventSource.USER
    user_message._id = 1
    user_message._timestamp = datetime.now(timezone.utc).isoformat()

    # Mock the EventStore class
    with patch(
        'openhands.utils.conversation_summary.EventStore'
    ) as mock_event_store_cls:
        # Configure the mock event stream to return our test message
        mock_event_store = MagicMock(spec=EventStore)
        mock_event_store.search_events.return_value = [user_message]
        mock_event_store_cls.return_value = mock_event_store

        # Mock the LLM registry response
        llm_registry.request_extraneous_completion.return_value = (
            'Python Data Analysis Script'
        )

        # Create test settings with LLM config
        settings = Settings(
            llm_model='test-model',
            llm_api_key='test-key',
            llm_base_url='test-url',
        )

        # Call the auto_generate_title function directly
        title = await auto_generate_title(
            conversation_id, user_id, file_store, settings, llm_registry
        )

        # Verify the result
        assert title == 'Python Data Analysis Script'

        # Verify EventStore was created with the correct parameters
        mock_event_store_cls.assert_called_once_with(
            conversation_id, file_store, user_id
        )

        # Verify LLM registry was called with appropriate parameters
        llm_registry.request_extraneous_completion.assert_called_once()


@pytest.mark.asyncio
async def test_auto_generate_title_fallback():
    """Test auto-generating a title with fallback to truncation when LLM fails."""
    # Mock dependencies
    file_store = InMemoryFileStore()
    llm_registry = MagicMock(spec=LLMRegistry)

    # Create test conversation with a user message
    conversation_id = 'test-conversation'
    user_id = 'test-user'

    # Create a mock event with a long message
    long_message = 'This is a very long message that should be truncated when used as a title because it exceeds the maximum length allowed for titles'
    user_message = MessageAction(content=long_message)
    user_message._source = EventSource.USER
    user_message._id = 1
    user_message._timestamp = datetime.now(timezone.utc).isoformat()

    # Mock the EventStore class
    with patch(
        'openhands.utils.conversation_summary.EventStore'
    ) as mock_event_store_cls:
        # Configure the mock event stream to return our test message
        mock_event_store = MagicMock(spec=EventStore)
        mock_event_store.search_events.return_value = [user_message]
        mock_event_store_cls.return_value = mock_event_store

        # Mock the LLM registry to raise an exception
        llm_registry.request_extraneous_completion.side_effect = Exception('Test error')

        # Create test settings with LLM config
        settings = Settings(
            llm_model='test-model',
            llm_api_key='test-key',
            llm_base_url='test-url',
        )

        # Call the auto_generate_title function directly
        title = await auto_generate_title(
            conversation_id, user_id, file_store, settings, llm_registry
        )

        # Verify the result is a truncated version of the message
        assert title == 'This is a very long message th...'
        assert len(title) <= 35

        # Verify EventStore was created with the correct parameters
        mock_event_store_cls.assert_called_once_with(
            conversation_id, file_store, user_id
        )


@pytest.mark.asyncio
async def test_auto_generate_title_no_messages():
    """Test auto-generating a title when there are no user messages."""
    # Mock dependencies
    file_store = InMemoryFileStore()
    llm_registry = MagicMock(spec=LLMRegistry)

    # Create test conversation with no messages
    conversation_id = 'test-conversation'
    user_id = 'test-user'

    # Mock the EventStore class
    with patch(
        'openhands.utils.conversation_summary.EventStore'
    ) as mock_event_store_cls:
        # Configure the mock event store to return no events
        mock_event_store = MagicMock(spec=EventStore)
        mock_event_store.search_events.return_value = []
        mock_event_store_cls.return_value = mock_event_store

        # Create test settings
        settings = Settings(
            llm_model='test-model',
            llm_api_key='test-key',
            llm_base_url='test-url',
        )

        # Call the auto_generate_title function directly
        title = await auto_generate_title(
            conversation_id, user_id, file_store, settings, llm_registry
        )

        # Verify the result is empty
        assert title == ''

        # Verify EventStore was created with the correct parameters
        mock_event_store_cls.assert_called_once_with(
            conversation_id, file_store, user_id
        )
