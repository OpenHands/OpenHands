"""Tests for SharedEventService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from server.sharing.filesystem_shared_event_service import (
    SharedEventServiceImpl,
)
from server.sharing.shared_conversation_info_service import (
    SharedConversationInfoService,
)
from server.sharing.shared_conversation_models import SharedConversation

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event.event_service import EventService
from openhands.sdk.llm import MetricsSnapshot
from openhands.sdk.llm.utils.metrics import TokenUsage


@pytest.fixture
def mock_shared_conversation_info_service():
    """Create a mock SharedConversationInfoService."""
    return AsyncMock(spec=SharedConversationInfoService)


@pytest.fixture
def mock_event_service():
    """Create a mock EventService."""
    return AsyncMock(spec=EventService)


@pytest.fixture
def shared_event_service(mock_shared_conversation_info_service, mock_event_service):
    """Create a SharedEventService for testing."""
    return SharedEventServiceImpl(
        shared_conversation_info_service=mock_shared_conversation_info_service,
        event_service=mock_event_service,
    )


@pytest.fixture
def sample_public_conversation():
    """Create a sample public conversation."""
    return SharedConversation(
        id=uuid4(),
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        title='Test Public Conversation',
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metrics=MetricsSnapshot(
            accumulated_cost=0.0,
            max_budget_per_task=10.0,
            accumulated_token_usage=TokenUsage(),
        ),
    )


@pytest.fixture
def sample_event():
    """Create a sample event."""
    # For testing purposes, we'll just use a mock that the EventPage can accept
    # The actual event creation is complex and not the focus of these tests
    return None


class TestSharedEventService:
    """Test cases for SharedEventService."""

    async def test_get_shared_event_returns_event_for_public_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
        sample_event,
    ):
        """Test that get_shared_event returns an event for a public conversation."""
        conversation_id = sample_public_conversation.id
        event_id = 'test_event_id'

        # Mock the public conversation service to return a public conversation
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = sample_public_conversation

        # Mock the event service to return an event
        mock_event_service.get_event.return_value = sample_event

        # Call the method
        result = await shared_event_service.get_shared_event(conversation_id, event_id)

        # Verify the result
        assert result == sample_event
        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )
        mock_event_service.get_event.assert_called_once_with(event_id)

    async def test_get_shared_event_returns_none_for_private_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
    ):
        """Test that get_shared_event returns None for a private conversation."""
        conversation_id = uuid4()
        event_id = 'test_event_id'

        # Mock the public conversation service to return None (private conversation)
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = None

        # Call the method
        result = await shared_event_service.get_shared_event(conversation_id, event_id)

        # Verify the result
        assert result is None
        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )
        # Event service should not be called
        mock_event_service.get_event.assert_not_called()

    async def test_search_shared_events_returns_events_for_public_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
        sample_event,
    ):
        """Test that search_shared_events returns events for a public conversation."""
        conversation_id = sample_public_conversation.id

        # Mock the public conversation service to return a public conversation
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = sample_public_conversation

        # Mock the event service to return events
        mock_event_page = EventPage(items=[], next_page_id=None)
        mock_event_service.search_events.return_value = mock_event_page

        # Call the method
        result = await shared_event_service.search_shared_events(
            conversation_id=conversation_id,
            kind__eq='ActionEvent',
            limit=10,
        )

        # Verify the result
        assert result == mock_event_page
        assert len(result.items) == 0  # Empty list as we mocked

        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )
        mock_event_service.search_events.assert_called_once_with(
            conversation_id__eq=conversation_id,
            kind__eq='ActionEvent',
            timestamp__gte=None,
            timestamp__lt=None,
            sort_order=EventSortOrder.TIMESTAMP,
            page_id=None,
            limit=10,
        )

    async def test_search_shared_events_returns_empty_for_private_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
    ):
        """Test that search_shared_events returns empty page for a private conversation."""
        conversation_id = uuid4()

        # Mock the public conversation service to return None (private conversation)
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = None

        # Call the method
        result = await shared_event_service.search_shared_events(
            conversation_id=conversation_id,
            limit=10,
        )

        # Verify the result
        assert isinstance(result, EventPage)
        assert len(result.items) == 0
        assert result.next_page_id is None

        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )
        # Event service should not be called
        mock_event_service.search_events.assert_not_called()

    async def test_count_shared_events_returns_count_for_public_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
    ):
        """Test that count_shared_events returns count for a public conversation."""
        conversation_id = sample_public_conversation.id

        # Mock the public conversation service to return a public conversation
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = sample_public_conversation

        # Mock the event service to return a count
        mock_event_service.count_events.return_value = 5

        # Call the method
        result = await shared_event_service.count_shared_events(
            conversation_id=conversation_id,
            kind__eq='ActionEvent',
        )

        # Verify the result
        assert result == 5

        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )
        mock_event_service.count_events.assert_called_once_with(
            conversation_id__eq=conversation_id,
            kind__eq='ActionEvent',
            timestamp__gte=None,
            timestamp__lt=None,
            sort_order=EventSortOrder.TIMESTAMP,
        )

    async def test_count_shared_events_returns_zero_for_private_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
    ):
        """Test that count_shared_events returns 0 for a private conversation."""
        conversation_id = uuid4()

        # Mock the public conversation service to return None (private conversation)
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = None

        # Call the method
        result = await shared_event_service.count_shared_events(
            conversation_id=conversation_id,
        )

        # Verify the result
        assert result == 0

        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )
        # Event service should not be called
        mock_event_service.count_events.assert_not_called()

    async def test_batch_get_shared_events_returns_events_for_public_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
        sample_event,
    ):
        """Test that batch_get_shared_events returns events for a public conversation."""
        conversation_id = sample_public_conversation.id
        event_ids = ['event1', 'event2']

        # Mock the public conversation service to return a public conversation
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = sample_public_conversation

        # Mock the event service to return events
        mock_event_service.get_event.side_effect = [sample_event, None]

        # Call the method
        result = await shared_event_service.batch_get_shared_events(
            conversation_id, event_ids
        )

        # Verify the result
        assert len(result) == 2
        assert result[0] == sample_event
        assert result[1] is None

        # Verify that get_shared_conversation_info was called for each event
        assert (
            mock_shared_conversation_info_service.get_shared_conversation_info.call_count
            == 2
        )
        # Verify that get_event was called for each event
        assert mock_event_service.get_event.call_count == 2

    async def test_batch_get_shared_events_returns_none_for_private_conversation(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
    ):
        """Test that batch_get_shared_events returns None for a private conversation."""
        conversation_id = uuid4()
        event_ids = ['event1', 'event2']

        # Mock the public conversation service to return None (private conversation)
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = None

        # Call the method
        result = await shared_event_service.batch_get_shared_events(
            conversation_id, event_ids
        )

        # Verify the result
        assert len(result) == 2
        assert result[0] is None
        assert result[1] is None

        # Verify that get_shared_conversation_info was called for each event
        assert (
            mock_shared_conversation_info_service.get_shared_conversation_info.call_count
            == 2
        )
        # Event service should not be called
        mock_event_service.get_event.assert_not_called()

    async def test_search_shared_events_with_all_parameters(
        self,
        shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
    ):
        """Test search_shared_events with all parameters."""
        conversation_id = sample_public_conversation.id
        timestamp_gte = datetime(2023, 1, 1, tzinfo=UTC)
        timestamp_lt = datetime(2023, 12, 31, tzinfo=UTC)

        # Mock the public conversation service to return a public conversation
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = sample_public_conversation

        # Mock the event service to return events
        mock_event_page = EventPage(items=[], next_page_id='next_page')
        mock_event_service.search_events.return_value = mock_event_page

        # Call the method with all parameters
        result = await shared_event_service.search_shared_events(
            conversation_id=conversation_id,
            kind__eq='ObservationEvent',
            timestamp__gte=timestamp_gte,
            timestamp__lt=timestamp_lt,
            sort_order=EventSortOrder.TIMESTAMP_DESC,
            page_id='current_page',
            limit=50,
        )

        # Verify the result
        assert result == mock_event_page

        mock_event_service.search_events.assert_called_once_with(
            conversation_id__eq=conversation_id,
            kind__eq='ObservationEvent',
            timestamp__gte=timestamp_gte,
            timestamp__lt=timestamp_lt,
            sort_order=EventSortOrder.TIMESTAMP_DESC,
            page_id='current_page',
            limit=50,
        )
