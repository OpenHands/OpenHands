"""Tests for public event router."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event_callback.event_callback_models import EventKind
from openhands.app_server.sharing.public_event_router import router
from openhands.app_server.sharing.public_event_service import PublicEventService
from openhands.sdk import Event


@pytest.fixture
def mock_public_event_service():
    """Create a mock PublicEventService."""
    return AsyncMock(spec=PublicEventService)


@pytest.fixture
def app(mock_public_event_service):
    """Create a FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)

    # Override the dependency
    app.dependency_overrides[router.public_event_service_dependency] = (
        lambda: mock_public_event_service
    )

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_event():
    """Create a sample event."""
    event = MagicMock(spec=Event)
    event.id = 'test_event_id'
    event.timestamp = datetime.now(UTC)
    # Make it JSON serializable
    event.model_dump.return_value = {
        'id': 'test_event_id',
        'timestamp': datetime.now(UTC).isoformat(),
        'type': 'action',
    }
    return event


class TestPublicEventRouter:
    """Test cases for public event router."""

    def test_search_public_events(
        self, client, mock_public_event_service, sample_event
    ):
        """Test searching public events."""
        conversation_id = uuid4()

        # Mock the service response
        mock_page = EventPage(items=[sample_event], next_page_id=None)
        mock_public_event_service.search_public_events.return_value = mock_page

        # Make the request
        response = client.get(
            '/public-events/search', params={'conversation_id': str(conversation_id)}
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert 'items' in data
        assert 'next_page_id' in data
        assert len(data['items']) == 1

        # Verify the service was called correctly
        mock_public_event_service.search_public_events.assert_called_once_with(
            conversation_id=conversation_id,
            kind__eq=None,
            timestamp__gte=None,
            timestamp__lt=None,
            sort_order=EventSortOrder.TIMESTAMP,
            page_id=None,
            limit=100,
        )

    def test_search_public_events_with_filters(self, client, mock_public_event_service):
        """Test searching public events with filters."""
        conversation_id = uuid4()

        # Mock the service response
        mock_page = EventPage(items=[], next_page_id=None)
        mock_public_event_service.search_public_events.return_value = mock_page

        # Make the request with filters
        response = client.get(
            '/public-events/search',
            params={
                'conversation_id': str(conversation_id),
                'kind__eq': 'ACTION',
                'sort_order': 'TIMESTAMP_DESC',
                'limit': 50,
                'page_id': 'test_page',
            },
        )

        # Verify the response
        assert response.status_code == 200

        # Verify the service was called with correct parameters
        mock_public_event_service.search_public_events.assert_called_once_with(
            conversation_id=conversation_id,
            kind__eq=EventKind.ACTION,
            timestamp__gte=None,
            timestamp__lt=None,
            sort_order=EventSortOrder.TIMESTAMP_DESC,
            page_id='test_page',
            limit=50,
        )

    def test_search_public_events_missing_conversation_id(self, client):
        """Test searching without conversation_id."""
        # Make the request without conversation_id
        response = client.get('/public-events/search')

        # Should fail validation
        assert response.status_code == 422

    def test_search_public_events_with_invalid_limit(self, client):
        """Test searching with invalid limit."""
        conversation_id = uuid4()

        # Test limit too high
        response = client.get(
            '/public-events/search',
            params={'conversation_id': str(conversation_id), 'limit': 101},
        )
        assert response.status_code == 422

        # Test limit too low
        response = client.get(
            '/public-events/search',
            params={'conversation_id': str(conversation_id), 'limit': 0},
        )
        assert response.status_code == 422

    def test_count_public_events(self, client, mock_public_event_service):
        """Test counting public events."""
        conversation_id = uuid4()

        # Mock the service response
        mock_public_event_service.count_public_events.return_value = 5

        # Make the request
        response = client.get(
            '/public-events/count', params={'conversation_id': str(conversation_id)}
        )

        # Verify the response
        assert response.status_code == 200
        assert response.json() == 5

        # Verify the service was called correctly
        mock_public_event_service.count_public_events.assert_called_once_with(
            conversation_id=conversation_id,
            kind__eq=None,
            timestamp__gte=None,
            timestamp__lt=None,
            sort_order=EventSortOrder.TIMESTAMP,
        )

    def test_count_public_events_with_filters(self, client, mock_public_event_service):
        """Test counting public events with filters."""
        conversation_id = uuid4()

        # Mock the service response
        mock_public_event_service.count_public_events.return_value = 2

        # Make the request with filters
        response = client.get(
            '/public-events/count',
            params={
                'conversation_id': str(conversation_id),
                'kind__eq': 'OBSERVATION',
            },
        )

        # Verify the response
        assert response.status_code == 200
        assert response.json() == 2

        # Verify the service was called with correct parameters
        mock_public_event_service.count_public_events.assert_called_once_with(
            conversation_id=conversation_id,
            kind__eq=EventKind.OBSERVATION,
            timestamp__gte=None,
            timestamp__lt=None,
            sort_order=EventSortOrder.TIMESTAMP,
        )

    def test_count_public_events_missing_conversation_id(self, client):
        """Test counting without conversation_id."""
        # Make the request without conversation_id
        response = client.get('/public-events/count')

        # Should fail validation
        assert response.status_code == 422

    def test_batch_get_public_events(
        self, client, mock_public_event_service, sample_event
    ):
        """Test batch getting public events."""
        conversation_id = uuid4()
        event_ids = ['event1', 'event2']

        # Mock the service response
        mock_public_event_service.batch_get_public_events.return_value = [
            sample_event,
            None,
        ]

        # Make the request
        response = client.get(
            '/public-events',
            params={'conversation_id': str(conversation_id), 'id': event_ids},
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[1] is None

        # Verify the service was called correctly
        mock_public_event_service.batch_get_public_events.assert_called_once_with(
            conversation_id, event_ids
        )

    def test_batch_get_public_events_too_many_ids(self, client):
        """Test batch getting with too many IDs."""
        conversation_id = uuid4()
        # Create 101 event IDs
        event_ids = [f'event_{i}' for i in range(101)]

        # Make the request
        response = client.get(
            '/public-events',
            params={'conversation_id': str(conversation_id), 'id': event_ids},
        )

        # Should fail validation
        assert response.status_code == 500  # Internal server error due to assertion

    def test_batch_get_public_events_missing_conversation_id(self, client):
        """Test batch getting without conversation_id."""
        # Make the request without conversation_id
        response = client.get('/public-events', params={'id': ['event1']})

        # Should fail validation
        assert response.status_code == 422

    def test_get_public_event(self, client, mock_public_event_service, sample_event):
        """Test getting a single public event."""
        conversation_id = uuid4()
        event_id = 'test_event_id'

        # Mock the service response
        mock_public_event_service.get_public_event.return_value = sample_event

        # Make the request
        response = client.get(f'/public-events/{conversation_id}/{event_id}')

        # Verify the response
        assert response.status_code == 200
        # The response should contain the event data
        data = response.json()
        assert data is not None

        # Verify the service was called correctly
        mock_public_event_service.get_public_event.assert_called_once_with(
            conversation_id, event_id
        )

    def test_get_public_event_not_found(self, client, mock_public_event_service):
        """Test getting a non-existent event or event from private conversation."""
        conversation_id = uuid4()
        event_id = 'nonexistent_event'

        # Mock the service response
        mock_public_event_service.get_public_event.return_value = None

        # Make the request
        response = client.get(f'/public-events/{conversation_id}/{event_id}')

        # Verify the response
        assert response.status_code == 200
        assert response.json() is None

        # Verify the service was called correctly
        mock_public_event_service.get_public_event.assert_called_once_with(
            conversation_id, event_id
        )

    def test_get_public_event_invalid_conversation_uuid(self, client):
        """Test getting an event with invalid conversation UUID."""
        event_id = 'test_event'

        # Make the request with invalid UUID
        response = client.get(f'/public-events/invalid-uuid/{event_id}')

        # Should fail validation
        assert response.status_code == 422

    def test_search_public_events_with_timestamps(
        self, client, mock_public_event_service
    ):
        """Test searching public events with timestamp filters."""
        conversation_id = uuid4()

        # Mock the service response
        mock_page = EventPage(items=[], next_page_id=None)
        mock_public_event_service.search_public_events.return_value = mock_page

        # Make the request with timestamp filters
        timestamp_gte = '2023-01-01T00:00:00Z'
        timestamp_lt = '2023-12-31T23:59:59Z'

        response = client.get(
            '/public-events/search',
            params={
                'conversation_id': str(conversation_id),
                'timestamp__gte': timestamp_gte,
                'timestamp__lt': timestamp_lt,
            },
        )

        # Verify the response
        assert response.status_code == 200

        # Verify the service was called with correct parameters
        mock_public_event_service.search_public_events.assert_called_once()
        call_args = mock_public_event_service.search_public_events.call_args
        assert call_args.kwargs['conversation_id'] == conversation_id
        assert call_args.kwargs['timestamp__gte'] is not None
        assert call_args.kwargs['timestamp__lt'] is not None
