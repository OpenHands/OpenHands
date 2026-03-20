"""Unit tests for the event_router endpoints.

This module tests the event router endpoints,
focusing on limit validation and error handling.
"""

from datetime import datetime
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.testclient import TestClient

from openhands.agent_server.models import EventSortOrder
from openhands.app_server.event.event_router import batch_get_events
from openhands.app_server.event_callback.event_callback_models import EventKind


def _make_mock_event_service(search_return=None, batch_get_return=None):
    """Create a mock EventService for testing."""
    service = MagicMock()
    service.search_events = AsyncMock(return_value=search_return)
    service.batch_get_events = AsyncMock(return_value=batch_get_return or [])
    return service


def _create_test_router():
    """Create a test router that mirrors the event router's search endpoint without auth dependencies.

    This allows us to test the Query parameter validation (gt=0, le=100) without
    needing to mock authentication.
    """
    test_router = APIRouter(prefix='/conversation/{conversation_id}/events')

    @test_router.get('/search')
    async def search_events(
        conversation_id: str,
        kind__eq: Annotated[
            EventKind | None,
            Query(title='Optional filter by event kind'),
        ] = None,
        timestamp__gte: Annotated[
            datetime | None,
            Query(title='Optional filter by timestamp greater than or equal to'),
        ] = None,
        timestamp__lt: Annotated[
            datetime | None,
            Query(title='Optional filter by timestamp less than'),
        ] = None,
        sort_order: Annotated[
            EventSortOrder,
            Query(title='Sort order for results'),
        ] = EventSortOrder.TIMESTAMP,
        page_id: Annotated[
            str | None,
            Query(title='Optional next_page_id from the previously returned page'),
        ] = None,
        limit: Annotated[
            int,
            Query(title='The max number of results in the page', gt=0, le=100),
        ] = 100,
    ) -> dict:
        """Test endpoint with same validation as the real search_events."""
        return {'limit': limit, 'conversation_id': conversation_id}

    return test_router


class TestSearchEventsValidation:
    """Test suite for search_events endpoint limit validation via FastAPI."""

    def test_returns_422_for_limit_exceeding_100(self):
        """Test that limit > 100 returns 422 Unprocessable Entity.

        FastAPI's Query validation (le=100) should reject limit=200.
        """
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(_create_test_router())

        client = TestClient(app, raise_server_exceptions=False)
        conversation_id = str(uuid4())

        response = client.get(
            f'/conversation/{conversation_id}/events/search',
            params={'limit': 200},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        # Verify the error message mentions the constraint
        error_detail = response.json()['detail']
        assert any(
            'less than or equal to 100' in str(err).lower() or 'le' in str(err).lower()
            for err in error_detail
        )

    def test_returns_422_for_limit_zero(self):
        """Test that limit=0 returns 422 Unprocessable Entity.

        FastAPI's Query validation (gt=0) should reject limit=0.
        """
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(_create_test_router())

        client = TestClient(app, raise_server_exceptions=False)
        conversation_id = str(uuid4())

        response = client.get(
            f'/conversation/{conversation_id}/events/search',
            params={'limit': 0},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_returns_422_for_negative_limit(self):
        """Test that negative limit returns 422 Unprocessable Entity.

        FastAPI's Query validation (gt=0) should reject limit=-1.
        """
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(_create_test_router())

        client = TestClient(app, raise_server_exceptions=False)
        conversation_id = str(uuid4())

        response = client.get(
            f'/conversation/{conversation_id}/events/search',
            params={'limit': -1},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_accepts_valid_limit_100(self):
        """Test that limit=100 is accepted (boundary case)."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(_create_test_router())

        client = TestClient(app, raise_server_exceptions=False)
        conversation_id = str(uuid4())

        response = client.get(
            f'/conversation/{conversation_id}/events/search',
            params={'limit': 100},
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()['limit'] == 100

    def test_accepts_valid_limit_1(self):
        """Test that limit=1 is accepted (boundary case)."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(_create_test_router())

        client = TestClient(app, raise_server_exceptions=False)
        conversation_id = str(uuid4())

        response = client.get(
            f'/conversation/{conversation_id}/events/search',
            params={'limit': 1},
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()['limit'] == 1


@pytest.mark.asyncio
class TestBatchGetEvents:
    """Test suite for batch_get_events endpoint."""

    async def test_returns_400_for_more_than_100_ids(self):
        """Test that requesting more than 100 IDs returns 400 Bad Request.

        Arrange: Create list with 101 IDs
        Act: Call batch_get_events
        Assert: HTTPException is raised with 400 status
        """
        # Arrange
        conversation_id = str(uuid4())
        ids = [str(uuid4()) for _ in range(101)]
        mock_service = _make_mock_event_service()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await batch_get_events(
                conversation_id=conversation_id,
                id=ids,
                event_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Cannot request more than 100 events' in exc_info.value.detail
        assert '101' in exc_info.value.detail

    async def test_accepts_exactly_100_ids(self):
        """Test that exactly 100 IDs is accepted.

        Arrange: Create list with 100 IDs
        Act: Call batch_get_events
        Assert: No exception is raised and service is called
        """
        # Arrange
        conversation_id = str(uuid4())
        ids = [str(uuid4()) for _ in range(100)]
        mock_return = [None] * 100
        mock_service = _make_mock_event_service(batch_get_return=mock_return)

        # Act
        result = await batch_get_events(
            conversation_id=conversation_id,
            id=ids,
            event_service=mock_service,
        )

        # Assert
        assert result == mock_return
        mock_service.batch_get_events.assert_called_once()

    async def test_accepts_empty_list(self):
        """Test that empty list of IDs is accepted.

        Arrange: Create empty list of IDs
        Act: Call batch_get_events
        Assert: No exception is raised
        """
        # Arrange
        conversation_id = str(uuid4())
        mock_service = _make_mock_event_service(batch_get_return=[])

        # Act
        result = await batch_get_events(
            conversation_id=conversation_id,
            id=[],
            event_service=mock_service,
        )

        # Assert
        assert result == []
        mock_service.batch_get_events.assert_called_once()
