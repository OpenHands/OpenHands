"""Tests for public conversation router."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openhands.app_server.sharing.public_conversation_info_service import (
    PublicConversationInfoService,
)
from openhands.app_server.sharing.public_conversation_models import (
    PublicConversation,
    PublicConversationPage,
    PublicConversationSortOrder,
)
from openhands.app_server.sharing.public_conversation_router import (
    public_conversation_service_dependency,
    router,
)
from openhands.sdk.llm import MetricsSnapshot
from openhands.sdk.llm.utils.metrics import TokenUsage


@pytest.fixture
def mock_public_conversation_service():
    """Create a mock PublicConversationInfoService."""
    return AsyncMock(spec=PublicConversationInfoService)


@pytest.fixture
def app(mock_public_conversation_service):
    """Create a FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)

    # Override the dependency
    app.dependency_overrides[public_conversation_service_dependency] = (
        lambda: mock_public_conversation_service
    )

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_public_conversation():
    """Create a sample public conversation."""
    return PublicConversation(
        id=uuid4(),
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        title='Test Public Conversation',
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metrics=MetricsSnapshot(
            accumulated_cost=1.5,
            max_budget_per_task=10.0,
            accumulated_token_usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
            ),
        ),
    )


class TestPublicConversationRouter:
    """Test cases for public conversation router."""

    def test_search_public_conversations(
        self, client, mock_public_conversation_service, sample_public_conversation
    ):
        """Test searching public conversations."""
        # Mock the service response
        mock_page = PublicConversationPage(
            items=[sample_public_conversation], next_page_id=None
        )
        mock_public_conversation_service.search_public_conversation_info.return_value = mock_page

        # Make the request
        response = client.get('/public-conversations/search')

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert 'items' in data
        assert 'next_page_id' in data
        assert len(data['items']) == 1
        assert data['items'][0]['title'] == 'Test Public Conversation'

        # Verify the service was called correctly
        mock_public_conversation_service.search_public_conversation_info.assert_called_once_with(
            title__contains=None,
            created_at__gte=None,
            created_at__lt=None,
            updated_at__gte=None,
            updated_at__lt=None,
            sort_order=PublicConversationSortOrder.CREATED_AT_DESC,
            page_id=None,
            limit=100,
            include_sub_conversations=False,
        )

    def test_search_public_conversations_with_filters(
        self, client, mock_public_conversation_service
    ):
        """Test searching public conversations with filters."""
        # Mock the service response
        mock_page = PublicConversationPage(items=[], next_page_id=None)
        mock_public_conversation_service.search_public_conversation_info.return_value = mock_page

        # Make the request with filters
        response = client.get(
            '/public-conversations/search',
            params={
                'title__contains': 'test',
                'sort_order': 'TITLE',
                'limit': 50,
                'include_sub_conversations': True,
            },
        )

        # Verify the response
        assert response.status_code == 200

        # Verify the service was called with correct parameters
        mock_public_conversation_service.search_public_conversation_info.assert_called_once_with(
            title__contains='test',
            created_at__gte=None,
            created_at__lt=None,
            updated_at__gte=None,
            updated_at__lt=None,
            sort_order=PublicConversationSortOrder.TITLE,
            page_id=None,
            limit=50,
            include_sub_conversations=True,
        )

    def test_search_public_conversations_with_invalid_limit(self, client):
        """Test searching with invalid limit."""
        # Test limit too high
        response = client.get('/public-conversations/search', params={'limit': 101})
        assert response.status_code == 422

        # Test limit too low
        response = client.get('/public-conversations/search', params={'limit': 0})
        assert response.status_code == 422

    def test_count_public_conversations(self, client, mock_public_conversation_service):
        """Test counting public conversations."""
        # Mock the service response
        mock_public_conversation_service.count_public_conversation_info.return_value = 5

        # Make the request
        response = client.get('/public-conversations/count')

        # Verify the response
        assert response.status_code == 200
        assert response.json() == 5

        # Verify the service was called correctly
        mock_public_conversation_service.count_public_conversation_info.assert_called_once_with(
            title__contains=None,
            created_at__gte=None,
            created_at__lt=None,
            updated_at__gte=None,
            updated_at__lt=None,
        )

    def test_count_public_conversations_with_filters(
        self, client, mock_public_conversation_service
    ):
        """Test counting public conversations with filters."""
        # Mock the service response
        mock_public_conversation_service.count_public_conversation_info.return_value = 2

        # Make the request with filters
        response = client.get(
            '/public-conversations/count', params={'title__contains': 'test'}
        )

        # Verify the response
        assert response.status_code == 200
        assert response.json() == 2

        # Verify the service was called with correct parameters
        mock_public_conversation_service.count_public_conversation_info.assert_called_once_with(
            title__contains='test',
            created_at__gte=None,
            created_at__lt=None,
            updated_at__gte=None,
            updated_at__lt=None,
        )

    def test_batch_get_public_conversations(
        self, client, mock_public_conversation_service, sample_public_conversation
    ):
        """Test batch getting public conversations."""
        conversation_id = sample_public_conversation.id

        # Mock the service response
        mock_public_conversation_service.batch_get_public_conversation_info.return_value = [
            sample_public_conversation,
            None,
        ]

        # Make the request
        response = client.get(
            '/public-conversations',
            params={'ids': [str(conversation_id), str(uuid4())]},
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]['title'] == 'Test Public Conversation'
        assert data[1] is None

        # Verify the service was called correctly
        mock_public_conversation_service.batch_get_public_conversation_info.assert_called_once()

    def test_batch_get_public_conversations_too_many_ids(self, client):
        """Test batch getting with too many IDs."""
        # Create 101 UUIDs
        ids = [str(uuid4()) for _ in range(101)]

        # Make the request
        response = client.get('/public-conversations', params={'ids': ids})

        # Should fail validation
        assert response.status_code == 500  # Internal server error due to assertion

    def test_get_public_conversation(
        self, client, mock_public_conversation_service, sample_public_conversation
    ):
        """Test getting a single public conversation."""
        conversation_id = sample_public_conversation.id

        # Mock the service response
        mock_public_conversation_service.get_public_conversation_info.return_value = (
            sample_public_conversation
        )

        # Make the request
        response = client.get(f'/public-conversations/{conversation_id}')

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data['title'] == 'Test Public Conversation'
        assert data['id'] == str(conversation_id)

        # Verify the service was called correctly
        mock_public_conversation_service.get_public_conversation_info.assert_called_once_with(
            conversation_id
        )

    def test_get_public_conversation_not_found(
        self, client, mock_public_conversation_service
    ):
        """Test getting a non-existent or private conversation."""
        conversation_id = uuid4()

        # Mock the service response
        mock_public_conversation_service.get_public_conversation_info.return_value = (
            None
        )

        # Make the request
        response = client.get(f'/public-conversations/{conversation_id}')

        # Verify the response
        assert response.status_code == 200
        assert response.json() is None

        # Verify the service was called correctly
        mock_public_conversation_service.get_public_conversation_info.assert_called_once_with(
            conversation_id
        )

    def test_get_public_conversation_invalid_uuid(self, client):
        """Test getting a conversation with invalid UUID."""
        # Make the request with invalid UUID
        response = client.get('/public-conversations/invalid-uuid')

        # Should fail validation
        assert response.status_code == 422
