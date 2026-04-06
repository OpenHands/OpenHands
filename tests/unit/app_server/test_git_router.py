"""Unit tests for the git_router endpoints.

This module tests the git router endpoints,
focusing on pagination and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from openhands.app_server.git.git_router import (
    get_user_installations,
    get_user_repositories,
    router,
)
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.dependencies import check_session_api_key
from openhands.app_server.utils.paging_utils import encode_page_id, paginate_results
from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import ProviderType, Repository


class TestPagination:
    """Test suite for pagination helper function."""

    def test_returns_first_page_when_no_page_id(self):
        """Test that first page is returned when no page_id is provided."""
        items = ['a', 'b', 'c', 'd', 'e']

        result, next_page_id = paginate_results(items, None, 2)

        assert result == ['a', 'b']
        # next_page_id is base64-encoded
        assert next_page_id == encode_page_id(2)

    def test_returns_second_page_when_page_id_provided(self):
        """Test that correct page is returned when page_id is provided."""
        items = ['a', 'b', 'c', 'd', 'e']
        # Use base64-encoded page_id
        encoded_page_id = encode_page_id(2)

        result, next_page_id = paginate_results(items, encoded_page_id, 2)

        assert result == ['c', 'd']
        assert next_page_id == encode_page_id(4)

    def test_returns_empty_when_page_id_exceeds_length(self):
        """Test that empty list is returned when page_id exceeds length."""
        items = ['a', 'b', 'c']
        # Use base64-encoded page_id
        encoded_page_id = encode_page_id(10)

        result, next_page_id = paginate_results(items, encoded_page_id, 2)

        assert result == []
        assert next_page_id is None

    def test_returns_none_next_page_when_last_page(self):
        """Test that next_page_id is None on last page."""
        items = ['a', 'b', 'c']
        # Use base64-encoded page_id
        encoded_page_id = encode_page_id(2)

        result, next_page_id = paginate_results(items, encoded_page_id, 2)

        assert result == ['c']
        assert next_page_id is None

    def test_respects_limit(self):
        """Test that limit is respected."""
        items = ['a', 'b', 'c', 'd', 'e']

        result, next_page_id = paginate_results(items, None, 5)

        assert result == items
        assert next_page_id is None


def _make_mock_user_context(
    provider_tokens: dict | None = None,
    user_id: str = 'test-user-id',
):
    """Create a mock UserContext for testing."""
    context = MagicMock(spec=UserContext)
    context.get_provider_tokens = AsyncMock(return_value=provider_tokens)
    context.get_user_id = AsyncMock(return_value=user_id)
    return context


def _make_mock_provider_handler():
    """Create a mock ProviderHandler."""
    handler = MagicMock()
    handler.get_github_installations = AsyncMock(
        return_value=['inst-1', 'inst-2', 'inst-3', 'inst-4', 'inst-5']
    )
    handler.get_bitbucket_workspaces = AsyncMock(return_value=['ws-1', 'ws-2'])
    handler.get_repositories = AsyncMock(return_value=[])
    return handler


@pytest.fixture
def test_client():
    """Create a test client with the actual git router and mocked dependencies.

    We override check_session_api_key to bypass auth checks.
    This allows us to test the actual Query parameter validation in the router.
    """
    app = FastAPI()
    app.include_router(router)

    # Override the auth dependency to always pass
    app.dependency_overrides[check_session_api_key] = lambda: None

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    # Clean up
    app.dependency_overrides.clear()


class TestInstallationsEndpoint:
    """Test suite for /installations endpoint."""

    def test_returns_401_when_no_provider_tokens(self, test_client, monkeypatch):
        """Test that 401 is returned when no provider tokens."""
        mock_context = _make_mock_user_context(provider_tokens=None)
        monkeypatch.setattr(
            'openhands.app_server.git.git_router.depends_user_context',
            lambda: mock_context,
        )

        response = test_client.get('/git/installations', params={'provider': 'github'})

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_returns_422_for_unsupported_provider(self, test_client, monkeypatch):
        """Test that 422 is returned for unsupported provider."""
        mock_context = _make_mock_user_context(provider_tokens={'github': 'token'})

        # Patch the ProviderHandler
        monkeypatch.setattr(
            'openhands.app_server.git.git_router.depends_user_context',
            lambda: mock_context,
        )

        response = test_client.get('/git/installations', params={'provider': 'invalid'})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestRepositoriesEndpoint:
    """Test suite for /repositories endpoint."""

    def test_returns_401_when_no_provider_tokens(self, test_client, monkeypatch):
        """Test that 401 is returned when no provider tokens."""
        mock_context = _make_mock_user_context(provider_tokens=None)
        monkeypatch.setattr(
            'openhands.app_server.git.git_router.depends_user_context',
            lambda: mock_context,
        )

        response = test_client.get('/git/repositories', params={'provider': 'github'})

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
class TestGetUserInstallations:
    """Test suite for get_user_installations function."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_paginated_installations(self, mock_handler_cls):
        """Test that installations are returned with pagination."""
        # Arrange
        mock_handler = _make_mock_provider_handler()
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        result = await get_user_installations(
            provider=ProviderType.GITHUB,
            page_id=None,
            limit=2,
            user_context=mock_context,
        )

        # Assert
        assert result.items == ['inst-1', 'inst-2']
        # next_page_id is base64-encoded
        assert result.next_page_id == encode_page_id(2)

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_second_page_correctly(self, mock_handler_cls):
        """Test that second page of installations is returned correctly."""
        # Arrange
        mock_handler = _make_mock_provider_handler()
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act - request second page
        result = await get_user_installations(
            provider=ProviderType.GITHUB,
            page_id=encode_page_id(2),  # Second page starts at offset 2
            limit=2,
            user_context=mock_context,
        )

        # Assert
        assert result.items == ['inst-3', 'inst-4']
        assert result.next_page_id == encode_page_id(4)


@pytest.mark.asyncio
class TestGetUserRepositories:
    """Test suite for get_user_repositories function."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_repositories_with_correct_structure(self, mock_handler_cls):
        """Test that get_user_repositories returns correct RepositoryPage structure."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.get_repositories = AsyncMock(
            return_value=[
                Repository(
                    id='1',
                    full_name='user/repo1',
                    git_provider=ProviderType.GITHUB,
                    is_public=True,
                ),
                Repository(
                    id='2',
                    full_name='user/repo2',
                    git_provider=ProviderType.GITHUB,
                    is_public=False,
                ),
                Repository(
                    id='3',
                    full_name='user/repo3',
                    git_provider=ProviderType.GITHUB,
                    is_public=True,
                ),
            ]
        )
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        result = await get_user_repositories(
            provider=ProviderType.GITHUB,
            sort='updated',
            installation_id=None,
            page_id=None,
            limit=10,
            user_context=mock_context,
        )

        # Assert
        assert len(result.items) == 3
        assert result.items[0].id == '1'
        assert result.items[1].id == '2'
        assert result.items[2].id == '3'
        assert result.next_page_id is None  # No more pages

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_pagination_works_across_pages(self, mock_handler_cls):
        """Test that pagination works correctly across multiple pages.

        Note: This endpoint uses page-based pagination (passing page number to provider),
        not offset-based pagination like installations. The provider returns limit+1 items,
        and we check if there are more to determine next_page_id.
        """
        # Arrange
        mock_handler = MagicMock()

        # We'll set up the mock to return different data based on the page parameter
        # First call (page=1): return 3 items (limit+1), meaning there's a next page
        # Second call (page=2): return 3 items, meaning there's a next page
        # Third call (page=3): return 2 items, meaning it's the last page
        def mock_get_repositories(**kwargs):
            page = kwargs.get('page', 1)
            if page == 1:
                return [
                    Repository(
                        id=str(i),
                        full_name=f'user/repo{i}',
                        git_provider=ProviderType.GITHUB,
                        is_public=True,
                    )
                    for i in range(1, 4)  # 3 items = limit+1
                ]
            elif page == 2:
                return [
                    Repository(
                        id=str(i),
                        full_name=f'user/repo{i}',
                        git_provider=ProviderType.GITHUB,
                        is_public=True,
                    )
                    for i in range(4, 7)  # 3 items = limit+1
                ]
            else:
                return [
                    Repository(
                        id=str(i),
                        full_name=f'user/repo{i}',
                        git_provider=ProviderType.GITHUB,
                        is_public=True,
                    )
                    for i in range(7, 9)  # 2 items < limit+1 = last page
                ]

        mock_handler.get_repositories = AsyncMock(side_effect=mock_get_repositories)
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act - First page (page=1)
        result_page1 = await get_user_repositories(
            provider=ProviderType.GITHUB,
            sort='pushed',
            installation_id=None,
            page_id=None,  # This means page 1
            limit=2,
            user_context=mock_context,
        )

        # Assert - First page returns 2 items (truncated from limit+1=3), with next_page_id
        assert len(result_page1.items) == 2
        assert result_page1.items[0].id == '1'
        assert result_page1.items[1].id == '2'
        assert result_page1.next_page_id == encode_page_id(2)

        # Act - Second page (page=2)
        result_page2 = await get_user_repositories(
            provider=ProviderType.GITHUB,
            sort='pushed',
            installation_id=None,
            page_id=encode_page_id(2),  # This means page 2
            limit=2,
            user_context=mock_context,
        )

        # Assert - Second page returns next 2 items
        assert len(result_page2.items) == 2
        assert result_page2.items[0].id == '4'
        assert result_page2.items[1].id == '5'
        # next_page_id = page + 1 = 2 + 1 = 3, encoded as base64 = 'Mw'
        assert result_page2.next_page_id == encode_page_id(3)

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_passes_sort_parameter_to_provider(self, mock_handler_cls):
        """Test that sort parameter is passed through to the provider handler."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.get_repositories = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        await get_user_repositories(
            provider=ProviderType.GITHUB,
            sort='stars',
            installation_id=None,
            page_id=None,
            limit=10,
            user_context=mock_context,
        )

        # Assert - verify get_repositories was called with the sort parameter
        mock_handler.get_repositories.assert_called_once()
        call_kwargs = mock_handler.get_repositories.call_args.kwargs
        assert call_kwargs.get('sort') == 'stars'

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_passes_installation_id_to_provider(self, mock_handler_cls):
        """Test that installation_id filtering is passed through to the provider."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.get_repositories = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        await get_user_repositories(
            provider=ProviderType.GITHUB,
            sort='pushed',
            installation_id='app-123',
            page_id=None,
            limit=10,
            user_context=mock_context,
        )

        # Assert - verify get_repositories was called with installation_id
        mock_handler.get_repositories.assert_called_once()
        call_kwargs = mock_handler.get_repositories.call_args.kwargs
        assert call_kwargs.get('installation_id') == 'app-123'
