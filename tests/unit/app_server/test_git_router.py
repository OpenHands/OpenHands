"""Unit tests for the git_router endpoints.

This module tests the git router endpoints,
focusing on pagination and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from openhands.app_server.git.git_models import SortOrder
from openhands.app_server.git.git_router import (
    router,
    search_branches,
    search_repositories,
    search_suggested_tasks,
    search_user_installations,
)
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.dependencies import check_session_api_key
from openhands.app_server.utils.paging_utils import encode_page_id, paginate_results
from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import (
    Branch,
    ProviderType,
    Repository,
    SuggestedTask,
    TaskType,
)


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

    def test_returns_401_when_no_provider_tokens(self, test_client):
        """Test that 401 is returned when no provider tokens."""
        with patch(
            'openhands.app_server.user.auth_user_context.AuthUserContext.get_provider_tokens',
            AsyncMock(return_value=None),
        ):
            response = test_client.get(
                '/git/installations/search', params={'provider': 'github'}
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_returns_422_for_unsupported_provider(self, test_client):
        """Test that 422 is returned for unsupported provider."""
        with patch(
            'openhands.app_server.user.auth_user_context.AuthUserContext.get_provider_tokens',
            AsyncMock(return_value={'github': 'token'}),
        ):
            response = test_client.get(
                '/git/installations/search', params={'provider': 'invalid'}
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
class TestSearchUserInstallations:
    """Test suite for search_user_installations function."""

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
        result = await search_user_installations(
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
        result = await search_user_installations(
            provider=ProviderType.GITHUB,
            page_id=encode_page_id(2),  # Second page starts at offset 2
            limit=2,
            user_context=mock_context,
        )

        # Assert
        assert result.items == ['inst-3', 'inst-4']
        assert result.next_page_id == encode_page_id(4)


@pytest.mark.asyncio
class TestSearchRepositories:
    """Test suite for search_repositories function (handles both user repos and search)."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_user_repositories_without_query(self, mock_handler_cls):
        """Test that repositories_search returns user repositories when no query is provided."""
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

        # Act - call without query to get user repos
        result = await search_repositories(
            provider=ProviderType.GITHUB,
            query=None,
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
    async def test_search_repositories_with_query(self, mock_handler_cls):
        """Test repository search when query is provided.

        This tests the search path (with query) which calls search_repositories
        instead of get_repositories, and verifies sort_order is parsed correctly.
        """
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_repositories = AsyncMock(
            return_value=[
                Repository(
                    id='10',
                    full_name='org/searched-repo',
                    git_provider=ProviderType.GITHUB,
                    is_public=True,
                ),
                Repository(
                    id='11',
                    full_name='user/searched-repo',
                    git_provider=ProviderType.GITHUB,
                    is_public=False,
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

        # Act - call with query and sort_order to trigger search path
        result = await search_repositories(
            provider=ProviderType.GITHUB,
            query='my-search-term',
            installation_id=None,
            page_id=None,
            limit=10,
            sort_order=SortOrder.STAR_DESC,  # This should be parsed into sort='stars', order='desc'
            user_context=mock_context,
        )

        # Assert - verify search_repositories was called (not get_repositories)
        mock_handler.search_repositories.assert_called_once()
        call_kwargs = mock_handler.search_repositories.call_args.kwargs

        # Verify query is passed
        assert call_kwargs.get('query') == 'my-search-term'

        # Verify sort and order are parsed from sort_order ('stars-desc' -> sort='stars', order='desc')
        assert call_kwargs.get('sort') == 'stars'
        assert call_kwargs.get('order') == 'desc'

        # Verify per_page is limit + 1
        assert call_kwargs.get('per_page') == 11

        # Verify results are returned
        assert len(result.items) == 2
        assert result.items[0].id == '10'
        assert result.items[1].id == '11'

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_search_repositories_sort_order_asc(self, mock_handler_cls):
        """Test that sort_order ascending is parsed correctly."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_repositories = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act - call with sort_order ascending
        await search_repositories(
            provider=ProviderType.GITHUB,
            query='test',
            installation_id=None,
            page_id=None,
            limit=10,
            sort_order=SortOrder.FORKS_ASC,
            user_context=mock_context,
        )

        # Assert
        call_kwargs = mock_handler.search_repositories.call_args.kwargs
        assert call_kwargs.get('sort') == 'forks'
        assert call_kwargs.get('order') == 'asc'

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_search_repositories_default_sort_order(self, mock_handler_cls):
        """Test default sort order when sort_order is not provided."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_repositories = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act - call with query but no sort_order (uses default: stars, desc)
        await search_repositories(
            provider=ProviderType.GITHUB,
            query='test',
            installation_id=None,
            page_id=None,
            limit=10,
            sort_order=None,
            user_context=mock_context,
        )

        # Assert - defaults should be used
        call_kwargs = mock_handler.search_repositories.call_args.kwargs
        assert call_kwargs.get('sort') == 'stars'  # Default
        assert call_kwargs.get('order') == 'desc'  # Default

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
        result_page1 = await search_repositories(
            provider=ProviderType.GITHUB,
            query=None,
            installation_id=None,
            page_id=None,  # This means page 1
            limit=2,
            sort_order=None,
            user_context=mock_context,
        )

        # Assert - First page returns 2 items (truncated from limit+1=3), with next_page_id
        assert len(result_page1.items) == 2
        assert result_page1.items[0].id == '1'
        assert result_page1.items[1].id == '2'
        assert result_page1.next_page_id == encode_page_id(2)

        # Act - Second page (page=2)
        result_page2 = await search_repositories(
            provider=ProviderType.GITHUB,
            query=None,
            installation_id=None,
            page_id=encode_page_id(2),  # This means page 2
            limit=2,
            sort_order=None,
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
        await search_repositories(
            provider=ProviderType.GITHUB,
            query=None,
            installation_id=None,
            page_id=None,
            limit=10,
            user_context=mock_context,
        )

        # Assert - verify get_repositories was called with the sort parameter
        mock_handler.get_repositories.assert_called_once()
        call_kwargs = mock_handler.get_repositories.call_args.kwargs
        assert call_kwargs.get('sort') == 'pushed'

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
        await search_repositories(
            provider=ProviderType.GITHUB,
            query=None,
            installation_id='app-123',
            page_id=None,
            limit=10,
            sort_order=None,
            user_context=mock_context,
        )

        # Assert - verify get_repositories was called with installation_id
        mock_handler.get_repositories.assert_called_once()
        call_kwargs = mock_handler.get_repositories.call_args.kwargs
        assert call_kwargs.get('installation_id') == 'app-123'

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_paginated_search_results(self, mock_handler_cls):
        """Test that search repositories are returned with pagination when query is provided."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_repositories = AsyncMock(
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
                    is_public=True,
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
        result = await search_repositories(
            provider=ProviderType.GITHUB,
            query='test',
            page_id=None,
            limit=2,
            sort_order=SortOrder.STAR_DESC,
            user_context=mock_context,
        )

        # Assert
        assert result.items == [
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
                is_public=True,
            ),
        ]
        assert result.next_page_id == encode_page_id(2)

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_parses_sort_order_correctly(self, mock_handler_cls):
        """Test that sort_order enum is parsed into sort and order components."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_repositories = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        await search_repositories(
            provider=ProviderType.GITHUB,
            query='test',
            page_id=None,
            limit=5,
            sort_order=SortOrder.FORKS_ASC,
            user_context=mock_context,
        )

        # Assert - verify search_repositories was called with parsed sort and order
        mock_handler.search_repositories.assert_called_once()
        call_kwargs = mock_handler.search_repositories.call_args.kwargs
        assert call_kwargs.get('sort') == 'forks'
        assert call_kwargs.get('order') == 'asc'

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_empty_first_page(self, mock_handler_cls):
        """Test that empty results return empty items with no next_page_id."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_repositories = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        result = await search_repositories(
            provider=ProviderType.GITHUB,
            query='nonexistent-repo-xyz',
            page_id=None,
            limit=5,
            sort_order=SortOrder.STAR_DESC,
            user_context=mock_context,
        )

        # Assert
        assert result.items == []
        assert result.next_page_id is None

    def test_returns_401_when_no_provider_tokens(self, test_client, monkeypatch):
        """Test that 401 is returned when no provider tokens."""
        with patch(
            'openhands.app_server.user.auth_user_context.AuthUserContext.get_provider_tokens',
            AsyncMock(return_value=None),
        ):
            response = test_client.get(
                '/git/repositories/search',
                params={'provider': 'github', 'query': 'test'},
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
class TestSearchBranches:
    """Test suite for search_branches function."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_paginated_branches(self, mock_handler_cls):
        """Test that search branches are returned with pagination."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_branches = AsyncMock(
            return_value=[
                Branch(name='main', commit_sha='abc123', protected=False),
                Branch(name='develop', commit_sha='def456', protected=False),
                Branch(name='feature-branch', commit_sha='ghi789', protected=False),
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
        result = await search_branches(
            provider=ProviderType.GITHUB,
            repository='user/repo',
            query='main',
            page_id=None,
            limit=2,
            user_context=mock_context,
        )

        # Assert
        assert len(result.items) == 2
        assert result.items[0].name == 'main'
        assert result.items[1].name == 'develop'
        assert result.next_page_id == encode_page_id(2)

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_passes_parameters_to_provider(self, mock_handler_cls):
        """Test that all parameters are passed through to the provider."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.search_branches = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        await search_branches(
            provider=ProviderType.GITHUB,
            repository='user/repo',
            query='feature',
            page_id=None,
            limit=10,
            user_context=mock_context,
        )

        # Assert
        mock_handler.search_branches.assert_called_once()
        call_kwargs = mock_handler.search_branches.call_args.kwargs
        assert call_kwargs.get('selected_provider') == ProviderType.GITHUB
        assert call_kwargs.get('repository') == 'user/repo'
        assert call_kwargs.get('query') == 'feature'
        assert call_kwargs.get('per_page') == 11  # limit + 1

    def test_returns_401_when_no_provider_tokens(self, test_client):
        """Test that 401 is returned when no provider tokens."""
        with patch(
            'openhands.app_server.user.auth_user_context.AuthUserContext.get_provider_tokens',
            AsyncMock(return_value=None),
        ):
            response = test_client.get(
                '/git/branches/search',
                params={
                    'provider': 'github',
                    'repository': 'user/repo',
                    'query': 'main',
                },
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
class TestSearchSuggestedTasks:
    """Test suite for search_suggested_tasks function."""

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_paginated_tasks(self, mock_handler_cls):
        """Test that suggested tasks are returned with pagination."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.get_suggested_tasks = AsyncMock(
            return_value=[
                SuggestedTask(
                    git_provider=ProviderType.GITHUB,
                    task_type=TaskType.OPEN_ISSUE,
                    repo='user/repo',
                    issue_number=1,
                    title='Fix bug in login',
                ),
                SuggestedTask(
                    git_provider=ProviderType.GITHUB,
                    task_type=TaskType.OPEN_PR,
                    repo='user/repo',
                    issue_number=2,
                    title='Add new feature',
                ),
                SuggestedTask(
                    git_provider=ProviderType.GITHUB,
                    task_type=TaskType.OPEN_ISSUE,
                    repo='user/repo2',
                    issue_number=3,
                    title='Update documentation',
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
        result = await search_suggested_tasks(
            page_id=None,
            limit=2,
            user_context=mock_context,
        )

        # Assert
        assert len(result.items) == 2
        assert result.items[0].issue_number == 1
        assert result.items[1].issue_number == 2
        assert result.next_page_id == encode_page_id(2)

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_second_page(self, mock_handler_cls):
        """Test that second page returns correct items."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.get_suggested_tasks = AsyncMock(
            return_value=[
                SuggestedTask(
                    git_provider=ProviderType.GITHUB,
                    task_type=TaskType.OPEN_ISSUE,
                    repo='user/repo',
                    issue_number=1,
                    title='Fix bug in login',
                ),
                SuggestedTask(
                    git_provider=ProviderType.GITHUB,
                    task_type=TaskType.OPEN_PR,
                    repo='user/repo',
                    issue_number=2,
                    title='Add new feature',
                ),
                SuggestedTask(
                    git_provider=ProviderType.GITHUB,
                    task_type=TaskType.OPEN_ISSUE,
                    repo='user/repo2',
                    issue_number=3,
                    title='Update documentation',
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

        # Act - request second page
        result = await search_suggested_tasks(
            page_id=encode_page_id(2),
            limit=2,
            user_context=mock_context,
        )

        # Assert
        assert len(result.items) == 1
        assert result.items[0].issue_number == 3
        assert result.next_page_id is None

    @pytest.mark.asyncio
    @patch('openhands.app_server.git.git_router.ProviderHandler')
    async def test_returns_empty_when_no_tasks(self, mock_handler_cls):
        """Test that empty results return empty items."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.get_suggested_tasks = AsyncMock(return_value=[])
        mock_handler_cls.return_value = mock_handler

        mock_context = _make_mock_user_context(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(user_id='user-123', token='token')
            },
            user_id='user-123',
        )

        # Act
        result = await search_suggested_tasks(
            page_id=None,
            limit=10,
            user_context=mock_context,
        )

        # Assert
        assert result.items == []
        assert result.next_page_id is None

    def test_returns_401_when_no_provider_tokens(self, test_client):
        """Test that 401 is returned when no provider tokens."""
        with patch(
            'openhands.app_server.user.auth_user_context.AuthUserContext.get_provider_tokens',
            AsyncMock(return_value=None),
        ):
            response = test_client.get('/git/suggested-tasks/search')
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
