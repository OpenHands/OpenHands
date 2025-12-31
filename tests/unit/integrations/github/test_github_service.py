from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic import SecretStr

from openhands.integrations.github.github_service import GitHubService
from openhands.integrations.service_types import (
    AuthenticationError,
    OwnerType,
    ProviderType,
    Repository,
    User,
)
from openhands.server.types import AppMode


@pytest.mark.asyncio
async def test_github_service_token_handling():
    # Test initialization with SecretStr token
    token = SecretStr('test-token')
    service = GitHubService(user_id=None, token=token)
    assert service.token == token
    assert service.token.get_secret_value() == 'test-token'

    # Test headers contain the token correctly
    headers = await service._get_headers()
    assert headers['Authorization'] == 'Bearer test-token'
    assert headers['Accept'] == 'application/vnd.github.v3+json'

    # Test initialization without token
    service = GitHubService(user_id='test-user')
    assert service.token == SecretStr('')


@pytest.mark.asyncio
async def test_github_service_token_refresh():
    # Test that token refresh is only attempted when refresh=True
    token = SecretStr('test-token')
    service = GitHubService(user_id=None, token=token)
    assert not service.refresh

    # Test token expiry detection
    assert service._has_token_expired(401)
    assert not service._has_token_expired(200)
    assert not service._has_token_expired(404)

    # Test get_latest_token returns a copy of the current token
    latest_token = await service.get_latest_token()
    assert isinstance(latest_token, SecretStr)
    assert latest_token.get_secret_value() == 'test-token'  # Compare with known value


@pytest.mark.asyncio
async def test_github_service_fetch_data():
    # Mock httpx.AsyncClient for testing API calls
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'login': 'test-user'}
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch('httpx.AsyncClient', return_value=mock_client):
        service = GitHubService(user_id=None, token=SecretStr('test-token'))
        _ = await service._make_request('https://api.github.com/user')

        # Verify the request was made with correct headers
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        headers = call_args[1]['headers']
        assert headers['Authorization'] == 'Bearer test-token'

        # Test error handling with 401 status code
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message='401 Unauthorized', request=Mock(), response=mock_response
        )

        # Reset the mock to test error handling
        mock_client.get.reset_mock()
        mock_client.get.return_value = mock_response

        with pytest.raises(AuthenticationError):
            _ = await service._make_request('https://api.github.com/user')


@pytest.mark.asyncio
async def test_github_get_repositories_with_user_owner_type():
    """Test that get_repositories correctly sets owner_type field for user repositories."""
    service = GitHubService(user_id=None, token=SecretStr('test-token'))

    # Mock repository data for user repositories
    mock_repo_data = [
        {
            'id': 123,
            'full_name': 'test-user/test-repo',
            'private': False,
            'stargazers_count': 10,
            'owner': {'type': 'User'},  # User repository
        },
        {
            'id': 456,
            'full_name': 'test-user/another-repo',
            'private': True,
            'stargazers_count': 5,
            'owner': {'type': 'User'},  # User repository
        },
    ]

    with (
        patch.object(service, '_fetch_paginated_repos', return_value=mock_repo_data),
        patch.object(service, 'get_installations', return_value=[123]),
    ):
        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for user repositories
        for repo in repositories:
            assert repo.owner_type == OwnerType.USER
            assert isinstance(repo, Repository)
            assert repo.git_provider == ProviderType.GITHUB


@pytest.mark.asyncio
async def test_github_get_repositories_with_organization_owner_type():
    """Test that get_repositories correctly sets owner_type field for organization repositories."""
    service = GitHubService(user_id=None, token=SecretStr('test-token'))

    # Mock repository data for organization repositories
    mock_repo_data = [
        {
            'id': 789,
            'full_name': 'test-org/org-repo',
            'private': False,
            'stargazers_count': 25,
            'owner': {'type': 'Organization'},  # Organization repository
        },
        {
            'id': 101,
            'full_name': 'test-org/another-org-repo',
            'private': True,
            'stargazers_count': 15,
            'owner': {'type': 'Organization'},  # Organization repository
        },
    ]

    with (
        patch.object(service, '_fetch_paginated_repos', return_value=mock_repo_data),
        patch.object(service, 'get_installations', return_value=[123]),
    ):
        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for organization repositories
        for repo in repositories:
            assert repo.owner_type == OwnerType.ORGANIZATION
            assert isinstance(repo, Repository)
            assert repo.git_provider == ProviderType.GITHUB


@pytest.mark.asyncio
async def test_github_get_repositories_mixed_owner_types():
    """Test that get_repositories correctly handles mixed user and organization repositories."""
    service = GitHubService(user_id=None, token=SecretStr('test-token'))

    # Mock repository data with mixed owner types
    mock_repo_data = [
        {
            'id': 123,
            'full_name': 'test-user/user-repo',
            'private': False,
            'stargazers_count': 10,
            'owner': {'type': 'User'},  # User repository
        },
        {
            'id': 456,
            'full_name': 'test-org/org-repo',
            'private': True,
            'stargazers_count': 25,
            'owner': {'type': 'Organization'},  # Organization repository
        },
    ]

    with (
        patch.object(service, '_fetch_paginated_repos', return_value=mock_repo_data),
        patch.object(service, 'get_installations', return_value=[123]),
    ):
        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for each repository
        user_repo = next(repo for repo in repositories if 'user-repo' in repo.full_name)
        org_repo = next(repo for repo in repositories if 'org-repo' in repo.full_name)

        assert user_repo.owner_type == OwnerType.USER
        assert org_repo.owner_type == OwnerType.ORGANIZATION


@pytest.mark.asyncio
async def test_github_get_repositories_owner_type_fallback():
    """Test that owner_type defaults to USER when owner type is not 'Organization'."""
    service = GitHubService(user_id=None, token=SecretStr('test-token'))

    # Mock repository data with missing or unexpected owner type
    mock_repo_data = [
        {
            'id': 123,
            'full_name': 'test-user/test-repo',
            'private': False,
            'stargazers_count': 10,
            'owner': {'type': 'User'},  # Explicitly User
        },
        {
            'id': 456,
            'full_name': 'test-user/another-repo',
            'private': True,
            'stargazers_count': 5,
            'owner': {'type': 'Bot'},  # Unexpected type
        },
        {
            'id': 789,
            'full_name': 'test-user/third-repo',
            'private': False,
            'stargazers_count': 15,
            'owner': {},  # Missing type
        },
    ]

    with (
        patch.object(service, '_fetch_paginated_repos', return_value=mock_repo_data),
        patch.object(service, 'get_installations', return_value=[123]),
    ):
        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify all repositories default to USER owner_type
        for repo in repositories:
            assert repo.owner_type == OwnerType.USER


@pytest.mark.asyncio
async def test_github_search_repositories_with_organizations():
    """Test that search_repositories includes user organizations in the search scope."""
    service = GitHubService(user_id='test-user', token=SecretStr('test-token'))

    # Mock user data
    mock_user = User(
        id='123', login='testuser', avatar_url='https://example.com/avatar.jpg'
    )

    # Mock search response
    mock_search_response = {
        'items': [
            {
                'id': 1,
                'name': 'OpenHands',
                'full_name': 'All-Hands-AI/OpenHands',
                'private': False,
                'html_url': 'https://github.com/All-Hands-AI/OpenHands',
                'clone_url': 'https://github.com/All-Hands-AI/OpenHands.git',
                'pushed_at': '2023-01-01T00:00:00Z',
                'owner': {'login': 'All-Hands-AI', 'type': 'Organization'},
            }
        ]
    }

    with (
        patch.object(service, 'get_user', return_value=mock_user),
        patch.object(
            service,
            'get_organizations_from_installations',
            return_value=['All-Hands-AI', 'example-org'],
        ),
        patch.object(
            service, '_make_request', return_value=(mock_search_response, {})
        ) as mock_request,
    ):
        repositories = await service.search_repositories(
            query='openhands',
            per_page=10,
            sort='stars',
            order='desc',
            public=False,
            app_mode=AppMode.SAAS,
        )

        # Verify that separate requests were made for user and each organization
        assert mock_request.call_count == 3

        # Check the calls made
        calls = mock_request.call_args_list

        # First call should be for user repositories
        user_call = calls[0]
        user_params = user_call[0][1]  # Second argument is params
        assert user_params['q'] == 'in:name openhands user:testuser'

        # Second call should be for first organization
        org1_call = calls[1]
        org1_params = org1_call[0][1]
        assert org1_params['q'] == 'openhands org:All-Hands-AI'

        # Third call should be for second organization
        org2_call = calls[2]
        org2_params = org2_call[0][1]
        assert org2_params['q'] == 'openhands org:example-org'

        # Verify repositories are returned (3 copies since each call returns the same mock response)
        assert len(repositories) == 3
        assert all(repo.full_name == 'All-Hands-AI/OpenHands' for repo in repositories)


@pytest.mark.asyncio
async def test_github_get_user_organizations():
    """Test that get_user_organizations fetches user's organizations."""
    service = GitHubService(user_id='test-user', token=SecretStr('test-token'))

    mock_orgs_response = [
        {'login': 'All-Hands-AI', 'id': 1},
        {'login': 'example-org', 'id': 2},
    ]

    with patch.object(service, '_make_request', return_value=(mock_orgs_response, {})):
        orgs = await service.get_user_organizations()

        assert orgs == ['All-Hands-AI', 'example-org']


@pytest.mark.asyncio
async def test_github_get_user_organizations_error_handling():
    """Test that get_user_organizations handles errors gracefully."""
    service = GitHubService(user_id='test-user', token=SecretStr('test-token'))

    with patch.object(service, '_make_request', side_effect=Exception('API Error')):
        orgs = await service.get_user_organizations()

        # Should return empty list on error
        assert orgs == []


@pytest.mark.asyncio
async def test_github_service_base_url_configuration():
    """Test that BASE_URL is correctly configured based on base_domain."""
    # Test default GitHub.com configuration
    service = GitHubService(user_id=None, token=SecretStr('test-token'))
    assert service.BASE_URL == 'https://api.github.com'

    # Test GitHub Enterprise Server configuration
    service = GitHubService(
        user_id=None, token=SecretStr('test-token'), base_domain='github.enterprise.com'
    )
    assert service.BASE_URL == 'https://github.enterprise.com/api/v3'

    # Test that github.com base_domain doesn't change the URL
    service = GitHubService(
        user_id=None, token=SecretStr('test-token'), base_domain='github.com'
    )
    assert service.BASE_URL == 'https://api.github.com'


@pytest.mark.asyncio
async def test_github_service_graphql_url_enterprise_server():
    """Test that GraphQL URL is correctly constructed for GitHub Enterprise Server."""
    # Mock httpx.AsyncClient for testing GraphQL calls
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'data': {'viewer': {'login': 'test-user'}}}
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch('httpx.AsyncClient', return_value=mock_client):
        # Test GitHub Enterprise Server
        service = GitHubService(
            user_id=None,
            token=SecretStr('test-token'),
            base_domain='github.enterprise.com',
        )

        query = 'query { viewer { login } }'
        variables = {}

        await service.execute_graphql_query(query, variables)

        # Verify the GraphQL request was made to the CORRECT URL
        # For GitHub Enterprise Server, it should be /api/graphql, not /api/v3/graphql
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        actual_url = call_args[0][0]  # First positional argument is the URL

        # After the fix, GraphQL URL should be correctly constructed for GitHub Enterprise Server
        # The URL should be /api/graphql, not /api/v3/graphql
        assert actual_url == 'https://github.enterprise.com/api/graphql'


@pytest.mark.asyncio
async def test_github_service_graphql_url_github_com():
    """Test that GraphQL URL is correctly constructed for GitHub.com."""
    # Mock httpx.AsyncClient for testing GraphQL calls
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'data': {'viewer': {'login': 'test-user'}}}
    mock_response.raise_for_status = Mock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch('httpx.AsyncClient', return_value=mock_client):
        # Test GitHub.com (should work correctly)
        service = GitHubService(user_id=None, token=SecretStr('test-token'))

        query = 'query { viewer { login } }'
        variables = {}

        await service.execute_graphql_query(query, variables)

        # Verify the GraphQL request was made to the correct URL for GitHub.com
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        actual_url = call_args[0][0]  # First positional argument is the URL

        # This should be correct for GitHub.com
        assert actual_url == 'https://api.github.com/graphql'
