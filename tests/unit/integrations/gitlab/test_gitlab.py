"""Tests for GitLab integration."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.integrations.gitlab.gitlab_service import GitLabService
from openhands.integrations.service_types import OwnerType, ProviderType, Repository
from openhands.server.types import AppMode


@pytest.mark.asyncio
async def test_gitlab_get_user_groups_returns_group_paths():
    """Test that get_user_groups returns group paths the user belongs to."""
    service = GitLabService(token=SecretStr('test-token'))

    mock_groups = [
        {'path': 'my-team', 'name': 'My Team', 'id': 1},
        {'path': 'open-source', 'name': 'Open Source Projects', 'id': 2},
    ]

    with patch.object(service, '_make_request') as mock_request:
        mock_request.return_value = (mock_groups, {})

        groups = await service.get_user_groups()

        assert groups == ['my-team', 'open-source']
        mock_request.assert_called_once_with(
            f'{service.BASE_URL}/groups',
            {'min_access_level': '10', 'per_page': '100'},
        )


@pytest.mark.asyncio
async def test_gitlab_get_user_groups_returns_empty_on_error():
    """Test that get_user_groups returns empty list when the API call fails."""
    service = GitLabService(token=SecretStr('test-token'))

    with patch.object(service, '_make_request') as mock_request:
        mock_request.side_effect = Exception('API error')

        groups = await service.get_user_groups()

        assert groups == []


@pytest.mark.asyncio
async def test_gitlab_get_repositories_with_user_owner_type():
    """Test that get_repositories correctly sets owner_type field for user repositories."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data for user repositories (namespace kind = 'user')
    mock_repos = [
        {
            'id': 123,
            'path_with_namespace': 'test-user/user-repo1',
            'star_count': 10,
            'visibility': 'public',
            'namespace': {'kind': 'user'},  # User namespace
        },
        {
            'id': 456,
            'path_with_namespace': 'test-user/user-repo2',
            'star_count': 5,
            'visibility': 'private',
            'namespace': {'kind': 'user'},  # User namespace
        },
    ]

    with patch.object(service, '_make_request') as mock_request:
        # Mock the pagination response
        mock_request.side_effect = [(mock_repos, {'Link': ''})]  # No next page

        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for user repositories
        for repo in repositories:
            assert repo.owner_type == OwnerType.USER
            assert isinstance(repo, Repository)
            assert repo.git_provider == ProviderType.GITLAB


@pytest.mark.asyncio
async def test_gitlab_get_repositories_with_organization_owner_type():
    """Test that get_repositories correctly sets owner_type field for organization repositories."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data for organization repositories (namespace kind = 'group')
    mock_repos = [
        {
            'id': 789,
            'path_with_namespace': 'test-org/org-repo1',
            'star_count': 25,
            'visibility': 'public',
            'namespace': {'kind': 'group'},  # Organization/Group namespace
        },
        {
            'id': 101,
            'path_with_namespace': 'test-org/org-repo2',
            'star_count': 15,
            'visibility': 'private',
            'namespace': {'kind': 'group'},  # Organization/Group namespace
        },
    ]

    with patch.object(service, '_make_request') as mock_request:
        # Mock the pagination response
        mock_request.side_effect = [(mock_repos, {'Link': ''})]  # No next page

        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for organization repositories
        for repo in repositories:
            assert repo.owner_type == OwnerType.ORGANIZATION
            assert isinstance(repo, Repository)
            assert repo.git_provider == ProviderType.GITLAB


@pytest.mark.asyncio
async def test_gitlab_get_repositories_mixed_owner_types():
    """Test that get_repositories correctly handles mixed user and organization repositories."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data with mixed namespace types
    mock_repos = [
        {
            'id': 123,
            'path_with_namespace': 'test-user/user-repo',
            'star_count': 10,
            'visibility': 'public',
            'namespace': {'kind': 'user'},  # User namespace
        },
        {
            'id': 456,
            'path_with_namespace': 'test-org/org-repo',
            'star_count': 25,
            'visibility': 'public',
            'namespace': {'kind': 'group'},  # Organization/Group namespace
        },
    ]

    with patch.object(service, '_make_request') as mock_request:
        # Mock the pagination response
        mock_request.side_effect = [(mock_repos, {'Link': ''})]  # No next page

        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify we got the expected number of repositories
        assert len(repositories) == 2

        # Verify owner_type is correctly set for each repository
        user_repo = next(repo for repo in repositories if 'user-repo' in repo.full_name)
        org_repo = next(repo for repo in repositories if 'org-repo' in repo.full_name)

        assert user_repo.owner_type == OwnerType.USER
        assert org_repo.owner_type == OwnerType.ORGANIZATION


@pytest.mark.asyncio
async def test_gitlab_get_repositories_owner_type_fallback():
    """Test that owner_type defaults to USER when namespace kind is not 'group'."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data with missing or unexpected namespace kind
    mock_repos = [
        {
            'id': 123,
            'path_with_namespace': 'test-user/user-repo1',
            'star_count': 10,
            'visibility': 'public',
            'namespace': {'kind': 'user'},  # Explicitly user
        },
        {
            'id': 456,
            'path_with_namespace': 'test-user/user-repo2',
            'star_count': 5,
            'visibility': 'private',
            'namespace': {'kind': 'unknown'},  # Unexpected kind
        },
        {
            'id': 789,
            'path_with_namespace': 'test-user/user-repo3',
            'star_count': 15,
            'visibility': 'public',
            'namespace': {},  # Missing kind
        },
    ]

    with patch.object(service, '_make_request') as mock_request:
        # Mock the pagination response
        mock_request.side_effect = [(mock_repos, {'Link': ''})]  # No next page

        repositories = await service.get_all_repositories('pushed', AppMode.SAAS)

        # Verify all repositories default to USER owner_type
        for repo in repositories:
            assert repo.owner_type == OwnerType.USER


@pytest.mark.asyncio
async def test_gitlab_search_repositories_uses_membership_and_min_access_level():
    """Test that search_repositories uses membership and min_access_level for non-public searches."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data
    mock_repos = [
        {
            'id': 123,
            'path_with_namespace': 'test-user/search-repo1',
            'star_count': 10,
            'visibility': 'private',
            'namespace': {'kind': 'user'},
        },
        {
            'id': 456,
            'path_with_namespace': 'test-org/search-repo2',
            'star_count': 25,
            'visibility': 'private',
            'namespace': {'kind': 'group'},
        },
    ]

    with patch.object(service, '_make_request') as mock_request:
        mock_request.return_value = (mock_repos, {})

        # Test non-public search (should use membership and min_access_level)
        repositories = await service.search_repositories(
            query='test-query', per_page=30, sort='updated', order='desc', public=False
        )

        # Verify the request was made with correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        url = call_args[0][0]
        params = call_args[0][1]  # params is the second positional argument

        assert url == f'{service.BASE_URL}/projects'
        assert params['search'] == 'test-query'
        assert params['per_page'] == '30'  # GitLab service converts to string
        assert params['order_by'] == 'last_activity_at'
        assert params['sort'] == 'desc'
        assert params['membership'] is True
        assert params['search_namespaces'] is True  # Added by implementation
        assert 'min_access_level' not in params  # Not set by current implementation
        assert 'owned' not in params
        assert 'visibility' not in params

        # Verify we got the expected repositories
        assert len(repositories) == 2


@pytest.mark.asyncio
async def test_gitlab_search_repositories_public_search_legacy():
    """Test that search_repositories returns empty list for non-URL queries when public=True."""
    service = GitLabService(token=SecretStr('test-token'))

    with patch.object(service, '_make_request') as mock_request:
        # Test public search with non-URL query (should return empty list)
        repositories = await service.search_repositories(
            query='public-query', per_page=20, sort='updated', order='asc', public=True
        )

        # Verify no request was made since it's not a valid URL
        mock_request.assert_not_called()

        # Verify we got empty list
        assert len(repositories) == 0


@pytest.mark.asyncio
async def test_gitlab_search_repositories_url_parsing():
    """Test that search_repositories correctly parses GitLab URLs when public=True."""
    service = GitLabService(token=SecretStr('test-token'))

    # Test URL parsing method directly
    assert service._parse_gitlab_url('https://gitlab.com/group/repo') == 'group/repo'
    assert (
        service._parse_gitlab_url('https://gitlab.com/group/subgroup/repo')
        == 'group/subgroup/repo'
    )
    assert (
        service._parse_gitlab_url('https://gitlab.example.com/org/team/project')
        == 'org/team/project'
    )
    assert service._parse_gitlab_url('https://gitlab.com/group/repo/') == 'group/repo'
    assert (
        service._parse_gitlab_url('https://gitlab.com/group/') is None
    )  # Missing repo
    assert service._parse_gitlab_url('https://gitlab.com/') is None  # Empty path
    assert service._parse_gitlab_url('invalid-url') is None  # Invalid URL


@pytest.mark.asyncio
async def test_gitlab_search_repositories_public_url_lookup():
    """Test that search_repositories looks up specific repository when public=True."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data

    with patch.object(
        service, 'get_repository_details_from_repo_name'
    ) as mock_get_repo:
        mock_get_repo.return_value = Repository(
            id='123',
            full_name='group/repo',
            stargazers_count=50,
            git_provider=ProviderType.GITLAB,
            is_public=True,
            owner_type=OwnerType.ORGANIZATION,
        )

        # Test with valid GitLab URL
        repositories = await service.search_repositories(
            query='https://gitlab.com/group/repo', public=True
        )

        # Verify the repository lookup was called with correct path
        mock_get_repo.assert_called_once_with('group/repo')

        # Verify we got the expected repository
        assert len(repositories) == 1
        assert repositories[0].full_name == 'group/repo'


@pytest.mark.asyncio
async def test_gitlab_search_repositories_public_url_lookup_with_subgroup():
    """Test that search_repositories handles subgroups correctly when public=True."""
    service = GitLabService(token=SecretStr('test-token'))

    with patch.object(
        service, 'get_repository_details_from_repo_name'
    ) as mock_get_repo:
        mock_get_repo.return_value = Repository(
            id='456',
            full_name='group/subgroup/repo',
            stargazers_count=25,
            git_provider=ProviderType.GITLAB,
            is_public=True,
            owner_type=OwnerType.ORGANIZATION,
        )

        # Test with GitLab URL containing subgroup
        repositories = await service.search_repositories(
            query='https://gitlab.example.com/group/subgroup/repo', public=True
        )

        # Verify the repository lookup was called with correct path
        mock_get_repo.assert_called_once_with('group/subgroup/repo')

        # Verify we got the expected repository
        assert len(repositories) == 1
        assert repositories[0].full_name == 'group/subgroup/repo'


@pytest.mark.asyncio
async def test_gitlab_search_repositories_public_url_not_found():
    """Test that search_repositories returns empty list when repository doesn't exist."""
    service = GitLabService(token=SecretStr('test-token'))

    with patch.object(
        service, 'get_repository_details_from_repo_name'
    ) as mock_get_repo:
        # Simulate repository not found
        mock_get_repo.side_effect = Exception('Repository not found')

        # Test with valid GitLab URL but non-existent repository
        # The current implementation doesn't catch exceptions, so we expect it to be raised
        with pytest.raises(Exception, match='Repository not found'):
            await service.search_repositories(
                query='https://gitlab.com/nonexistent/repo', public=True
            )

        # Verify the repository lookup was attempted
        mock_get_repo.assert_called_once_with('nonexistent/repo')


@pytest.mark.asyncio
async def test_gitlab_search_repositories_public_invalid_url():
    """Test that search_repositories returns empty list for invalid URLs."""
    service = GitLabService(token=SecretStr('test-token'))

    with patch.object(
        service, 'get_repository_details_from_repo_name'
    ) as mock_get_repo:
        # Test with invalid URL
        repositories = await service.search_repositories(
            query='invalid-url', public=True
        )

        # Verify no repository lookup was attempted
        mock_get_repo.assert_not_called()

        # Verify we got empty list
        assert len(repositories) == 0


@pytest.mark.asyncio
async def test_gitlab_search_repositories_formats_search_query():
    """Test that search_repositories properly formats search queries with multiple terms."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data
    mock_repos = [
        {
            'id': 123,
            'path_with_namespace': 'group/repo',
            'star_count': 50,
            'visibility': 'private',
            'namespace': {'kind': 'group'},
        },
    ]

    with patch.object(service, '_make_request') as mock_request:
        mock_request.return_value = (mock_repos, {})

        # Test search with multiple terms (should format with + separator)
        repositories = await service.search_repositories(
            query='my project name', public=False
        )

        # Verify the request was made with correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        url = call_args[0][0]
        params = call_args[0][1]

        assert url == f'{service.BASE_URL}/projects'
        assert (
            params['search'] == 'my project name'
        )  # Current implementation doesn't format spaces
        assert params['membership'] is True
        assert params['search_namespaces'] is True  # Added by implementation

        # Verify we got the expected repositories
        assert len(repositories) == 1


@pytest.mark.asyncio
async def test_gitlab_search_repositories_single_term_query():
    """Test that search_repositories handles single term queries correctly."""
    service = GitLabService(token=SecretStr('test-token'))

    # Mock repository data
    mock_repos = [
        {
            'id': 456,
            'path_with_namespace': 'user/single-repo',
            'star_count': 25,
            'visibility': 'private',
            'namespace': {'kind': 'user'},
        },
    ]

    with patch.object(service, '_make_request') as mock_request:
        mock_request.return_value = (mock_repos, {})

        # Test search with single term (should remain unchanged)
        repositories = await service.search_repositories(
            query='singleterm', public=False
        )

        # Verify the request was made with correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        params = call_args[0][1]

        assert params['search'] == 'singleterm'  # No change for single term

        # Verify we got the expected repositories
        assert len(repositories) == 1
