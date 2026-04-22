from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.github.github_service import GitHubService
from openhands.integrations.service_types import Branch, PaginatedBranchesResponse


@pytest.mark.asyncio
async def test_get_paginated_branches_github_basic_next_page():
    service = GitHubService(token=SecretStr('t'))

    mock_response = [
        {
            'name': 'main',
            'commit': {
                'sha': 'abc123',
                'commit': {'committer': {'date': '2024-01-01T12:00:00Z'}},
            },
            'protected': True,
        },
        {
            'name': 'feature/foo',
            'commit': {
                'sha': 'def456',
                'commit': {'committer': {'date': '2024-01-02T15:30:00Z'}},
            },
            'protected': False,
        },
    ]
    headers = {
        # Include rel="next" to indicate there is another page
        'Link': '<https://api.github.com/repos/o/r/branches?page=3>; rel="next"'
    }

    with patch.object(service, '_make_request', return_value=(mock_response, headers)):
        result = await service.get_paginated_branches('owner/repo', page=2, per_page=2)

        assert isinstance(result, PaginatedBranchesResponse)
        assert result.current_page == 2
        assert result.per_page == 2
        assert result.has_next_page is True
        assert result.total_count is None  # GitHub does not provide total count
        assert len(result.branches) == 2

        b0, b1 = result.branches
        assert isinstance(b0, Branch) and isinstance(b1, Branch)
        assert b0.name == 'main'
        assert b0.commit_sha == 'abc123'
        assert b0.protected is True
        assert b0.last_push_date == '2024-01-01T12:00:00Z'
        assert b1.name == 'feature/foo'
        assert b1.commit_sha == 'def456'
        assert b1.protected is False
        assert b1.last_push_date == '2024-01-02T15:30:00Z'


@pytest.mark.asyncio
async def test_get_paginated_branches_github_no_next_page():
    service = GitHubService(token=SecretStr('t'))

    mock_response = [
        {
            'name': 'dev',
            'commit': {
                'sha': 'zzz999',
                'commit': {'committer': {'date': '2024-01-03T00:00:00Z'}},
            },
            'protected': False,
        }
    ]
    headers = {
        # No rel="next" – should be treated as last page
        'Link': '<https://api.github.com/repos/o/r/branches?page=2>; rel="prev"'
    }

    with patch.object(service, '_make_request', return_value=(mock_response, headers)):
        result = await service.get_paginated_branches('owner/repo', page=1, per_page=1)
        assert result.has_next_page is False
        assert len(result.branches) == 1
        assert result.branches[0].name == 'dev'


@pytest.mark.asyncio
async def test_search_branches_github_success_and_variables():
    service = GitHubService(token=SecretStr('t'))

    # Prepare a fake GraphQL response structure
    graphql_result = {
        'data': {
            'repository': {
                'refs': {
                    'nodes': [
                        {
                            'name': 'feature/bar',
                            'target': {
                                '__typename': 'Commit',
                                'oid': 'aaa111',
                                'committedDate': '2024-01-05T10:00:00Z',
                            },
                            'branchProtectionRule': {},  # indicates protected
                        },
                        {
                            'name': 'chore/update',
                            'target': {
                                '__typename': 'Tag',
                                'oid': 'should_be_ignored_for_commit',
                            },
                            'branchProtectionRule': None,
                        },
                    ]
                }
            }
        }
    }

    exec_mock = AsyncMock(return_value=graphql_result)
    with patch.object(service, 'execute_graphql_query', exec_mock) as mock_exec:
        branches = await service.search_branches('foo/bar', query='fe', per_page=999)

        # per_page should be clamped to <= 100 when passed to GraphQL variables
        args, kwargs = mock_exec.call_args
        _query = args[0]
        variables = args[1]
        assert variables['owner'] == 'foo'
        assert variables['name'] == 'bar'
        assert variables['query'] == 'fe'
        assert 1 <= variables['perPage'] <= 100

        assert len(branches) == 2
        b0, b1 = branches
        assert b0.name == 'feature/bar'
        assert b0.commit_sha == 'aaa111'
        assert b0.protected is True
        assert b0.last_push_date == '2024-01-05T10:00:00Z'

        # Non-commit target results in empty sha and no date
        assert b1.name == 'chore/update'
        assert b1.commit_sha == ''
        assert b1.last_push_date is None
        assert b1.protected is False


@pytest.mark.asyncio
async def test_search_branches_github_edge_cases():
    service = GitHubService(token=SecretStr('t'))

    # Empty query should return [] without issuing a GraphQL call
    branches = await service.search_branches('foo/bar', query='')
    assert branches == []

    # Invalid repository string should return [] without calling GraphQL
    exec_mock = AsyncMock()
    with patch.object(service, 'execute_graphql_query', exec_mock):
        branches = await service.search_branches('invalidrepo', query='q')
        assert branches == []
        exec_mock.assert_not_called()


@pytest.mark.asyncio
async def test_search_branches_github_graphql_error_returns_empty():
    service = GitHubService(token=SecretStr('t'))

    exec_mock = AsyncMock(side_effect=Exception('Boom'))
    with patch.object(service, 'execute_graphql_query', exec_mock):
        branches = await service.search_branches('foo/bar', query='q')
        assert branches == []
