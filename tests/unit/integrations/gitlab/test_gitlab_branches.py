from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.integrations.gitlab.gitlab_service import GitLabService
from openhands.integrations.service_types import Branch, PaginatedBranchesResponse


@pytest.mark.asyncio
async def test_get_paginated_branches_gitlab_headers_and_parsing():
    service = GitLabService(token=SecretStr('t'))

    mock_response = [
        {
            'name': 'main',
            'commit': {'id': 'abc', 'committed_date': '2024-01-01T00:00:00Z'},
            'protected': True,
        },
        {
            'name': 'dev',
            'commit': {'id': 'def', 'committed_date': '2024-01-02T00:00:00Z'},
            'protected': False,
        },
    ]

    headers = {
        'X-Total': '42',
        'Link': '<https://gitlab.example.com/api/v4/projects/group%2Frepo/repository/branches?page=3&per_page=2>; rel="next"',  # indicates has next page
    }

    with patch.object(service, '_make_request', return_value=(mock_response, headers)):
        res = await service.get_paginated_branches('group/repo', page=2, per_page=2)

        assert isinstance(res, PaginatedBranchesResponse)
        assert res.has_next_page is True
        assert res.current_page == 2
        assert res.per_page == 2
        assert res.total_count == 42
        assert len(res.branches) == 2
        assert res.branches[0] == Branch(
            name='main',
            commit_sha='abc',
            protected=True,
            last_push_date='2024-01-01T00:00:00Z',
        )
        assert res.branches[1] == Branch(
            name='dev',
            commit_sha='def',
            protected=False,
            last_push_date='2024-01-02T00:00:00Z',
        )


@pytest.mark.asyncio
async def test_get_paginated_branches_gitlab_no_next_or_total():
    service = GitLabService(token=SecretStr('t'))

    mock_response = [
        {
            'name': 'fix',
            'commit': {'id': 'zzz', 'committed_date': '2024-01-03T00:00:00Z'},
            'protected': False,
        }
    ]

    headers = {}  # No pagination headers; should be has_next_page False

    with patch.object(service, '_make_request', return_value=(mock_response, headers)):
        res = await service.get_paginated_branches('group/repo', page=1, per_page=1)
        assert res.has_next_page is False
        assert res.total_count is None
        assert len(res.branches) == 1
        assert res.branches[0].name == 'fix'


@pytest.mark.asyncio
async def test_search_branches_gitlab_uses_search_param():
    service = GitLabService(token=SecretStr('t'))

    mock_response = [
        {
            'name': 'feat/new',
            'commit': {'id': '111', 'committed_date': '2024-01-04T00:00:00Z'},
            'protected': False,
        },
        {
            'name': 'feature/xyz',
            'commit': {'id': '222', 'committed_date': '2024-01-05T00:00:00Z'},
            'protected': True,
        },
    ]

    with patch.object(service, '_make_request', return_value=(mock_response, {})) as m:
        branches = await service.search_branches(
            'group/repo', query='feat', per_page=50
        )

        # Verify parameters
        args, kwargs = m.call_args
        url = args[0]
        params = args[1]
        assert 'repository/branches' in url
        assert params['per_page'] == '50'
        assert params['search'] == 'feat'

        assert len(branches) == 2
        assert branches[0] == Branch(
            name='feat/new',
            commit_sha='111',
            protected=False,
            last_push_date='2024-01-04T00:00:00Z',
        )
        assert branches[1] == Branch(
            name='feature/xyz',
            commit_sha='222',
            protected=True,
            last_push_date='2024-01-05T00:00:00Z',
        )
