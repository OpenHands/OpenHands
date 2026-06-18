from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.github.github_service import GitHubService
from openhands.app_server.integrations.service_types import (
    Branch,
    PaginatedBranchesResponse,
)


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
