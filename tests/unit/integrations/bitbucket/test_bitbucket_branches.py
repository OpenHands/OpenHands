from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.bitbucket.bitbucket_service import (
    BitBucketService,
)
from openhands.app_server.integrations.service_types import (
    Branch,
    PaginatedBranchesResponse,
)


@pytest.mark.asyncio
async def test_get_paginated_branches_bitbucket_parsing_and_pagination():
    service = BitBucketService(token=SecretStr('t'))

    mock_response = {
        'values': [
            {
                'name': 'main',
                'target': {'hash': 'abc', 'date': '2024-01-01T00:00:00Z'},
            },
            {
                'name': 'feature/x',
                'target': {'hash': 'def', 'date': '2024-01-02T00:00:00Z'},
            },
        ],
        'next': 'https://api.bitbucket.org/2.0/repositories/w/r/refs/branches?page=3',
        'size': 123,
    }

    with patch.object(service, '_make_request', return_value=(mock_response, {})):
        res = await service.get_paginated_branches('w/r', page=2, per_page=2)

        assert isinstance(res, PaginatedBranchesResponse)
        assert res.has_next_page is True
        assert res.current_page == 2
        assert res.per_page == 2
        assert res.total_count == 123
        assert res.branches == [
            Branch(
                name='main',
                commit_sha='abc',
                protected=False,
                last_push_date='2024-01-01T00:00:00Z',
            ),
            Branch(
                name='feature/x',
                commit_sha='def',
                protected=False,
                last_push_date='2024-01-02T00:00:00Z',
            ),
        ]
