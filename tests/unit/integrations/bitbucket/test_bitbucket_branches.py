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


@pytest.mark.asyncio
async def test_search_branches_bitbucket_filters_by_name_contains():
    service = BitBucketService(token=SecretStr('t'))

    mock_response = {
        'values': [
            {
                'name': 'bugfix/issue-1',
                'target': {'hash': 'hhh', 'date': '2024-01-10T10:00:00Z'},
            }
        ]
    }

    with patch.object(service, '_make_request', return_value=(mock_response, {})) as m:
        branches = await service.search_branches('w/r', query='bugfix', per_page=15)

        args, kwargs = m.call_args
        url = args[0]
        params = args[1]
        assert 'refs/branches' in url
        assert params['pagelen'] == 15
        assert params['q'] == 'name~"bugfix"'
        assert params['sort'] == '-target.date'

        assert branches == [
            Branch(
                name='bugfix/issue-1',
                commit_sha='hhh',
                protected=False,
                last_push_date='2024-01-10T10:00:00Z',
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'raw_target, expected_sha, expected_date',
    [
        pytest.param(None, '', None, id='target_null'),
        pytest.param([], '', None, id='target_non_dict'),
        pytest.param({}, '', None, id='target_empty_dict'),
        pytest.param(
            {'hash': 'abc', 'date': '2024-01-01T00:00:00Z'},
            'abc',
            '2024-01-01T00:00:00Z',
            id='happy_path',
        ),
    ],
)
async def test_get_branches_handles_null_target(
    raw_target, expected_sha, expected_date
):
    """Bitbucket cloud branches API can return ``target: null`` for branches
    whose tip commit was rewritten by a corporate proxy or in edge-case repo
    states. The original ``branch.get('target', {}).get(...)`` chain only
    short-circuited the absent-key case and crashed with AttributeError on
    explicit nulls; the new ``_target`` helper degrades to an empty dict so
    the fallback values defined in each ``.get`` call (``''`` for hash,
    ``None`` for date) actually fire."""
    service = BitBucketService(token=SecretStr('t'))
    raw_branches = [{'name': 'b1', 'target': raw_target}]

    with patch.object(service, '_fetch_paginated_data', return_value=raw_branches):
        branches = await service.get_branches('w/r')

    assert len(branches) == 1
    assert branches[0].name == 'b1'
    assert branches[0].commit_sha == expected_sha
    assert branches[0].last_push_date == expected_date
