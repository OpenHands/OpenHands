"""Tests for BitbucketDCBranchesMixin: get_paginated_branches, search_branches, get_branches."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)
from openhands.integrations.service_types import Branch, PaginatedBranchesResponse


def make_service():
    return BitbucketDCService(token=SecretStr('tok'), base_domain='host.example.com')


def _dc_branch(display_id='main', commit='abc123'):
    return {'displayId': display_id, 'latestCommit': commit}


# ── get_paginated_branches ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_paginated_branches_parses_display_id_and_commit():
    svc = make_service()
    mock_response = {
        'values': [
            _dc_branch('main', 'abc'),
            _dc_branch('feature/x', 'def'),
        ],
        'isLastPage': True,
        'size': 2,
    }

    with patch.object(
        svc, '_make_request', return_value=(mock_response, {})
    ) as mock_req:
        res = await svc.get_paginated_branches('PROJ/myrepo', page=1, per_page=30)

    # Verify the URL uses the DC format
    call_url = mock_req.call_args[0][0]
    assert '/projects/PROJ/repos/myrepo/branches' in call_url

    assert isinstance(res, PaginatedBranchesResponse)
    assert res.branches == [
        Branch(name='main', commit_sha='abc', protected=False, last_push_date=None),
        Branch(
            name='feature/x', commit_sha='def', protected=False, last_push_date=None
        ),
    ]


@pytest.mark.asyncio
async def test_get_paginated_branches_has_next_page():
    svc = make_service()
    mock_response = {
        'values': [_dc_branch()],
        'isLastPage': False,
        'nextPageStart': 30,
        'size': 100,
    }

    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        res = await svc.get_paginated_branches('PROJ/myrepo', page=1, per_page=30)

    assert res.has_next_page is True


@pytest.mark.asyncio
async def test_get_paginated_branches_last_page():
    svc = make_service()
    mock_response = {
        'values': [_dc_branch()],
        'isLastPage': True,
        'size': 1,
    }

    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        res = await svc.get_paginated_branches('PROJ/myrepo', page=1, per_page=30)

    assert res.has_next_page is False


@pytest.mark.asyncio
async def test_get_paginated_branches_total_count():
    svc = make_service()
    mock_response = {
        'values': [_dc_branch()],
        'isLastPage': True,
        'size': 42,
    }

    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        res = await svc.get_paginated_branches('PROJ/myrepo', page=1, per_page=30)

    assert res.total_count == 42


# ── search_branches ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_branches_uses_filter_text():
    svc = make_service()
    mock_response = {'values': [_dc_branch('feature/my-thing', 'sha1')]}

    with patch.object(
        svc, '_make_request', return_value=(mock_response, {})
    ) as mock_req:
        branches = await svc.search_branches(
            'PROJ/myrepo', query='my-thing', per_page=15
        )

    call_url, call_params = mock_req.call_args[0]
    assert 'filterText' in call_params
    assert call_params['filterText'] == 'my-thing'
    assert 'q' not in call_params
    assert len(branches) == 1
    assert branches[0].name == 'feature/my-thing'


# ── get_branches (all pages via _fetch_paginated_data) ───────────────────────


@pytest.mark.asyncio
async def test_get_branches_returns_all_pages():
    svc = make_service()

    async def fake_fetch(url, params, max_items):
        return [_dc_branch('main', 'a'), _dc_branch('dev', 'b')]

    with patch.object(svc, '_fetch_paginated_data', side_effect=fake_fetch):
        branches = await svc.get_branches('PROJ/myrepo')

    assert len(branches) == 2
    assert branches[0].name == 'main'
    assert branches[1].name == 'dev'
