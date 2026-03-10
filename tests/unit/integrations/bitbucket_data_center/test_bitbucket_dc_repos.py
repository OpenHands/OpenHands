"""Tests for BitbucketDCReposMixin: URL parsing, get_paginated_repos, get_all_repositories."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)
from openhands.server.types import AppMode


def make_service():
    return BitbucketDCService(token=SecretStr('tok'), base_domain='host.example.com')


def _repo_dict(key='PROJ', slug='myrepo', name='My Repository'):
    return {
        'id': 1,
        'slug': slug,
        'name': name,
        'project': {'key': key},
        'public': False,
    }


# ── search_repositories URL parsing ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_repositories_projects_url():
    svc = make_service()
    query = 'https://host.example.com/projects/PROJ/repos/myrepo'

    mock_repo_data = _repo_dict('PROJ', 'myrepo')
    mock_response = {'id': 1, **mock_repo_data}
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_make_request',
        side_effect=[
            (mock_response, {}),
            (mock_default_branch, {}),
        ],
    ):
        repos = await svc.search_repositories(
            query, 25, 'name', 'asc', True, AppMode.SAAS
        )

    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/myrepo'


@pytest.mark.asyncio
async def test_search_repositories_projects_url_with_extra_segments():
    svc = make_service()
    # URL with extra segments after repo name
    query = 'https://host.example.com/projects/PROJ/repos/myrepo/browse/src/main.py'

    mock_repo_data = _repo_dict('PROJ', 'myrepo')
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_make_request',
        side_effect=[
            (mock_repo_data, {}),
            (mock_default_branch, {}),
        ],
    ):
        repos = await svc.search_repositories(
            query, 25, 'name', 'asc', True, AppMode.SAAS
        )

    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/myrepo'


@pytest.mark.asyncio
async def test_search_repositories_invalid_url():
    svc = make_service()
    with patch.object(svc, '_make_request') as mock_req:
        repos = await svc.search_repositories(
            'not-a-valid-url', 25, 'name', 'asc', True, AppMode.SAAS
        )
    assert repos == []
    mock_req.assert_not_called()


@pytest.mark.asyncio
async def test_search_repositories_insufficient_path_segments():
    svc = make_service()
    # URL with only one path segment (just a project, no repo)
    with patch.object(svc, '_make_request') as mock_req:
        repos = await svc.search_repositories(
            'https://host.example.com/projects/PROJ',
            25,
            'name',
            'asc',
            True,
            AppMode.SAAS,
        )
    assert repos == []
    mock_req.assert_not_called()


@pytest.mark.asyncio
async def test_search_repositories_slash_query():
    svc = make_service()
    query = 'PROJ/myrepo'

    mock_repo = _repo_dict('PROJ', slug='myrepo', name='My Repository')
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_fetch_paginated_data',
        new=AsyncMock(return_value=[mock_repo]),
    ) as mock_fetch:
        with patch.object(
            svc,
            '_make_request',
            new=AsyncMock(return_value=(mock_default_branch, {})),
        ):
            repos = await svc.search_repositories(
                query, 25, 'name', 'asc', False, AppMode.SAAS
            )

    mock_fetch.assert_called_once_with(
        'https://host.example.com/rest/api/1.0/projects/PROJ/repos',
        {'limit': 25},
        1000,
    )
    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/myrepo'


@pytest.mark.asyncio
async def test_search_repositories_slash_query_filters_by_name():
    """Filter matches the human-readable name when slug doesn't match."""
    svc = make_service()
    matching = _repo_dict('PROJ', slug='proj-alpha', name='My Repository')
    non_matching = _repo_dict('PROJ', slug='proj-beta', name='Other Repo')
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_fetch_paginated_data',
        new=AsyncMock(return_value=[matching, non_matching]),
    ):
        with patch.object(
            svc,
            '_make_request',
            new=AsyncMock(return_value=(mock_default_branch, {})),
        ):
            repos = await svc.search_repositories(
                'PROJ/my repository', 25, 'name', 'asc', False, AppMode.SAAS
            )

    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/proj-alpha'


@pytest.mark.asyncio
async def test_search_repositories_slash_query_filters_by_slug():
    """Filter matches the slug when the human-readable name doesn't match."""
    svc = make_service()
    matching = _repo_dict('PROJ', slug='my-repo', name='My Repository')
    non_matching = _repo_dict('PROJ', slug='other-repo', name='Other Repository')
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_fetch_paginated_data',
        new=AsyncMock(return_value=[matching, non_matching]),
    ):
        with patch.object(
            svc,
            '_make_request',
            new=AsyncMock(return_value=(mock_default_branch, {})),
        ):
            repos = await svc.search_repositories(
                'PROJ/my-repo', 25, 'name', 'asc', False, AppMode.SAAS
            )

    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/my-repo'


# ── get_paginated_repos ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_paginated_repos_parses_values():
    svc = make_service()
    mock_response = {
        'values': [_repo_dict()],
        'isLastPage': True,
    }
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_make_request',
        side_effect=[(mock_response, {}), (mock_default_branch, {})],
    ):
        repos = await svc.get_paginated_repos(1, 25, 'name', 'PROJ')

    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/myrepo'
    assert repos[0].link_header == ''


@pytest.mark.asyncio
async def test_get_paginated_repos_has_next_page():
    svc = make_service()
    mock_response = {
        'values': [_repo_dict()],
        'isLastPage': False,
        'nextPageStart': 25,
    }
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_make_request',
        side_effect=[(mock_response, {}), (mock_default_branch, {})],
    ):
        repos = await svc.get_paginated_repos(1, 25, 'name', 'PROJ')

    assert len(repos) == 1
    assert 'rel="next"' in repos[0].link_header


@pytest.mark.asyncio
async def test_get_paginated_repos_last_page():
    svc = make_service()
    mock_response = {
        'values': [_repo_dict()],
        'isLastPage': True,
    }
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_make_request',
        side_effect=[(mock_response, {}), (mock_default_branch, {})],
    ):
        repos = await svc.get_paginated_repos(1, 25, 'name', 'PROJ')

    assert len(repos) == 1
    assert repos[0].link_header == ''


@pytest.mark.asyncio
async def test_get_paginated_repos_filters_by_slug():
    """Query matches slug when name doesn't contain the search term."""
    svc = make_service()
    mock_response = {
        'values': [
            _repo_dict('PROJ', slug='my-repo', name='My Repository'),
            _repo_dict('PROJ', slug='other-repo', name='Other Repository'),
        ],
        'isLastPage': True,
    }
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_make_request',
        side_effect=[(mock_response, {}), (mock_default_branch, {})],
    ):
        repos = await svc.get_paginated_repos(1, 25, 'name', 'PROJ', query='my-repo')

    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/my-repo'


@pytest.mark.asyncio
async def test_get_paginated_repos_filters_by_name():
    """Query matches human-readable name when slug doesn't contain the search term."""
    svc = make_service()
    mock_response = {
        'values': [
            _repo_dict('PROJ', slug='proj-alpha', name='My Repository'),
            _repo_dict('PROJ', slug='proj-beta', name='Other Repository'),
        ],
        'isLastPage': True,
    }
    mock_default_branch = {'displayId': 'main'}

    with patch.object(
        svc,
        '_make_request',
        side_effect=[(mock_response, {}), (mock_default_branch, {})],
    ):
        repos = await svc.get_paginated_repos(
            1, 25, 'name', 'PROJ', query='my repository'
        )

    assert len(repos) == 1
    assert repos[0].full_name == 'PROJ/proj-alpha'


# ── get_all_repositories ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_all_repositories_iterates_projects():
    svc = make_service()

    async def fake_fetch(url, params, max_items):
        if '/projects' in url and '/repos' not in url:
            return [{'key': 'PROJ1'}, {'key': 'PROJ2'}]
        if 'PROJ1' in url:
            return [_repo_dict('PROJ1', 'repo1')]
        if 'PROJ2' in url:
            return [_repo_dict('PROJ2', 'repo2')]
        return []

    mock_default_branch = {'displayId': 'main'}
    with patch.object(svc, '_fetch_paginated_data', side_effect=fake_fetch):
        with patch.object(svc, '_make_request', return_value=(mock_default_branch, {})):
            repos = await svc.get_all_repositories('name', AppMode.SAAS)

    full_names = {r.full_name for r in repos}
    assert 'PROJ1/repo1' in full_names
    assert 'PROJ2/repo2' in full_names


# ── get_installations ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_installations_returns_project_keys():
    svc = make_service()

    async def fake_fetch(url, params, max_items):
        return [{'key': 'PROJ1'}, {'key': 'PROJ2'}, {'name': 'no-key'}]

    with patch.object(svc, '_fetch_paginated_data', side_effect=fake_fetch):
        keys = await svc.get_installations()

    assert keys == ['PROJ1', 'PROJ2']


# ── helper ────────────────────────────────────────────────────────────────────


async def _make_parsed_repo(svc, repo_dict):
    """Helper to create a parsed Repository from a repo dict (with mocked default branch)."""
    with patch.object(svc, '_make_request', return_value=({'displayId': 'main'}, {})):
        return await svc._parse_repository(repo_dict)
