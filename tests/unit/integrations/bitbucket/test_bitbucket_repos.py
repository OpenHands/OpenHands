"""Tests for Bitbucket repository service URL parsing."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.bitbucket.bitbucket_service import (
    BitBucketService,
)
from openhands.app_server.integrations.service_types import OwnerType, Repository
from openhands.app_server.integrations.service_types import (
    ProviderType as ServiceProviderType,
)
from openhands.app_server.types import AppMode


@pytest.fixture
def bitbucket_service():
    """Create a BitBucketService instance for testing."""
    return BitBucketService(token=SecretStr('test-token'))


@pytest.mark.asyncio
async def test_search_repositories_url_parsing_standard_url(bitbucket_service):
    """Test URL parsing with standard Bitbucket URL and verify correct workspace/repo extraction."""
    mock_repo = Repository(
        id='1',
        full_name='workspace/repo',
        name='repo',
        owner=OwnerType.USER,
        git_provider=ServiceProviderType.BITBUCKET,
        is_public=True,
        clone_url='https://bitbucket.org/workspace/repo.git',
        html_url='https://bitbucket.org/workspace/repo',
    )

    with patch.object(
        bitbucket_service,
        'get_repository_details_from_repo_name',
        return_value=mock_repo,
    ) as mock_get_repo:
        url = 'https://bitbucket.org/workspace/repo'
        repositories = await bitbucket_service.search_repositories(
            query=url,
            per_page=10,
            sort='updated',
            order='desc',
            public=True,
            app_mode=AppMode.OPENHANDS,
        )

        # Verify the correct workspace/repo combination was extracted and passed
        assert len(repositories) == 1
        assert repositories[0].full_name == 'workspace/repo'
        mock_get_repo.assert_called_once_with('workspace/repo')


@pytest.mark.asyncio
async def test_search_repositories_url_parsing_with_extra_path_segments(
    bitbucket_service,
):
    """Test URL parsing with additional path segments and verify correct workspace/repo extraction."""
    mock_repo = Repository(
        id='1',
        full_name='my-workspace/my-repo',
        name='my-repo',
        owner=OwnerType.USER,
        git_provider=ServiceProviderType.BITBUCKET,
        is_public=True,
        clone_url='https://bitbucket.org/my-workspace/my-repo.git',
        html_url='https://bitbucket.org/my-workspace/my-repo',
    )

    with patch.object(
        bitbucket_service,
        'get_repository_details_from_repo_name',
        return_value=mock_repo,
    ) as mock_get_repo:
        # Test complex URL with query params, fragments, and extra paths
        url = 'https://bitbucket.org/my-workspace/my-repo/src/feature-branch/src/main.py?at=feature-branch&fileviewer=file-view-default#lines-25'
        repositories = await bitbucket_service.search_repositories(
            query=url,
            per_page=10,
            sort='updated',
            order='desc',
            public=True,
            app_mode=AppMode.OPENHANDS,
        )

        # Verify the correct workspace/repo combination was extracted from complex URL
        assert len(repositories) == 1
        assert repositories[0].full_name == 'my-workspace/my-repo'
        mock_get_repo.assert_called_once_with('my-workspace/my-repo')


@pytest.mark.asyncio
async def test_search_repositories_url_parsing_invalid_url(bitbucket_service):
    """Test URL parsing with invalid URL returns empty results."""
    with patch.object(
        bitbucket_service, 'get_repository_details_from_repo_name'
    ) as mock_get_repo:
        url = 'not-a-valid-url'
        repositories = await bitbucket_service.search_repositories(
            query=url,
            per_page=10,
            sort='updated',
            order='desc',
            public=True,
            app_mode=AppMode.OPENHANDS,
        )

        # Should return empty list for invalid URL and not call API
        assert len(repositories) == 0
        mock_get_repo.assert_not_called()


@pytest.mark.asyncio
async def test_search_repositories_url_parsing_insufficient_path_segments(
    bitbucket_service,
):
    """Test URL parsing with insufficient path segments returns empty results."""
    with patch.object(
        bitbucket_service, 'get_repository_details_from_repo_name'
    ) as mock_get_repo:
        url = 'https://bitbucket.org/workspace'
        repositories = await bitbucket_service.search_repositories(
            query=url,
            per_page=10,
            sort='updated',
            order='desc',
            public=True,
            app_mode=AppMode.OPENHANDS,
        )

        # Should return empty list for insufficient path segments and not call API
        assert len(repositories) == 0
        mock_get_repo.assert_not_called()


@pytest.mark.parametrize(
    'raw_repo, expected_full_name, expected_main_branch',
    [
        pytest.param(
            {
                'uuid': '{abc}',
                'slug': 'empty-repo',
                'workspace': None,
                'mainbranch': None,
                'is_private': True,
            },
            '',
            None,
            id='workspace_and_mainbranch_both_null',
        ),
        pytest.param(
            {
                'uuid': '{abc}',
                'slug': 'empty-repo',
                'workspace': {'slug': 'ws'},
                'mainbranch': None,
                'is_private': True,
            },
            'ws/empty-repo',
            None,
            id='mainbranch_null_only',
        ),
        pytest.param(
            {
                'uuid': '{abc}',
                'slug': 'repo',
                'workspace': None,
                'mainbranch': {'name': 'main'},
                'is_private': True,
            },
            '',
            'main',
            id='workspace_null_only',
        ),
        pytest.param(
            {
                'uuid': '{abc}',
                'slug': 'repo',
                'workspace': {'slug': 'ws'},
                'mainbranch': {'name': 'main'},
                'is_private': False,
            },
            'ws/repo',
            'main',
            id='happy_path',
        ),
        pytest.param(
            {
                'uuid': '{abc}',
                'slug': 'repo',
                'workspace': [],
                'mainbranch': 'not-a-dict',
                'is_private': True,
            },
            '',
            None,
            id='workspace_and_mainbranch_non_dict_types',
        ),
    ],
)
def test_parse_repository_handles_null_nested_fields(
    bitbucket_service, raw_repo, expected_full_name, expected_main_branch
):
    """_parse_repository must not crash when Bitbucket returns null (or a
    non-dict) for `workspace` / `mainbranch`; downstream callers
    (`_get_cursorrules_url`, `_get_microagents_directory_url`) already expect
    `main_branch is None` on empty repos and raise `ResourceNotFoundError`
    themselves, so this extraction should degrade gracefully rather than
    crashing with AttributeError."""
    repository = bitbucket_service._parse_repository(raw_repo)

    assert repository.full_name == expected_full_name
    assert repository.main_branch == expected_main_branch
    assert repository.id == '{abc}'
