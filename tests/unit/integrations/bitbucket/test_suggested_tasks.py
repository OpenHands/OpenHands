from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from openhands.integrations.bitbucket.bitbucket_service import BitBucketService
from openhands.integrations.service_types import (
    OwnerType,
    ProviderType,
    Repository,
    TaskType,
)
from openhands.server.types import AppMode


@pytest.fixture
def bitbucket_service() -> BitBucketService:
    """Create a BitBucketService instance for testing."""
    return BitBucketService(token=SecretStr('test-token'))


def _make_repo(full_name: str | None) -> Repository:
    """Helper to construct a minimal Repository instance for tests."""
    return Repository(
        id='repo-id',
        full_name=full_name,
        name=full_name.split('/', 1)[1] if full_name else '',
        owner=OwnerType.USER,
        git_provider=ProviderType.BITBUCKET,
        is_public=True,
        clone_url=f'https://bitbucket.org/{full_name}.git' if full_name else '',
        html_url=f'https://bitbucket.org/{full_name}' if full_name else '',
    )


@pytest.mark.asyncio
async def test_get_suggested_tasks_returns_open_issues_across_repositories(
    bitbucket_service: BitBucketService,
) -> None:
    """Open Bitbucket issues should be converted into SuggestedTask instances."""
    # Arrange: two repositories the user can access
    repos: list[Repository] = [
        _make_repo('workspace-one/repo-one'),
        _make_repo('workspace-two/repo-two'),
    ]

    # Patch get_all_repositories to avoid real network calls
    bitbucket_service.get_all_repositories = AsyncMock(
        return_value=repos,
    )

    # Prepare fake issue data for each repository
    repo_one_issues = [
        {'id': 1, 'title': 'First issue (new)', 'state': 'new'},
        {'id': 2, 'title': 'Second issue (open)', 'state': 'open'},
        {'id': 3, 'title': 'Closed issue', 'state': 'resolved'},
    ]
    repo_two_issues = [
        {'id': 10, 'title': 'Another open issue', 'state': 'open'},
    ]

    async def fake_fetch_paginated_data(url: str, params: dict, max_items: int):
        # Route requests based on which repository is being queried
        if 'workspace-one/repo-one' in url:
            return repo_one_issues[:max_items]
        if 'workspace-two/repo-two' in url:
            return repo_two_issues[:max_items]
        return []

    bitbucket_service._fetch_paginated_data = AsyncMock(
        side_effect=fake_fetch_paginated_data
    )

    # Act
    tasks = await bitbucket_service.get_suggested_tasks()

    # Assert
    # Only open/new issues should be surfaced as SuggestedTask instances
    assert len(tasks) == 3

    # All tasks should be OPEN_ISSUE tasks for the Bitbucket provider
    assert all(task.git_provider is ProviderType.BITBUCKET for task in tasks)
    assert all(task.task_type is TaskType.OPEN_ISSUE for task in tasks)

    # Verify mapping of repository and issue identifiers
    task_keys = {(t.repo, t.issue_number, t.title) for t in tasks}
    assert (
        'workspace-one/repo-one',
        1,
        'First issue (new)',
    ) in task_keys
    assert (
        'workspace-one/repo-one',
        2,
        'Second issue (open)',
    ) in task_keys
    assert (
        'workspace-two/repo-two',
        10,
        'Another open issue',
    ) in task_keys


@pytest.mark.asyncio
async def test_get_suggested_tasks_skips_repositories_without_full_name(
    bitbucket_service: BitBucketService,
) -> None:
    """Repositories missing a full_name should be ignored."""
    repos: list[Repository] = [
        _make_repo('workspace/repo'),
        _make_repo(None),
    ]

    bitbucket_service.get_all_repositories = AsyncMock(return_value=repos)

    async def fake_fetch_paginated_data(url: str, params: dict, max_items: int):
        # Only one repository should result in a call for issues
        assert 'workspace/repo' in url
        return [{'id': 1, 'title': 'Valid issue', 'state': 'open'}]

    bitbucket_service._fetch_paginated_data = AsyncMock(
        side_effect=fake_fetch_paginated_data
    )

    tasks = await bitbucket_service.get_suggested_tasks()

    assert len(tasks) == 1
    assert tasks[0].repo == 'workspace/repo'
    assert tasks[0].issue_number == 1
    assert tasks[0].title == 'Valid issue'


@pytest.mark.asyncio
async def test_get_suggested_tasks_handles_repository_listing_failure(
    bitbucket_service: BitBucketService,
) -> None:
    """If listing repositories fails, the method should return an empty list."""

    async def failing_get_all_repositories(sort: str, app_mode: AppMode):
        raise RuntimeError('simulated failure')

    bitbucket_service.get_all_repositories = failing_get_all_repositories  # type: ignore[assignment]

    tasks = await bitbucket_service.get_suggested_tasks()

    assert tasks == []


@pytest.mark.asyncio
async def test_get_suggested_tasks_skips_repositories_with_issue_fetch_failures(
    bitbucket_service: BitBucketService,
) -> None:
    """Errors fetching issues for a particular repository should not abort the whole call."""
    repos: list[Repository] = [
        _make_repo('workspace/failing-repo'),
        _make_repo('workspace/working-repo'),
    ]

    bitbucket_service.get_all_repositories = AsyncMock(return_value=repos)

    async def fake_fetch_paginated_data(url: str, params: dict, max_items: int):
        if 'failing-repo' in url:
            raise RuntimeError('simulated issue list failure')
        return [{'id': 42, 'title': 'Working issue', 'state': 'open'}]

    bitbucket_service._fetch_paginated_data = AsyncMock(
        side_effect=fake_fetch_paginated_data
    )

    tasks = await bitbucket_service.get_suggested_tasks()

    assert len(tasks) == 1
    task = tasks[0]
    assert task.repo == 'workspace/working-repo'
    assert task.issue_number == 42
    assert task.title == 'Working issue'
