from unittest.mock import AsyncMock, call

import pytest

from openhands.app_server.integrations.github.github_service import GitHubService
from openhands.app_server.integrations.github.queries import (
    suggested_task_issue_graphql_query,
    suggested_task_pr_graphql_query,
)
from openhands.app_server.integrations.service_types import TaskType, User


@pytest.mark.asyncio
async def test_get_suggested_tasks():
    # Mock responses
    mock_user = User(
        id='1',
        login='test-user',
        avatar_url='https://example.com/avatar.jpg',
        name='Test User',
    )

    # Mock PR GraphQL response
    mock_pr_graphql_response = {
        'data': {
            'user': {
                'pullRequests': {
                    'nodes': [
                        {
                            'number': 1,
                            'title': 'PR with conflicts',
                            'repository': {'nameWithOwner': 'test-org/repo-1'},
                            'mergeable': 'CONFLICTING',
                            'commits': {
                                'nodes': [{'commit': {'statusCheckRollup': None}}]
                            },
                            'reviews': {'nodes': []},
                        },
                        {
                            'number': 2,
                            'title': 'PR with failing checks',
                            'repository': {'nameWithOwner': 'test-org/repo-1'},
                            'mergeable': 'MERGEABLE',
                            'commits': {
                                'nodes': [
                                    {
                                        'commit': {
                                            'statusCheckRollup': {'state': 'FAILURE'}
                                        }
                                    }
                                ]
                            },
                            'reviews': {'nodes': []},
                        },
                        {
                            'number': 4,
                            'title': 'PR with comments',
                            'repository': {'nameWithOwner': 'test-user/repo-2'},
                            'mergeable': 'MERGEABLE',
                            'commits': {
                                'nodes': [
                                    {
                                        'commit': {
                                            'statusCheckRollup': {'state': 'SUCCESS'}
                                        }
                                    }
                                ]
                            },
                            'reviews': {'nodes': [{'state': 'CHANGES_REQUESTED'}]},
                        },
                    ]
                },
            }
        }
    }

    # Mock issue GraphQL response
    mock_issue_graphql_response = {
        'data': {
            'user': {
                'issues': {
                    'nodes': [
                        {
                            'number': 3,
                            'title': 'Assigned issue 1',
                            'repository': {'nameWithOwner': 'test-org/repo-1'},
                        },
                        {
                            'number': 5,
                            'title': 'Assigned issue 2',
                            'repository': {'nameWithOwner': 'test-user/repo-2'},
                        },
                    ]
                },
            }
        }
    }

    # Create service instance with mocked methods
    service = GitHubService()
    service.get_user = AsyncMock(return_value=mock_user)
    service.execute_graphql_query = AsyncMock(
        side_effect=[mock_pr_graphql_response, mock_issue_graphql_response]
    )

    # Call the function
    tasks = await service.get_suggested_tasks()

    # Verify both GraphQL queries were called
    assert service.execute_graphql_query.call_count == 2
    expected_calls = [
        call(suggested_task_pr_graphql_query, {'login': 'test-user'}),
        call(suggested_task_issue_graphql_query, {'login': 'test-user'}),
    ]
    service.execute_graphql_query.assert_has_calls(expected_calls)

    # Verify the results
    assert len(tasks) == 5  # Should have 5 tasks total

    # Verify each task type is present
    task_types = [task.task_type for task in tasks]
    assert TaskType.MERGE_CONFLICTS in task_types
    assert TaskType.FAILING_CHECKS in task_types
    assert TaskType.UNRESOLVED_COMMENTS in task_types
    assert TaskType.OPEN_ISSUE in task_types
    assert (
        len([t for t in task_types if t == TaskType.OPEN_ISSUE]) == 2
    )  # Should have 2 open issues

    # Verify repositories are correct
    repos = {task.repo for task in tasks}
    assert 'test-org/repo-1' in repos
    assert 'test-user/repo-2' in repos

    # Verify specific tasks
    conflict_pr = next(t for t in tasks if t.task_type == TaskType.MERGE_CONFLICTS)
    assert conflict_pr.issue_number == 1
    assert conflict_pr.title == 'PR with conflicts'

    failing_pr = next(t for t in tasks if t.task_type == TaskType.FAILING_CHECKS)
    assert failing_pr.issue_number == 2
    assert failing_pr.title == 'PR with failing checks'

    commented_pr = next(t for t in tasks if t.task_type == TaskType.UNRESOLVED_COMMENTS)
    assert commented_pr.issue_number == 4
    assert commented_pr.title == 'PR with comments'


@pytest.mark.asyncio
async def test_get_suggested_tasks_pr_query_fails():
    """Test that issues are still returned when PR query fails."""
    mock_user = User(
        id='1',
        login='test-user',
        avatar_url='https://example.com/avatar.jpg',
        name='Test User',
    )

    # Mock issue response only
    mock_issue_graphql_response = {
        'data': {
            'user': {
                'issues': {
                    'nodes': [
                        {
                            'number': 1,
                            'title': 'Assigned issue',
                            'repository': {'nameWithOwner': 'test-org/repo'},
                        },
                    ]
                },
            }
        }
    }

    service = GitHubService()
    service.get_user = AsyncMock(return_value=mock_user)
    service.execute_graphql_query = AsyncMock(
        side_effect=[
            Exception('PR query failed'),  # PR query fails
            mock_issue_graphql_response,  # Issue query succeeds
        ]
    )

    # Call the function - should not raise despite PR query failure
    tasks = await service.get_suggested_tasks()

    # Verify we still get the issue task
    assert len(tasks) == 1
    assert tasks[0].task_type == TaskType.OPEN_ISSUE
    assert tasks[0].issue_number == 1


@pytest.mark.asyncio
async def test_get_suggested_tasks_issue_query_fails():
    """Test that PRs are still returned when issue query fails."""
    mock_user = User(
        id='1',
        login='test-user',
        avatar_url='https://example.com/avatar.jpg',
        name='Test User',
    )

    # Mock PR response only
    mock_pr_graphql_response = {
        'data': {
            'user': {
                'pullRequests': {
                    'nodes': [
                        {
                            'number': 1,
                            'title': 'PR with conflicts',
                            'repository': {'nameWithOwner': 'test-org/repo'},
                            'mergeable': 'CONFLICTING',
                            'commits': {
                                'nodes': [{'commit': {'statusCheckRollup': None}}]
                            },
                            'reviews': {'nodes': []},
                        },
                    ]
                },
            }
        }
    }

    service = GitHubService()
    service.get_user = AsyncMock(return_value=mock_user)
    service.execute_graphql_query = AsyncMock(
        side_effect=[
            mock_pr_graphql_response,  # PR query succeeds
            Exception('Issue query failed'),  # Issue query fails
        ]
    )

    # Call the function - should not raise despite issue query failure
    tasks = await service.get_suggested_tasks()

    # Verify we still get the PR task
    assert len(tasks) == 1
    assert tasks[0].task_type == TaskType.MERGE_CONFLICTS
    assert tasks[0].issue_number == 1
