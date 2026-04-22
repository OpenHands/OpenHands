from unittest.mock import AsyncMock

import pytest

from openhands.integrations.github.github_service import GitHubService
from openhands.integrations.service_types import TaskType, User


@pytest.mark.asyncio
async def test_get_suggested_tasks():
    # Mock responses
    mock_user = User(
        id='1',
        login='test-user',
        avatar_url='https://example.com/avatar.jpg',
        name='Test User',
    )

    # Mock GraphQL response
    mock_graphql_response = {
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
    service.execute_graphql_query = AsyncMock(return_value=mock_graphql_response)

    # Call the function
    tasks = await service.get_suggested_tasks()

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
