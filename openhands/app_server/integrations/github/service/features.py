from openhands.app_server.integrations.github.queries import (
    suggested_task_issue_graphql_query,
    suggested_task_pr_graphql_query,
)
from openhands.app_server.integrations.github.service.base import GitHubMixinBase
from openhands.app_server.integrations.service_types import (
    ProviderType,
    SuggestedTask,
    TaskType,
)
from openhands.core.logger import openhands_logger as logger


class GitHubFeaturesMixin(GitHubMixinBase):
    """
    Methods used for custom features in UI driven via GitHub integration
    """

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        """Get suggested tasks for the authenticated user across all repositories.

        Returns:
        - PRs authored by the user.
        - Issues assigned to the user.

        Note: Queries are split to avoid timeout issues.
        """
        # Get user info to use in queries
        user = await self.get_user()
        login = user.login
        tasks: list[SuggestedTask] = []
        variables = {'login': login}

        try:
            pr_response = await self.execute_graphql_query(
                suggested_task_pr_graphql_query, variables
            )
            pr_data = pr_response['data']['user']

            # Process pull requests
            for pr in pr_data['pullRequests']['nodes']:
                repo_name = pr['repository']['nameWithOwner']

                # Start with default task type
                task_type = TaskType.OPEN_PR

                # Check for specific states
                if pr['mergeable'] == 'CONFLICTING':
                    task_type = TaskType.MERGE_CONFLICTS
                elif (
                    pr['commits']['nodes']
                    and pr['commits']['nodes'][0]['commit']['statusCheckRollup']
                    and pr['commits']['nodes'][0]['commit']['statusCheckRollup'][
                        'state'
                    ]
                    == 'FAILURE'
                ):
                    task_type = TaskType.FAILING_CHECKS
                elif any(
                    review['state'] in ['CHANGES_REQUESTED', 'COMMENTED']
                    for review in pr['reviews']['nodes']
                ):
                    task_type = TaskType.UNRESOLVED_COMMENTS

                # Only add the task if it's not OPEN_PR
                if task_type != TaskType.OPEN_PR:
                    tasks.append(
                        SuggestedTask(
                            git_provider=ProviderType.GITHUB,
                            task_type=task_type,
                            repo=repo_name,
                            issue_number=pr['number'],
                            title=pr['title'],
                        )
                    )

        except Exception as e:
            logger.info(
                f'Error fetching suggested task for PRs: {e}',
                extra={
                    'signal': 'github_suggested_tasks',
                    'user_id': self.external_auth_id,
                },
            )

        try:
            # Execute issue query
            issue_response = await self.execute_graphql_query(
                suggested_task_issue_graphql_query, variables
            )
            issue_data = issue_response['data']['user']

            # Process issues
            for issue in issue_data['issues']['nodes']:
                repo_name = issue['repository']['nameWithOwner']
                tasks.append(
                    SuggestedTask(
                        git_provider=ProviderType.GITHUB,
                        task_type=TaskType.OPEN_ISSUE,
                        repo=repo_name,
                        issue_number=issue['number'],
                        title=issue['title'],
                    )
                )

            return tasks

        except Exception as e:
            logger.info(
                f'Error fetching suggested task for issues: {e}',
                extra={
                    'signal': 'github_suggested_tasks',
                    'user_id': self.external_auth_id,
                },
            )

        return tasks
