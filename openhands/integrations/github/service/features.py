import base64

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.queries import (
    suggested_task_issue_graphql_query,
    suggested_task_pr_graphql_query,
)
from openhands.integrations.github.service.base import GitHubMixinBase
from openhands.integrations.service_types import (
    MicroagentContentResponse,
    ProviderType,
    SuggestedTask,
    TaskType,
)


class GitHubFeaturesMixin(GitHubMixinBase):
    """Methods used for custom features in UI driven via GitHub integration
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

    """
    Methods specifically for microagent management page
    """

    async def _get_cursorrules_url(self, repository: str) -> str:
        """Get the URL for checking .cursorrules file."""
        return f'{self.BASE_URL}/repos/{repository}/contents/.cursorrules'

    async def _get_microagents_directory_url(
        self, repository: str, microagents_path: str
    ) -> str:
        """Get the URL for checking microagents directory."""
        return f'{self.BASE_URL}/repos/{repository}/contents/{microagents_path}'

    def _is_valid_microagent_file(self, item: dict) -> bool:
        """Check if an item represents a valid microagent file."""
        return (
            item['type'] == 'file'
            and item['name'].endswith('.md')
            and item['name'] != 'README.md'
        )

    def _get_file_name_from_item(self, item: dict) -> str:
        """Extract file name from directory item."""
        return item['name']

    def _get_file_path_from_item(self, item: dict, microagents_path: str) -> str:
        """Extract file path from directory item."""
        return f'{microagents_path}/{item["name"]}'

    def _get_microagents_directory_params(self, microagents_path: str) -> dict | None:
        """Get parameters for the microagents directory request. Return None if no parameters needed."""
        return None

    async def get_microagent_content(
        self, repository: str, file_path: str
    ) -> MicroagentContentResponse:
        """Fetch individual file content from GitHub repository.

        Args:
            repository: Repository name in format 'owner/repo'
            file_path: Path to the file within the repository

        Returns:
            MicroagentContentResponse with parsed content and triggers

        Raises:
            RuntimeError: If file cannot be fetched or doesn't exist
        """
        file_url = f'{self.BASE_URL}/repos/{repository}/contents/{file_path}'

        file_data, _ = await self._make_request(file_url)
        file_content = base64.b64decode(file_data['content']).decode('utf-8')

        # Parse the content to extract triggers from frontmatter
        return self._parse_microagent_content(file_content, file_path)
