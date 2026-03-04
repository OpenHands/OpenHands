from openhands.integrations.gitlab.service.base import GitLabMixinBase
from openhands.integrations.service_types import (
    MicroagentContentResponse,
    ProviderType,
    RequestMethod,
    SuggestedTask,
    TaskType,
)


class GitLabFeaturesMixin(GitLabMixinBase):
    """Methods used for custom features in UI driven via GitLab integration
    """

    async def _get_cursorrules_url(self, repository: str) -> str:
        """Get the URL for checking .cursorrules file."""
        project_id = self._extract_project_id(repository)
        return (
            f'{self.BASE_URL}/projects/{project_id}/repository/files/.cursorrules/raw'
        )

    async def _get_microagents_directory_url(
        self, repository: str, microagents_path: str
    ) -> str:
        """Get the URL for checking microagents directory."""
        project_id = self._extract_project_id(repository)
        return f'{self.BASE_URL}/projects/{project_id}/repository/tree'

    def _get_microagents_directory_params(self, microagents_path: str) -> dict:
        """Get parameters for the microagents directory request."""
        return {'path': microagents_path, 'recursive': 'true'}

    def _is_valid_microagent_file(self, item: dict) -> bool:
        """Check if an item represents a valid microagent file."""
        return (
            item['type'] == 'blob'
            and item['name'].endswith('.md')
            and item['name'] != 'README.md'
        )

    def _get_file_name_from_item(self, item: dict) -> str:
        """Extract file name from directory item."""
        return item['name']

    def _get_file_path_from_item(self, item: dict, microagents_path: str) -> str:
        """Extract file path from directory item."""
        return item['path']

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        """Get suggested tasks for the authenticated user across all repositories.

        Returns:
        - Merge requests authored by the user.
        - Issues assigned to the user.
        """
        # Get user info to use in queries
        user = await self.get_user()
        username = user.login

        # GraphQL query to get merge requests
        query = """
        query GetUserTasks {
          currentUser {
            authoredMergeRequests(state: opened, sort: UPDATED_DESC, first: 100) {
              nodes {
                id
                iid
                title
                project {
                  fullPath
                }
                conflicts
                mergeStatus
                pipelines(first: 1) {
                  nodes {
                    status
                  }
                }
                discussions(first: 100) {
                  nodes {
                    notes {
                      nodes {
                        resolvable
                        resolved
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """

        try:
            tasks: list[SuggestedTask] = []

            # Get merge requests using GraphQL
            response = await self.execute_graphql_query(query)
            data = response.get('currentUser', {})

            # Process merge requests
            merge_requests = data.get('authoredMergeRequests', {}).get('nodes', [])
            for mr in merge_requests:
                repo_name = mr.get('project', {}).get('fullPath', '')
                mr_number = mr.get('iid')
                title = mr.get('title', '')

                # Start with default task type
                task_type = TaskType.OPEN_PR

                # Check for specific states
                if mr.get('conflicts'):
                    task_type = TaskType.MERGE_CONFLICTS
                elif (
                    mr.get('pipelines', {}).get('nodes', [])
                    and mr.get('pipelines', {}).get('nodes', [])[0].get('status')
                    == 'FAILED'
                ):
                    task_type = TaskType.FAILING_CHECKS
                else:
                    # Check for unresolved comments
                    has_unresolved_comments = False
                    for discussion in mr.get('discussions', {}).get('nodes', []):
                        for note in discussion.get('notes', {}).get('nodes', []):
                            if note.get('resolvable') and not note.get('resolved'):
                                has_unresolved_comments = True
                                break
                        if has_unresolved_comments:
                            break

                    if has_unresolved_comments:
                        task_type = TaskType.UNRESOLVED_COMMENTS

                # Only add the task if it's not OPEN_PR
                if task_type != TaskType.OPEN_PR:
                    tasks.append(
                        SuggestedTask(
                            git_provider=ProviderType.GITLAB,
                            task_type=task_type,
                            repo=repo_name,
                            issue_number=mr_number,
                            title=title,
                        )
                    )

            # Get assigned issues using REST API
            url = f'{self.BASE_URL}/issues'
            params = {
                'assignee_username': username,
                'state': 'opened',
                'scope': 'assigned_to_me',
            }

            issues_response, _ = await self._make_request(
                method=RequestMethod.GET, url=url, params=params
            )

            # Process issues
            for issue in issues_response:
                repo_name = (
                    issue.get('references', {}).get('full', '').split('#')[0].strip()
                )
                issue_number = issue.get('iid')
                title = issue.get('title', '')

                tasks.append(
                    SuggestedTask(
                        git_provider=ProviderType.GITLAB,
                        task_type=TaskType.OPEN_ISSUE,
                        repo=repo_name,
                        issue_number=issue_number,
                        title=title,
                    )
                )

            return tasks
        except Exception:
            return []

    async def get_microagent_content(
        self, repository: str, file_path: str
    ) -> MicroagentContentResponse:
        """Fetch individual file content from GitLab repository.

        Args:
            repository: Repository name in format 'owner/repo' or 'domain/owner/repo'
            file_path: Path to the file within the repository

        Returns:
            MicroagentContentResponse with parsed content and triggers

        Raises:
            RuntimeError: If file cannot be fetched or doesn't exist
        """
        # Extract project_id from repository name
        project_id = self._extract_project_id(repository)

        encoded_file_path = file_path.replace('/', '%2F')
        base_url = f'{self.BASE_URL}/projects/{project_id}'
        file_url = f'{base_url}/repository/files/{encoded_file_path}/raw'

        response, _ = await self._make_request(file_url)

        # Parse the content to extract triggers from frontmatter
        return self._parse_microagent_content(response, file_path)
