"""Feature operations for Azure DevOps integration (microagents, suggested tasks, user)."""

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.service.base import AzureDevOpsMixinBase
from openhands.integrations.service_types import (
    MicroagentContentResponse,
    ProviderType,
    RequestMethod,
    SuggestedTask,
    TaskType,
    User,
)


class AzureDevOpsFeaturesMixin(AzureDevOpsMixinBase):
    """Mixin for Azure DevOps feature operations (microagents, suggested tasks, user info)."""

    async def get_user(self) -> User:
        """Get the authenticated user's information."""
        url = f'{self.base_url}/_apis/connectionData?api-version=7.1-preview.1'
        response, _ = await self._make_request(url)

        # Extract authenticated user details
        authenticated_user = response.get('authenticatedUser', {})
        user_id = authenticated_user.get('id', '')
        display_name = authenticated_user.get('providerDisplayName', '')

        # Get descriptor for potential additional details
        authenticated_user.get('descriptor', '')

        return User(
            id=str(user_id),
            login=display_name,
            avatar_url='',
            name=display_name,
            email='',
            company=None,
        )

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        """Get suggested tasks for the authenticated user across all repositories."""
        # Azure DevOps requires querying each project separately for PRs and work items
        # Since we no longer specify a single project, we need to query all projects
        # Get all projects first
        projects_url = f'{self.base_url}/_apis/projects?api-version=7.1'
        projects_response, _ = await self._make_request(projects_url)
        projects = projects_response.get('value', [])

        # Get user info
        user = await self.get_user()
        tasks = []

        # Query each project for pull requests and work items
        for project in projects:
            project_name = project.get('name')

            try:
                # URL-encode project name to handle spaces and special characters
                project_enc = self._encode_url_component(project_name)

                # Get pull requests created by the user in this project
                url = f'{self.base_url}/{project_enc}/_apis/git/pullrequests?api-version=7.1&searchCriteria.creatorId={user.id}&searchCriteria.status=active'
                response, _ = await self._make_request(url)

                pull_requests = response.get('value', [])

                for pr in pull_requests:
                    repo_name = pr.get('repository', {}).get('name', '')
                    pr_id = pr.get('pullRequestId')
                    title = pr.get('title', '')

                    # Check for merge conflicts
                    if pr.get('mergeStatus') == 'conflicts':
                        tasks.append(
                            SuggestedTask(
                                git_provider=ProviderType.AZURE_DEVOPS,
                                task_type=TaskType.MERGE_CONFLICTS,
                                repo=f'{self.organization}/{project_name}/{repo_name}',
                                issue_number=pr_id,
                                title=title,
                            )
                        )
                    # Check for failing checks
                    elif pr.get('status') == 'failed':
                        tasks.append(
                            SuggestedTask(
                                git_provider=ProviderType.AZURE_DEVOPS,
                                task_type=TaskType.FAILING_CHECKS,
                                repo=f'{self.organization}/{project_name}/{repo_name}',
                                issue_number=pr_id,
                                title=title,
                            )
                        )
                    # Check for unresolved comments
                    elif pr.get('hasUnresolvedComments', False):
                        tasks.append(
                            SuggestedTask(
                                git_provider=ProviderType.AZURE_DEVOPS,
                                task_type=TaskType.UNRESOLVED_COMMENTS,
                                repo=f'{self.organization}/{project_name}/{repo_name}',
                                issue_number=pr_id,
                                title=title,
                            )
                        )

                # Get work items assigned to the user in this project
                work_items_url = (
                    f'{self.base_url}/{project_enc}/_apis/wit/wiql?api-version=7.1'
                )
                wiql_query = {
                    'query': "SELECT [System.Id], [System.Title], [System.State] FROM WorkItems WHERE [System.AssignedTo] = @me AND [System.State] = 'Active'"
                }

                work_items_response, _ = await self._make_request(
                    url=work_items_url, params=wiql_query, method=RequestMethod.POST
                )

                work_item_references = work_items_response.get('workItems', [])

                # Get details for each work item
                for work_item_ref in work_item_references:
                    work_item_id = work_item_ref.get('id')
                    work_item_url = f'{self.base_url}/{project_enc}/_apis/wit/workitems/{work_item_id}?api-version=7.1'
                    work_item, _ = await self._make_request(work_item_url)

                    title = work_item.get('fields', {}).get('System.Title', '')

                    tasks.append(
                        SuggestedTask(
                            git_provider=ProviderType.AZURE_DEVOPS,
                            task_type=TaskType.OPEN_ISSUE,
                            repo=f'{self.organization}/{project_name}',
                            issue_number=work_item_id,
                            title=title,
                        )
                    )
            except Exception:
                # Skip projects that fail (e.g., no access, no work items enabled)
                continue

        return tasks

    async def _get_cursorrules_url(self, repository: str) -> str:
        """Get the URL for checking .cursorrules file in Azure DevOps."""
        org, project, repo = self._parse_repository(repository)
        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)
        return f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/items?path=/.cursorrules&api-version=7.1'

    async def _get_microagents_directory_url(
        self, repository: str, microagents_path: str
    ) -> str:
        """Get the URL for checking microagents directory in Azure DevOps.

        Note: For org-level microagents (e.g., 'org/.openhands'), Azure DevOps doesn't support
        this concept, so we raise ValueError to let the caller fall back to other providers.
        """
        parts = repository.split('/')
        if len(parts) < 3:
            # Azure DevOps doesn't support org-level configs, only full repo paths
            raise ValueError(
                f'Invalid repository format: {repository}. Expected format: organization/project/repo'
            )
        org, project, repo = parts[0], parts[1], parts[2]
        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)
        return f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/items?path=/{microagents_path}&recursionLevel=OneLevel&api-version=7.1'

    def _get_microagents_directory_params(self, microagents_path: str) -> dict | None:
        """Get parameters for the microagents directory request. Return None if no parameters needed."""
        return None

    def _is_valid_microagent_file(self, item: dict) -> bool:
        """Check if an item represents a valid microagent file in Azure DevOps."""
        return (
            not item.get('isFolder', False)
            and item.get('path', '').endswith('.md')
            and not item.get('path', '').endswith('README.md')
        )

    def _get_file_name_from_item(self, item: dict) -> str:
        """Extract file name from directory item in Azure DevOps."""
        path = item.get('path', '')
        return path.split('/')[-1] if path else ''

    def _get_file_path_from_item(self, item: dict, microagents_path: str) -> str:
        """Extract file path from directory item in Azure DevOps."""
        return item.get('path', '').lstrip('/')

    async def get_microagent_content(
        self, repository: str, file_path: str
    ) -> MicroagentContentResponse:
        """Get content of a specific microagent file.

        Args:
            repository: Repository name in Azure DevOps format 'org/project/repo'
            file_path: Path to the microagent file

        Returns:
            MicroagentContentResponse with parsed content and triggers
        """
        org, project, repo = self._parse_repository(repository)
        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)
        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/items?path={file_path}&api-version=7.1'

        try:
            response, _ = await self._make_request(url)
            content = (
                response if isinstance(response, str) else response.get('content', '')
            )

            # Parse the content using the base class method
            return self._parse_microagent_content(content, file_path)
        except Exception as e:
            logger.warning(f'Failed to fetch microagent content from {file_path}: {e}')
            raise
