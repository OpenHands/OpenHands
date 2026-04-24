"""Feature operations for Azure DevOps integration (microagents, suggested tasks, user)."""

from openhands.integrations.azure_devops.service.base import AzureDevOpsMixinBase
from openhands.integrations.service_types import (
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
