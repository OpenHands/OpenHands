"""Repository operations for Azure DevOps integration."""

from openhands.integrations.azure_devops.service.base import AzureDevOpsMixinBase
from openhands.integrations.service_types import ProviderType, Repository
from openhands.server.types import AppMode


class AzureDevOpsReposMixin(AzureDevOpsMixinBase):
    """Mixin for Azure DevOps repository operations."""

    async def search_repositories(
        self,
        query: str,
        per_page: int = 30,
        sort: str = 'updated',
        order: str = 'desc',
        public: bool = False,
        app_mode: AppMode = AppMode.OSS,
    ) -> list[Repository]:
        """Search for repositories in Azure DevOps."""
        # Get all repositories across all projects in the organization
        url = f'{self.base_url}/_apis/git/repositories?api-version=7.1'

        response, _ = await self._make_request(url)

        # Filter repositories by query if provided
        repos = response.get('value', [])
        if query:
            repos = [
                repo for repo in repos if query.lower() in repo.get('name', '').lower()
            ]

        # Limit to per_page
        repos = repos[:per_page]

        return [
            Repository(
                id=str(repo.get('id')),
                full_name=f'{self.organization}/{repo.get("project", {}).get("name", "")}/{repo.get("name")}',
                git_provider=ProviderType.AZURE_DEVOPS,
                is_public=False,  # Azure DevOps repos are private by default
            )
            for repo in repos
        ]

    async def get_repositories(self, sort: str, app_mode: AppMode) -> list[Repository]:
        """Get repositories for the authenticated user."""
        MAX_REPOS = 1000

        # Get all projects first
        projects_url = f'{self.base_url}/_apis/projects?api-version=7.1'
        projects_response, _ = await self._make_request(projects_url)
        projects = projects_response.get('value', [])

        all_repos = []

        # For each project, get its repositories
        for project in projects:
            project_name = project.get('name')
            project_enc = self._encode_url_component(project_name)
            repos_url = (
                f'{self.base_url}/{project_enc}/_apis/git/repositories?api-version=7.1'
            )
            repos_response, _ = await self._make_request(repos_url)
            repos = repos_response.get('value', [])

            for repo in repos:
                all_repos.append(
                    {
                        'id': repo.get('id'),
                        'name': repo.get('name'),
                        'project_name': project_name,
                        'updated_date': repo.get('lastUpdateTime'),
                    }
                )

                if len(all_repos) >= MAX_REPOS:
                    break

            if len(all_repos) >= MAX_REPOS:
                break

        # Sort repositories based on the sort parameter
        if sort == 'updated':
            all_repos.sort(key=lambda r: r.get('updated_date', ''), reverse=True)
        elif sort == 'name':
            all_repos.sort(key=lambda r: r.get('name', '').lower())

        return [
            Repository(
                id=str(repo.get('id')),
                full_name=f'{self.organization}/{repo.get("project_name")}/{repo.get("name")}',
                git_provider=ProviderType.AZURE_DEVOPS,
                is_public=False,  # Azure DevOps repos are private by default
            )
            for repo in all_repos[:MAX_REPOS]
        ]

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode
    ) -> list[Repository]:
        """Get repositories for the authenticated user (alias for get_repositories)."""
        return await self.get_repositories(sort, app_mode)

    def _parse_repository_response(
        self, repo: dict, project_name: str, link_header: str | None = None
    ) -> Repository:
        """Parse an Azure DevOps API repository response into a Repository object.

        Args:
            repo: Repository data from Azure DevOps API
            project_name: The project name the repository belongs to
            link_header: Optional link header for pagination

        Returns:
            Repository object
        """
        return Repository(
            id=str(repo.get('id')),
            full_name=f'{self.organization}/{project_name}/{repo.get("name")}',
            git_provider=ProviderType.AZURE_DEVOPS,
            is_public=False,  # Azure DevOps repos are private by default
            link_header=link_header,
        )

    async def get_paginated_repos(
        self,
        page: int,
        per_page: int,
        sort: str,
        installation_id: str | None,
        query: str | None = None,
    ) -> list[Repository]:
        """Get a page of repositories for the authenticated user."""
        # Get all repos first, then paginate manually
        # Azure DevOps doesn't have native pagination for repositories
        all_repos = await self.get_repositories(sort, AppMode.SAAS)

        # Calculate pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Filter by query if provided
        if query:
            query_lower = query.lower()
            all_repos = [
                repo for repo in all_repos if query_lower in repo.full_name.lower()
            ]

        return all_repos[start_idx:end_idx]

    async def get_repository_details_from_repo_name(
        self, repository: str
    ) -> Repository:
        """Gets all repository details from repository name.

        Args:
            repository: Repository name in format 'organization/project/repo'

        Returns:
            Repository object with details
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}?api-version=7.1'
        response, _ = await self._make_request(url)

        return Repository(
            id=str(response.get('id')),
            full_name=f'{org}/{project}/{repo}',
            git_provider=ProviderType.AZURE_DEVOPS,
            is_public=False,  # Azure DevOps repos are private by default
        )
