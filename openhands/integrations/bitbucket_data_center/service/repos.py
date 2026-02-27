from typing import Any
from urllib.parse import urlparse

from openhands.integrations.bitbucket_data_center.service.base import (
    BitbucketDCMixinBase,
)
from openhands.integrations.service_types import Repository, SuggestedTask
from openhands.server.types import AppMode


class BitbucketDCReposMixin(BitbucketDCMixinBase):
    """
    Mixin for BitBucket data center repository-related operations
    """

    async def search_repositories(
        self,
        query: str,
        per_page: int,
        sort: str,
        order: str,
        public: bool,
        app_mode: AppMode,
    ) -> list[Repository]:
        """Search for repositories."""
        repositories = []

        if public:
            try:
                parsed_url = urlparse(query)
                path_segments = [
                    segment for segment in parsed_url.path.split('/') if segment
                ]

                if 'projects' in path_segments:
                    idx = path_segments.index('projects')
                    if (
                        len(path_segments) > idx + 2
                        and path_segments[idx + 1]
                        and path_segments[idx + 2] == 'repos'
                    ):
                        project_key = path_segments[idx + 1]
                        repo_name = (
                            path_segments[idx + 3]
                            if len(path_segments) > idx + 3
                            else ''
                        )
                    elif len(path_segments) > idx + 2:
                        project_key = path_segments[idx + 1]
                        repo_name = path_segments[idx + 2]
                    else:
                        project_key = ''
                        repo_name = ''
                else:
                    project_key = path_segments[0] if len(path_segments) >= 1 else ''
                    repo_name = path_segments[1] if len(path_segments) >= 2 else ''

                if project_key and repo_name:
                    repo = await self.get_repository_details_from_repo_name(
                        f'{project_key}/{repo_name}'
                    )
                    repositories.append(repo)
            except (ValueError, IndexError):
                pass

            return repositories

        MAX_REPOS = 1000
        # Search for repos once project prefix exists
        if '/' in query:
            project_slug, repo_query = query.split('/', 1)
            project_repos_url = f'{self.BASE_URL}/projects/{project_slug}/repos'
            raw_repos = await self._fetch_paginated_data(
                project_repos_url, {'limit': per_page}, MAX_REPOS
            )
            if repo_query:
                raw_repos = [
                    r
                    for r in raw_repos
                    if repo_query.lower() in r.get('slug', '').lower()
                    or repo_query.lower() in r.get('name', '').lower()
                ]
            return [await self._parse_repository(repo) for repo in raw_repos]

        # No '/' in query, search across all projects
        all_projects = await self.get_installations()
        for project_key in all_projects:
            try:
                repos = await self.get_paginated_repos(
                    1, per_page, sort, project_key, query
                )
                repositories.extend(repos)
            except Exception:
                continue
        return repositories

    async def _get_user_projects(self) -> list[dict[str, Any]]:
        """Get all projects the user has access to"""
        projects_url = f'{self.BASE_URL}/projects'
        projects = await self._fetch_paginated_data(projects_url, {}, 100)
        return projects

    async def get_installations(
        self, query: str | None = None, limit: int = 100
    ) -> list[str]:
        projects_url = f'{self.BASE_URL}/projects'
        params: dict[str, Any] = {'limit': limit}
        projects = await self._fetch_paginated_data(projects_url, params, limit)
        project_keys: list[str] = []
        for project in projects:
            key = project.get('key')
            name = project.get('name', '')
            if not key:
                continue
            if query and query.lower() not in f'{key}{name}'.lower():
                continue
            project_keys.append(key)
        return project_keys

    async def get_paginated_repos(
        self,
        page: int,
        per_page: int,
        sort: str,
        installation_id: str | None,
        query: str | None = None,
    ) -> list[Repository]:
        """Get paginated repositories for a specific project.

        Args:
            page: The page number to fetch
            per_page: The number of repositories per page
            sort: The sort field ('pushed', 'updated', 'created', 'full_name')
            installation_id: The project slug to fetch repositories from (as int, will be converted to string)

        Returns:
            A list of Repository objects
        """
        if not installation_id:
            return []

        # Convert installation_id to string for use as project_slug
        project_slug = installation_id

        project_repos_url = f'{self.BASE_URL}/projects/{project_slug}/repos'
        # Calculate start offset from page number (Bitbucket Server uses 0-based start index)
        start = (page - 1) * per_page
        params: dict[str, Any] = {'limit': per_page, 'start': start}
        response, _ = await self._make_request(project_repos_url, params)
        repos = response.get('values', [])
        if query:
            repos = [
                repo
                for repo in repos
                if query.lower() in repo.get('slug', '').lower()
                or query.lower() in repo.get('name', '').lower()
            ]
        formatted_link_header = ''
        if not response.get('isLastPage', True):
            next_page = page + 1
            # Use 'page=' format for frontend compatibility with extractNextPageFromLink
            formatted_link_header = (
                f'<{project_repos_url}?page={next_page}>; rel="next"'
            )
        return [
            await self._parse_repository(repo, link_header=formatted_link_header)
            for repo in repos
        ]

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode
    ) -> list[Repository]:
        """Get repositories for the authenticated user using workspaces endpoint.

        This method gets all repositories (both public and private) that the user has access to
        by iterating through their workspaces and fetching repositories from each workspace.
        This approach is more comprehensive and efficient than the previous implementation
        that made separate calls for public and private repositories.
        """
        MAX_REPOS = 1000
        PER_PAGE = 100  # Maximum allowed by Bitbucket data center API
        repositories: list[Repository] = []

        projects = await self.get_installations(limit=MAX_REPOS)
        for project_key in projects:
            project_repos_url = f'{self.BASE_URL}/projects/{project_key}/repos'
            project_repos = await self._fetch_paginated_data(
                project_repos_url,
                {'limit': PER_PAGE},
                MAX_REPOS - len(repositories),
            )
            for repo in project_repos:
                repositories.append(await self._parse_repository(repo))
                if len(repositories) >= MAX_REPOS:
                    break
            if len(repositories) >= MAX_REPOS:
                break
        return repositories

    async def get_suggested_tasks(self) -> list[SuggestedTask]:
        """Get suggested tasks for the authenticated user across all repositories."""
        # TODO: implemented suggested tasks
        return []
