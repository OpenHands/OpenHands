from datetime import datetime

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.service.base import GitHubMixinBase
from openhands.integrations.service_types import OwnerType, ProviderType, Repository
from openhands.server.types import AppMode


class GitHubReposMixin(GitHubMixinBase):
    """
    Methods for interacting with GitHub repositories (from both personal and app installations)
    """

    async def get_installations(self) -> list[str]:
        url = f'{self.BASE_URL}/user/installations'
        response, _ = await self._make_request(url)
        installations = response.get('installations', [])
        return [str(i['id']) for i in installations]

    async def _fetch_paginated_repos(
        self, url: str, params: dict, max_repos: int, extract_key: str | None = None
    ) -> list[dict]:
        """Fetch repositories with pagination support.

        Args:
            url: The API endpoint URL
            params: Query parameters for the request
            max_repos: Maximum number of repositories to fetch
            extract_key: If provided, extract repositories from this key in the response

        Returns:
            List of repository dictionaries
        """
        repos: list[dict] = []
        page = 1

        while len(repos) < max_repos:
            page_params = {**params, 'page': str(page)}
            response, headers = await self._make_request(url, page_params)

            # Extract repositories from response
            page_repos = response.get(extract_key, []) if extract_key else response

            if not page_repos:  # No more repositories
                break

            repos.extend(page_repos)
            page += 1

            # Check if we've reached the last page
            link_header = headers.get('Link', '')
            if 'rel="next"' not in link_header:
                break

        return repos[:max_repos]  # Trim to max_repos if needed

    def parse_pushed_at_date(self, repo):
        ts = repo.get('pushed_at')
        return datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ') if ts else datetime.min

    def _parse_repository(
        self, repo: dict, link_header: str | None = None
    ) -> Repository:
        """Parse a GitHub API repository response into a Repository object.

        Args:
            repo: Repository data from GitHub API
            link_header: Optional link header for pagination

        Returns:
            Repository object
        """
        return Repository(
            id=str(repo.get('id')),  # type: ignore[arg-type]
            full_name=repo.get('full_name'),  # type: ignore[arg-type]
            stargazers_count=repo.get('stargazers_count'),
            git_provider=ProviderType.GITHUB,
            is_public=not repo.get('private', True),
            owner_type=(
                OwnerType.ORGANIZATION
                if repo.get('owner', {}).get('type') == 'Organization'
                else OwnerType.USER
            ),
            link_header=link_header,
            main_branch=repo.get('default_branch'),
        )

    async def get_paginated_repos(
        self,
        page: int,
        per_page: int,
        sort: str,
        installation_id: str | None,
        query: str | None = None,
    ):
        params = {'page': str(page), 'per_page': str(per_page)}
        if installation_id:
            url = f'{self.BASE_URL}/user/installations/{installation_id}/repositories'
            response, headers = await self._make_request(url, params)
            response = response.get('repositories', [])
        else:
            url = f'{self.BASE_URL}/user/repos'
            params['sort'] = sort
            response, headers = await self._make_request(url, params)

        next_link: str = headers.get('Link', '')
        return [
            self._parse_repository(repo, link_header=next_link) for repo in response
        ]

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode
    ) -> list[Repository]:
        MAX_REPOS = 1000
        PER_PAGE = 100  # Maximum allowed by GitHub API
        all_repos: list[dict] = []

        if app_mode == AppMode.SAAS:
            # Get all installation IDs and fetch repos for each one
            installation_ids = await self.get_installations()

            # Iterate through each installation ID
            for installation_id in installation_ids:
                params = {'per_page': str(PER_PAGE)}
                url = (
                    f'{self.BASE_URL}/user/installations/{installation_id}/repositories'
                )

                # Fetch repositories for this installation
                installation_repos = await self._fetch_paginated_repos(
                    url, params, MAX_REPOS - len(all_repos), extract_key='repositories'
                )

                all_repos.extend(installation_repos)

                # If we've already reached MAX_REPOS, no need to check other installations
                if len(all_repos) >= MAX_REPOS:
                    break

            if sort == 'pushed':
                all_repos.sort(key=self.parse_pushed_at_date, reverse=True)
        else:
            # Original behavior for non-SaaS mode
            params = {'per_page': str(PER_PAGE), 'sort': sort}
            url = f'{self.BASE_URL}/user/repos'

            # Fetch user repositories
            all_repos = await self._fetch_paginated_repos(url, params, MAX_REPOS)

        # Convert to Repository objects
        return [self._parse_repository(repo) for repo in all_repos]

    async def get_user_organizations(self) -> list[str]:
        """Get list of organization logins that the user is a member of."""
        url = f'{self.BASE_URL}/user/orgs'
        try:
            response, _ = await self._make_request(url)
            orgs = [org['login'] for org in response]
            return orgs
        except Exception as e:
            logger.warning(f'Failed to get user organizations: {e}')
            return []

    async def get_organizations_from_installations(self) -> list[str]:
        """Get list of organization logins from GitHub App installations.

        This method provides a more reliable way to get organizations that the
        GitHub App has access to, regardless of user membership context.
        """
        try:
            # Get installations with account details
            url = f'{self.BASE_URL}/user/installations'
            response, _ = await self._make_request(url)
            installations = response.get('installations', [])

            orgs = []
            for installation in installations:
                account = installation.get('account', {})
                if account.get('type') == 'Organization':
                    orgs.append(account.get('login'))

            return orgs
        except Exception as e:
            logger.warning(f'Failed to get organizations from installations: {e}')
            return []

    def _fuzzy_match_org_name(self, query: str, org_name: str) -> bool:
        """Check if query fuzzy matches organization name."""
        query_lower = query.lower().replace('-', '').replace('_', '').replace(' ', '')
        org_lower = org_name.lower().replace('-', '').replace('_', '').replace(' ', '')

        # Exact match after normalization
        if query_lower == org_lower:
            return True

        # Query is a substring of org name
        if query_lower in org_lower:
            return True

        # Org name is a substring of query (less common but possible)
        if org_lower in query_lower:
            return True

        return False

    async def search_repositories(
        self,
        query: str,
        per_page: int,
        sort: str,
        order: str,
        public: bool,
        app_mode: AppMode,
    ) -> list[Repository]:
        url = f'{self.BASE_URL}/search/repositories'
        params = {
            'per_page': per_page,
            'sort': sort,
            'order': order,
        }

        if public:
            url_parts = query.split('/')
            if len(url_parts) < 4:
                return []

            org = url_parts[3]
            repo_name = url_parts[4]
            # Add is:public to the query to ensure we only search for public repositories
            params['q'] = f'in:name {org}/{repo_name} is:public'

        # Handle private repository searches
        if not public and '/' in query:
            org, repo_query = query.split('/', 1)
            query_with_user = f'org:{org} in:name {repo_query}'
            params['q'] = query_with_user
        elif not public:
            # Expand search scope to include user's repositories and organizations the app has access to
            user = await self.get_user()
            if app_mode == AppMode.SAAS:
                user_orgs = await self.get_organizations_from_installations()
            else:
                user_orgs = await self.get_user_organizations()

            # Search in user repos and org repos separately
            all_repos = []

            # Search in user repositories
            user_query = f'in:name {query} user:{user.login}'
            user_params = params.copy()
            user_params['q'] = user_query

            try:
                user_response, _ = await self._make_request(url, user_params)
                user_items = user_response.get('items', [])
                all_repos.extend(user_items)
            except Exception as e:
                logger.warning(f'User search failed: {e}')

            # Search for repos named "query" in each organization
            for org in user_orgs:
                org_query = f'{query} org:{org}'
                org_params = params.copy()
                org_params['q'] = org_query

                try:
                    org_response, _ = await self._make_request(url, org_params)
                    org_items = org_response.get('items', [])
                    all_repos.extend(org_items)
                except Exception as e:
                    logger.warning(f'Org {org} search failed: {e}')

            # Also search for top repos from orgs that match the query name
            for org in user_orgs:
                if self._fuzzy_match_org_name(query, org):
                    org_repos_query = f'org:{org}'
                    org_repos_params = params.copy()
                    org_repos_params['q'] = org_repos_query
                    org_repos_params['sort'] = 'stars'
                    org_repos_params['per_page'] = 2  # Limit to first 2 repos

                    try:
                        org_repos_response, _ = await self._make_request(
                            url, org_repos_params
                        )
                        org_repo_items = org_repos_response.get('items', [])
                        all_repos.extend(org_repo_items)
                    except Exception as e:
                        logger.warning(f'Org repos search for {org} failed: {e}')

            return [self._parse_repository(repo) for repo in all_repos]

        # Default case (public search or slash query)
        response, _ = await self._make_request(url, params)
        repo_items = response.get('items', [])
        return [self._parse_repository(repo) for repo in repo_items]

    async def get_repository_details_from_repo_name(
        self, repository: str
    ) -> Repository:
        url = f'{self.BASE_URL}/repos/{repository}'
        repo, _ = await self._make_request(url)

        return self._parse_repository(repo)
