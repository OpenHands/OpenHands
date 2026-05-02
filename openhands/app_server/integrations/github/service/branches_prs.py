from openhands.app_server.integrations.github.queries import (
    search_branches_graphql_query,
)
from openhands.app_server.integrations.github.service.base import GitHubMixinBase
from openhands.app_server.integrations.service_types import (
    Branch,
    PaginatedBranchesResponse,
    SearchBranchesOutcome,
)
from openhands.app_server.utils.logger import openhands_logger as logger


class GitHubBranchesMixin(GitHubMixinBase):
    """
    Methods for interacting with branches for a repo
    """

    async def get_branches(self, repository: str) -> list[Branch]:
        """Get branches for a repository"""
        url = f'{self.BASE_URL}/repos/{repository}/branches'

        # Set maximum branches to fetch (100 per page)
        MAX_BRANCHES = 5_000
        PER_PAGE = 100

        all_branches: list[Branch] = []
        page = 1

        # Fetch up to 10 pages of branches
        while len(all_branches) < MAX_BRANCHES:
            params = {'per_page': str(PER_PAGE), 'page': str(page)}
            response, headers = await self._make_request(url, params)

            if not response:  # No more branches
                break

            for branch_data in response:
                # Extract the last commit date if available
                last_push_date = None
                if branch_data.get('commit') and branch_data['commit'].get('commit'):
                    commit_info = branch_data['commit']['commit']
                    if commit_info.get('committer') and commit_info['committer'].get(
                        'date'
                    ):
                        last_push_date = commit_info['committer']['date']

                branch = Branch(
                    name=branch_data.get('name'),
                    commit_sha=branch_data.get('commit', {}).get('sha', ''),
                    protected=branch_data.get('protected', False),
                    last_push_date=last_push_date,
                )
                all_branches.append(branch)

            page += 1

            # Check if we've reached the last page
            link_header = headers.get('Link', '')
            if 'rel="next"' not in link_header:
                break

        return all_branches

    async def get_paginated_branches(
        self, repository: str, page: int = 1, per_page: int = 30
    ) -> PaginatedBranchesResponse:
        """Get branches for a repository with pagination"""
        url = f'{self.BASE_URL}/repos/{repository}/branches'

        params = {'per_page': str(per_page), 'page': str(page)}
        response, headers = await self._make_request(url, params)

        branches: list[Branch] = []
        for branch_data in response:
            # Extract the last commit date if available
            last_push_date = None
            if branch_data.get('commit') and branch_data['commit'].get('commit'):
                commit_info = branch_data['commit']['commit']
                if commit_info.get('committer') and commit_info['committer'].get(
                    'date'
                ):
                    last_push_date = commit_info['committer']['date']

            branch = Branch(
                name=branch_data.get('name'),
                commit_sha=branch_data.get('commit', {}).get('sha', ''),
                protected=branch_data.get('protected', False),
                last_push_date=last_push_date,
            )
            branches.append(branch)

        # Parse Link header to determine if there's a next page
        has_next_page = False
        if 'Link' in headers:
            link_header = headers['Link']
            has_next_page = 'rel="next"' in link_header

        return PaginatedBranchesResponse(
            branches=branches,
            has_next_page=has_next_page,
            current_page=page,
            per_page=per_page,
            total_count=None,  # GitHub doesn't provide total count in branch API
        )

    async def search_branches(
        self,
        repository: str,
        query: str,
        per_page: int = 30,
        *,
        page: int = 1,
        after: str | None = None,
    ) -> list[Branch]:
        """Search branches by name using GitHub GraphQL with a partial query."""
        out = await self.search_branches_with_continuation(
            repository, query, per_page, after
        )
        return out.branches

    async def search_branches_with_continuation(
        self,
        repository: str,
        query: str,
        per_page: int = 30,
        after: str | None = None,
    ) -> SearchBranchesOutcome:
        """Search branches and return one page plus optional GraphQL `after` cursor."""
        if not query:
            return SearchBranchesOutcome(branches=[], github_next_after_cursor=None)

        per_page = min(max(per_page, 1), 100)

        parts = repository.split('/')
        if len(parts) < 2:
            return SearchBranchesOutcome(branches=[], github_next_after_cursor=None)

        owner, name = parts[-2], parts[-1]

        first = min(per_page, 100)
        variables = {
            'owner': owner,
            'name': name,
            'query': query or '',
            'first': first,
            'after': after,
        }

        try:
            result = await self.execute_graphql_query(
                search_branches_graphql_query, variables
            )
        except Exception as e:
            logger.warning(f'Failed to search for branches: {e}')
            return SearchBranchesOutcome(branches=[], github_next_after_cursor=None)

        repo = result.get('data', {}).get('repository')
        refs_block = repo.get('refs') if repo else None
        if not refs_block:
            return SearchBranchesOutcome(branches=[], github_next_after_cursor=None)

        edges = refs_block.get('edges') or []
        branches: list[Branch] = []
        for edge in edges:
            node = edge.get('node') or {}
            bname = node.get('name') or ''
            target = node.get('target') or {}
            typename = target.get('__typename')
            commit_sha = ''
            last_push_date = None
            if typename == 'Commit':
                commit_sha = target.get('oid', '') or ''
                last_push_date = target.get('committedDate')

            protected = node.get('branchProtectionRule') is not None

            branches.append(
                Branch(
                    name=bname,
                    commit_sha=commit_sha,
                    protected=protected,
                    last_push_date=last_push_date,
                )
            )

        page_info = refs_block.get('pageInfo') or {}
        next_cursor: str | None = None
        if edges and page_info.get('hasNextPage'):
            next_cursor = edges[-1].get('cursor')

        return SearchBranchesOutcome(
            branches=branches, github_next_after_cursor=next_cursor
        )
