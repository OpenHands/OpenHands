from datetime import datetime, timezone

from openhands.integrations.bitbucket.service.base import BitBucketMixinBase
from openhands.integrations.service_types import Branch, PaginatedBranchesResponse


class BitBucketBranchesMixin(BitBucketMixinBase):
    """
    Mixin for BitBucket branch-related operations
    """

    async def get_branches(self, repository: str) -> list[Branch]:
        """Get branches for a repository."""
        owner, repo = self._extract_owner_and_repo(repository)

        url = f'{self._repo_api_base(owner, repo)}/branches'

        # Set maximum branches to fetch (similar to GitHub/GitLab implementations)
        MAX_BRANCHES = 1000
        PER_PAGE = 100

        if self._is_server:
            params = {
                'limit': PER_PAGE,
                'orderBy': 'MODIFICATION',
            }
        else:
            params = {
                'pagelen': PER_PAGE,
                'sort': '-target.date',  # Sort by most recent commit date, descending
            }

        # Fetch all branches with pagination
        branch_data = await self._fetch_paginated_data(url, params, MAX_BRANCHES)

        return [self._parse_branch(branch) for branch in branch_data]

    async def get_paginated_branches(
        self, repository: str, page: int = 1, per_page: int = 30
    ) -> PaginatedBranchesResponse:
        """Get branches for a repository with pagination."""
        # Extract owner and repo from the repository string (e.g., "owner/repo")
        owner, repo = self._extract_owner_and_repo(repository)

        url = f'{self._repo_api_base(owner, repo)}/branches'

        if self._is_server:
            start = max((page - 1) * per_page, 0)
            params = {
                'limit': per_page,
                'start': start,
                'orderBy': 'MODIFICATION',
            }
        else:
            params = {
                'pagelen': per_page,
                'page': page,
                'sort': '-target.date',  # Sort by most recent commit date, descending
            }

        response, _ = await self._make_request(url, params)

        branches = [self._parse_branch(branch) for branch in response.get('values', [])]

        # Bitbucket provides pagination info in the response
        if self._is_server:
            has_next_page = not response.get('isLastPage', True)
            total_count = response.get('size')
        else:
            has_next_page = response.get('next') is not None
            total_count = response.get('size')  # Total number of items

        return PaginatedBranchesResponse(
            branches=branches,
            has_next_page=has_next_page,
            current_page=page,
            per_page=per_page,
            total_count=total_count,
        )

    async def search_branches(
        self, repository: str, query: str, per_page: int = 30
    ) -> list[Branch]:
        """Search branches by name using Bitbucket API with `q` param."""
        owner, repo = self._extract_owner_and_repo(repository)

        if self._is_server:
            url = f'{self._repo_api_base(owner, repo)}/branches'
            params = {
                'limit': per_page,
                'filterText': query,
                'orderBy': 'MODIFICATION',
            }
        else:
            url = f'{self._repo_api_base(owner, repo)}/refs/branches'
            # Bitbucket filtering: name ~ "query"
            params = {
                'pagelen': per_page,
                'q': f'name~"{query}"',
                'sort': '-target.date',
            }
        response, _ = await self._make_request(url, params)

        return [self._parse_branch(branch) for branch in response.get('values', [])]

    def _parse_branch(self, branch: dict) -> Branch:
        """Normalize Bitbucket branch representations across Cloud and Server."""

        if self._is_server:
            name = branch.get('displayId') or ''
            if not name:
                branch_id = branch.get('id', '')
                if isinstance(branch_id, str) and branch_id.startswith('refs/heads/'):
                    name = branch_id.split('refs/heads/', 1)[-1]
                elif isinstance(branch_id, str):
                    name = branch_id

            commit_sha = branch.get('latestCommit', '')
            last_push_date = self._extract_server_branch_last_modified(branch)
        else:
            name = branch.get('name', '')
            target = branch.get('target', {}) or {}
            commit_sha = target.get('hash', '')
            last_push_date = target.get('date')

        return Branch(
            name=name,
            commit_sha=commit_sha,
            protected=False,  # Bitbucket doesn't expose branch protection via these endpoints
            last_push_date=last_push_date,
        )

    def _extract_server_branch_last_modified(self, branch: dict) -> str | None:
        """Extract the last modified timestamp from a Bitbucket Server branch payload."""

        metadata = branch.get('metadata')
        if not isinstance(metadata, dict):
            return None

        for value in metadata.values():
            if not isinstance(value, list):
                continue
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                timestamp = (
                    entry.get('authorTimestamp')
                    or entry.get('committerTimestamp')
                    or entry.get('timestamp')
                    or entry.get('lastModified')
                )
                if isinstance(timestamp, (int, float)):
                    return datetime.fromtimestamp(
                        timestamp / 1000, tz=timezone.utc
                    ).isoformat()
                if isinstance(timestamp, str):
                    # Some Bitbucket instances might already return ISO 8601 strings
                    return timestamp

        return None
