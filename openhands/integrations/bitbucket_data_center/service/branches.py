from datetime import datetime, timezone

from openhands.integrations.bitbucket_data_center.service.base import (
    BitbucketDCMixinBase,
)
from openhands.integrations.service_types import Branch, PaginatedBranchesResponse


class BitbucketDCBranchesMixin(BitbucketDCMixinBase):
    """
    Mixin for BitBucket data center branch-related operations
    """

    async def get_branches(self, repository: str) -> list[Branch]:
        """Get branches for a repository."""
        owner, repo = self._extract_owner_and_repo(repository)

        url = f'{self._repo_api_base(owner, repo)}/branches'

        # Set maximum branches to fetch (similar to GitHub/GitLab implementations)
        MAX_BRANCHES = 1000
        PER_PAGE = 100

        params = {
            'limit': PER_PAGE,
            'orderBy': 'MODIFICATION',
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
        parts = repository.split('/')
        if len(parts) < 2:
            raise ValueError(f'Invalid repository name: {repository}')

        owner = parts[-2]
        repo = parts[-1]

        url = f'{self._repo_api_base(owner, repo)}/branches'

        start = max((page - 1) * per_page, 0)
        params = {
            'limit': per_page,
            'start': start,
            'orderBy': 'MODIFICATION',
        }

        response, _ = await self._make_request(url, params)

        branches = [self._parse_branch(branch) for branch in response.get('values', [])]

        has_next_page = not response.get('isLastPage', True)
        total_count = response.get('size')

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
        """Search branches by name using Bitbucket data center API with `q` param."""
        owner, repo = self._extract_owner_and_repo(repository)

        url = f'{self._repo_api_base(owner, repo)}/branches'
        params = {
            'limit': per_page,
            'filterText': query,
            'orderBy': 'MODIFICATION',
        }

        response, _ = await self._make_request(url, params)

        return [self._parse_branch(branch) for branch in response.get('values', [])]

    def _parse_branch(self, branch: dict) -> Branch:
        """Normalize Bitbucket branch representations across Cloud and Server."""

        name = branch.get('displayId') or ''
        if not name:
            branch_id = branch.get('id', '')
            if isinstance(branch_id, str) and branch_id.startswith('refs/heads/'):
                name = branch_id.split('refs/heads/', 1)[-1]
            elif isinstance(branch_id, str):
                name = branch_id

        commit_sha = branch.get('latestCommit', '')
        last_push_date = self._extract_server_branch_last_modified(branch)

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
