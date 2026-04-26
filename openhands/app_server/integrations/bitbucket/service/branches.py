from openhands.app_server.integrations.bitbucket.service.base import BitBucketMixinBase
from openhands.app_server.integrations.service_types import (
    Branch,
    PaginatedBranchesResponse,
)


def _target(branch: dict) -> dict:
    """Return ``branch['target']`` as a dict, treating ``null`` /
    non-dict values as empty so callers can safely chain ``.get``.

    Bitbucket cloud branches API normally returns a ``target`` commit object,
    but corporate proxies and edge-case repo states have been observed to
    return ``"target": null``. The original ``branch.get('target', {})``
    pattern only short-circuited the absent-key case and crashed with
    ``AttributeError`` on an explicit null. Mirrors the defensive idiom
    used in ``bitbucket/service/base.py::get_user`` (#14070) and
    ``_parse_repository`` (#14085).
    """
    target = branch.get('target') or {}
    return target if isinstance(target, dict) else {}


class BitBucketBranchesMixin(BitBucketMixinBase):
    """
    Mixin for BitBucket branch-related operations
    """

    async def get_branches(self, repository: str) -> list[Branch]:
        """Get branches for a repository."""
        owner, repo = self._extract_owner_and_repo(repository)

        url = f'{self.BASE_URL}/repositories/{owner}/{repo}/refs/branches'

        # Set maximum branches to fetch (similar to GitHub/GitLab implementations)
        MAX_BRANCHES = 1000
        PER_PAGE = 100

        params = {
            'pagelen': PER_PAGE,
            'sort': '-target.date',  # Sort by most recent commit date, descending
        }

        # Fetch all branches with pagination
        branch_data = await self._fetch_paginated_data(url, params, MAX_BRANCHES)

        branches = []
        for branch in branch_data:
            branches.append(
                Branch(
                    name=branch.get('name', ''),
                    commit_sha=_target(branch).get('hash', ''),
                    protected=False,  # Bitbucket doesn't expose this in the API
                    last_push_date=_target(branch).get('date', None),
                )
            )

        return branches

    async def get_paginated_branches(
        self, repository: str, page: int = 1, per_page: int = 30
    ) -> PaginatedBranchesResponse:
        """Get branches for a repository with pagination."""
        # Extract owner and repo from the repository string (e.g., "owner/repo")
        parts = repository.split('/')
        if len(parts) < 2:
            raise ValueError(f'Invalid repository name: {repository}')

        owner = parts[-2]
        repo = parts[-1]

        url = f'{self.BASE_URL}/repositories/{owner}/{repo}/refs/branches'

        params = {
            'pagelen': per_page,
            'page': page,
            'sort': '-target.date',  # Sort by most recent commit date, descending
        }

        response, _ = await self._make_request(url, params)

        branches = []
        for branch in response.get('values', []):
            branches.append(
                Branch(
                    name=branch.get('name', ''),
                    commit_sha=_target(branch).get('hash', ''),
                    protected=False,  # Bitbucket doesn't expose this in the API
                    last_push_date=_target(branch).get('date', None),
                )
            )

        # Bitbucket provides pagination info in the response
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
        parts = repository.split('/')
        if len(parts) < 2:
            raise ValueError(f'Invalid repository name: {repository}')

        owner = parts[-2]
        repo = parts[-1]

        url = f'{self.BASE_URL}/repositories/{owner}/{repo}/refs/branches'
        # Bitbucket filtering: name ~ "query"
        params = {
            'pagelen': per_page,
            'q': f'name~"{query}"',
            'sort': '-target.date',
        }
        response, _ = await self._make_request(url, params)

        branches: list[Branch] = []
        for branch in response.get('values', []):
            branches.append(
                Branch(
                    name=branch.get('name', ''),
                    commit_sha=_target(branch).get('hash', ''),
                    protected=False,
                    last_push_date=_target(branch).get('date', None),
                )
            )
        return branches
