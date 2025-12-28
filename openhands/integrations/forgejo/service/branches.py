from __future__ import annotations

from openhands.integrations.forgejo.service.base import ForgejoMixinBase
from openhands.integrations.service_types import Branch, PaginatedBranchesResponse


class ForgejoBranchesMixin(ForgejoMixinBase):
    """Branch-related operations for Forgejo."""

    async def get_branches(self, repository: str) -> list[Branch]:  # type: ignore[override]
        branches: list[Branch] = []
        page = 1
        per_page = 100

        while True:
            paginated = await self.get_paginated_branches(repository, page, per_page)
            branches.extend(paginated.branches)
            if not paginated.has_next_page:
                break
            page += 1

        return branches

    async def get_paginated_branches(
        self, repository: str, page: int = 1, per_page: int = 30
    ) -> PaginatedBranchesResponse:  # type: ignore[override]
        owner, repo = self._split_repo(repository)
        url = self._build_repo_api_url(owner, repo, 'branches')
        params = {
            'page': str(page),
            'limit': str(per_page),
        }

        response, headers = await self._make_request(url, params)
        branch_items = response if isinstance(response, list) else []

        branches: list[Branch] = []
        for branch in branch_items:
            commit_info = branch.get('commit') or {}
            commit_sha = (
                commit_info.get('id')
                or commit_info.get('sha')
                or commit_info.get('commit', {}).get('sha')
            )
            branches.append(
                Branch(
                    name=branch.get('name', ''),
                    commit_sha=commit_sha or '',
                    protected=branch.get('protected', False),
                    last_push_date=None,
                )
            )

        link_header = headers.get('Link', '')
        total_count_header = headers.get('X-Total-Count') or headers.get('X-Total')
        total_count = int(total_count_header) if total_count_header else None
        has_next_page = 'rel="next"' in link_header

        return PaginatedBranchesResponse(
            branches=branches,
            has_next_page=has_next_page,
            current_page=page,
            per_page=per_page,
            total_count=total_count,
        )

    async def search_branches(
        self, repository: str, query: str, per_page: int = 30
    ) -> list[Branch]:  # type: ignore[override]
        all_branches = await self.get_branches(repository)
        lowered = query.lower()
        return [branch for branch in all_branches if lowered in branch.name.lower()][
            :per_page
        ]
