from __future__ import annotations

from openhands.integrations.forgejo.service.base import ForgejoMixinBase
from openhands.integrations.service_types import Repository
from openhands.server.types import AppMode


class ForgejoReposMixin(ForgejoMixinBase):
    """Repository operations for Forgejo."""

    async def search_repositories(
        self,
        query: str,
        per_page: int,
        sort: str,
        order: str,
        public: bool,
        app_mode: AppMode,
    ) -> list[Repository]:  # type: ignore[override]
        url = f'{self.BASE_URL}/repos/search'
        params = {
            'q': query,
            'limit': per_page,
            'sort': sort,
            'order': order,
            'mode': 'source',
        }

        response, _ = await self._make_request(url, params)
        repos = response.get('data', []) if isinstance(response, dict) else []
        if public:
            repos = [repo for repo in repos if not repo.get('private', False)]
        return [self._parse_repository(repo) for repo in repos]

    async def get_all_repositories(
        self, sort: str, app_mode: AppMode
    ) -> list[Repository]:  # type: ignore[override]
        max_repos = 1000
        per_page = 100
        collected: list[dict] = []
        page = 1
        last_link_header: str | None = None

        url = f'{self.BASE_URL}/user/repos'
        forgejo_sort = self._map_sort(sort)

        while len(collected) < max_repos:
            params = {
                'page': str(page),
                'limit': str(per_page),
                'sort': forgejo_sort,
            }
            response, headers = await self._make_request(url, params)
            last_link_header = headers.get('Link')

            page_repos = response if isinstance(response, list) else []
            if not page_repos:
                break

            collected.extend(page_repos)
            if 'rel="next"' not in (last_link_header or ''):
                break

            page += 1

        collected = collected[:max_repos]
        return [
            self._parse_repository(repo, link_header=last_link_header)
            for repo in collected
        ]

    async def get_paginated_repos(
        self,
        page: int,
        per_page: int,
        sort: str,
        installation_id: str | None,
        query: str | None = None,
    ) -> list[Repository]:  # type: ignore[override]
        _ = installation_id
        url = f'{self.BASE_URL}/user/repos'
        params = {
            'page': str(page),
            'limit': str(per_page),
            'sort': self._map_sort(sort),
        }

        response, headers = await self._make_request(url, params)
        repos = response if isinstance(response, list) else []

        if query:
            lowered = query.lower()
            repos = [
                repo
                for repo in repos
                if lowered in (repo.get('full_name') or '').lower()
            ]

        link_header = headers.get('Link')
        return [self._parse_repository(repo, link_header=link_header) for repo in repos]

    async def get_repository_details_from_repo_name(
        self, repository: str
    ) -> Repository:  # type: ignore[override]
        owner, repo = self._split_repo(repository)
        url = self._build_repo_api_url(owner, repo)
        response, headers = await self._make_request(url)
        link_header = headers.get('Link')
        return self._parse_repository(response, link_header=link_header)
