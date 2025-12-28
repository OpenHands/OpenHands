from __future__ import annotations

from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.forgejo.service.base import ForgejoMixinBase
from openhands.integrations.service_types import RequestMethod, UnknownException


class ForgejoPRsMixin(ForgejoMixinBase):
    """Pull request helpers for Forgejo."""

    async def create_pull_request(self, data: dict[str, Any] | None = None) -> dict:
        payload: dict[str, Any] = dict(data or {})

        repository = payload.pop('repository', None)
        owner = payload.pop('owner', None)
        repo_name = payload.pop('repo', None)

        if repository and isinstance(repository, str):
            owner, repo_name = self._split_repo(repository)
        else:
            owner = str(owner or self.user_id or '').strip()
            repo_name = str(repo_name or '').strip()

        if not owner or not repo_name:
            raise ValueError(
                'Repository information is required to create a pull request'
            )

        url = self._build_repo_api_url(owner, repo_name, 'pulls')
        response, _ = await self._make_request(
            url,
            payload,
            method=RequestMethod.POST,
        )

        if not isinstance(response, dict):
            raise UnknownException('Unexpected response creating Forgejo pull request')

        if 'number' not in response and 'index' in response:
            response['number'] = response['index']

        if 'html_url' not in response and 'url' in response:
            response['html_url'] = response['url']

        return response

    async def request_reviewers(
        self, repository: str, pr_number: int, reviewers: list[str]
    ) -> None:
        if not reviewers:
            return

        owner, repo = self._split_repo(repository)
        url = self._build_repo_api_url(
            owner, repo, 'pulls', str(pr_number), 'requested_reviewers'
        )

        try:
            await self._make_request(
                url,
                {'reviewers': reviewers},
                method=RequestMethod.POST,
            )
        except Exception as exc:  # pragma: no cover - log and continue
            logger.warning(
                'Failed to request Forgejo reviewers %s for %s/%s PR #%s: %s',
                reviewers,
                owner,
                repo,
                pr_number,
                exc,
            )

    async def get_pr_details(self, repository: str, pr_number: int) -> dict:  # type: ignore[override]
        owner, repo = self._split_repo(repository)
        url = self._build_repo_api_url(owner, repo, 'pulls', str(pr_number))
        response, _ = await self._make_request(url)
        return response

    async def is_pr_open(self, repository: str, pr_number: int) -> bool:  # type: ignore[override]
        pr_details = await self.get_pr_details(repository, pr_number)
        return (pr_details.get('state') or '').lower() == 'open'
