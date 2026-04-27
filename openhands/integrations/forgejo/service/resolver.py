from __future__ import annotations

from datetime import datetime

from openhands.integrations.forgejo.service.base import ForgejoMixinBase
from openhands.integrations.service_types import Comment


class ForgejoResolverMixin(ForgejoMixinBase):
    """Lightweight helpers used by resolver flows for Forgejo."""

    async def get_issue_title_and_body(
        self, repository: str, issue_number: int
    ) -> tuple[str, str]:
        owner, repo = self._split_repo(repository)
        url = self._build_repo_api_url(owner, repo, 'issues', str(issue_number))
        response, _ = await self._make_request(url)
        title = response.get('title') or ''
        body = response.get('body') or response.get('content') or ''
        return title, body

    async def get_issue_comments(
        self,
        repository: str,
        issue_number: int,
        max_comments: int = 20,
    ) -> list[Comment]:
        owner, repo = self._split_repo(repository)
        url = self._build_repo_api_url(
            owner, repo, 'issues', str(issue_number), 'comments'
        )
        per_page = min(max_comments, 50)
        params = {
            'page': '1',
            'limit': str(per_page),
            'order': 'desc',
        }

        response, _ = await self._make_request(url, params)
        raw_comments = response if isinstance(response, list) else []

        comments: list[Comment] = []
        for payload in raw_comments:
            comment = self._to_comment(payload)
            if comment is not None:
                comments.append(comment)

        comments.sort(key=lambda c: c.created_at)
        return comments[-max_comments:]

    async def get_pr_comments(
        self,
        repository: str,
        pr_number: int,
        max_comments: int = 50,
    ) -> list[Comment]:
        owner, repo = self._split_repo(repository)
        url = self._build_repo_api_url(owner, repo, 'pulls', str(pr_number), 'comments')
        per_page = min(max_comments, 50)
        params = {
            'page': '1',
            'limit': str(per_page),
            'order': 'desc',
        }

        response, _ = await self._make_request(url, params)
        raw_comments = response if isinstance(response, list) else []

        comments: list[Comment] = []
        for payload in raw_comments:
            comment = self._to_comment(payload)
            if comment is not None:
                comments.append(comment)

        comments.sort(key=lambda c: c.created_at)
        return comments[-max_comments:]

    def _to_comment(self, payload: dict | None) -> Comment | None:
        if not isinstance(payload, dict):
            return None
        body = payload.get('body') or ''
        author = (payload.get('user') or {}).get('login') or 'unknown'
        created_at = self._parse_datetime(payload.get('created_at'))
        updated_at = self._parse_datetime(payload.get('updated_at'))

        return Comment(
            id=str(payload.get('id', 'unknown')),
            body=body,
            author=author,
            created_at=created_at,
            updated_at=updated_at,
            system=payload.get('void', False),
        )

    def _parse_datetime(self, value: str | None) -> datetime:
        if not value:
            return datetime.fromtimestamp(0)
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return datetime.fromtimestamp(0)
