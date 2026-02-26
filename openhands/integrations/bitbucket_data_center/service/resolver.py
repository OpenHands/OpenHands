from datetime import datetime

from openhands.integrations.bitbucket_data_center.service.base import (
    BitbucketDCMixinBase,
)
from openhands.integrations.service_types import Comment


class BitbucketDCResolverMixin(BitbucketDCMixinBase):
    """
    Helper methods used for the Bitbucket Data Center Resolver
    """

    async def get_pr_title_and_body(
        self, owner: str, repo_slug: str, pr_id: int
    ) -> tuple[str, str]:
        """Get the title and body of a pull request.

        Args:
            owner: Project key (e.g. 'PROJ')
            repo_slug: Repository slug
            pr_id: Pull request ID

        Returns:
            A tuple of (title, body)
        """
        url = (
            f'{self.BASE_URL}/projects/{owner}/repos/{repo_slug}/pull-requests/{pr_id}'
        )
        response, _ = await self._make_request(url)
        title = response.get('title') or ''
        body = response.get('description') or ''
        return title, body

    async def get_pr_comments(
        self, owner: str, repo_slug: str, pr_id: int, max_comments: int = 10
    ) -> list[Comment]:
        """Get comments for a pull request.

        Uses the pull-requests/{id}/activities endpoint, filtering for
        COMMENTED actions — the same approach used by the resolver interface.

        Args:
            owner: Project key (e.g. 'PROJ')
            repo_slug: Repository slug
            pr_id: Pull request ID
            max_comments: Maximum number of comments to retrieve

        Returns:
            List of Comment objects ordered by creation date
        """
        url = f'{self.BASE_URL}/projects/{owner}/repos/{repo_slug}/pull-requests/{pr_id}/activities'
        all_raw: list[dict] = []

        params: dict = {'limit': 100, 'start': 0}
        while len(all_raw) < max_comments:
            response, _ = await self._make_request(url, params)
            for activity in response.get('values', []):
                if activity.get('action') == 'COMMENTED':
                    comment = activity.get('comment', {})
                    if comment:
                        all_raw.append(comment)

            if response.get('isLastPage', True):
                break

            next_start = response.get('nextPageStart')
            if next_start is None:
                break
            params = {'limit': 100, 'start': next_start}

        return self._process_raw_comments(all_raw, max_comments)

    def _process_raw_comments(
        self, comments: list, max_comments: int = 10
    ) -> list[Comment]:
        """Convert raw Bitbucket DC comment dicts to Comment objects."""
        all_comments: list[Comment] = []
        for comment_data in comments:
            # Bitbucket DC activities use epoch milliseconds for createdDate/updatedDate
            created_ms = comment_data.get('createdDate')
            updated_ms = comment_data.get('updatedDate')

            created_at = (
                datetime.fromtimestamp(created_ms / 1000)
                if created_ms is not None
                else datetime.fromtimestamp(0)
            )
            updated_at = (
                datetime.fromtimestamp(updated_ms / 1000)
                if updated_ms is not None
                else datetime.fromtimestamp(0)
            )

            author = (
                comment_data.get('author', {}).get('slug')
                or comment_data.get('author', {}).get('name')
                or 'unknown'
            )

            all_comments.append(
                Comment(
                    id=str(comment_data.get('id', 'unknown')),
                    body=self._truncate_comment(comment_data.get('text', '')),
                    author=author,
                    created_at=created_at,
                    updated_at=updated_at,
                    system=False,
                )
            )

        all_comments.sort(key=lambda c: c.created_at)
        return all_comments[-max_comments:]
